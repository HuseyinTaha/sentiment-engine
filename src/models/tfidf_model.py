from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel, ModelConfig, ModelResult

logger = logging.getLogger(__name__)

class TFIDFModel(BaseModel):
    """
    TF-IDF + Logistic Regression tabanlı sentiment sınıflandırıcı.

    Avantajları:
        - GPU gerektirmez
        - Hızlı eğitim (saniyeler)
        - Yorumlanabilir (hangi kelime etkili?)
        - Baseline model olarak ideal

    Dezavantajları:
        - Kelime sırası bilgisi yok
        - Bağlam anlayışı yok ("değil güzel" vs "güzel değil")
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._label_encoder = LabelEncoder()
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """TF-IDF + Logistic Regression pipeline'ı oluştur."""
        self._model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000,
                ngram_range=(1,2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=self.config.random_seed,
                n_jobs=-1,
            )),
        ])
        logger.info("TF-IDF pipeline oluşturuldu")

    
    def train(
        self, 
        X_train: list[str], 
        y_train: list[int], 
        X_val: Optional[list[str]] = None, 
        y_val: Optional[list[int]] = None,
    ) -> dict:
        """Pipeline'ı eğit, metrikleri döndür."""
        logger.info(
            "TF-IDF modeli eğitiliyor: %d örnek", len(X_train)
        )

        self._model.fit(X_train, y_train)
        self.is_trained = True

        metrics = {"train_samples": len(X_train)}

        train_preds = self._model.predict(X_train)
        from sklearn.metrics import accuracy_score
        metrics["train_accuracy"] = accuracy_score(y_train, train_preds)

        logger.info(
            "Eğitim tamamlandı. Train accuracy: %.4f",
            metrics["train_accuracy"]
        )

        if X_val is not None and y_val is not None:
            val_result = self.evaluate(X_val, y_val)
            metrics["val_accuracy"] = val_result["accuracy"]
            metrics["val_f1"] = val_result["f1_macro"]
            logger.info(
                "Val accuracy: %.4f | Val F1: %.4f",
                metrics["val_accuracy"], metrics["val_f1"]
            )
        
        return metrics
    
    def predict(self, texts: list[str]) -> ModelResult:
        """Metinler için tahmin yap."""
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi.")
        
        preds = self._model.predict(texts)
        probs = self._model.predict_proba(texts)

        return ModelResult(
            predictions=preds,
            probabilities=probs,
        )
    
    def save(self, path: str | Path) -> None:
        """Modeli pickle ile kaydet."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        
        self.config.save(path.with_suffix(".json"))
        logger.info("Model kaydedildi: %s", path)

    def load(self, path: str | Path) -> None:
        """Kaydedilmiş modeli yükle."""
        path = Path(path)

        with open(path, "rb") as f:
            self._model = pickle.load(f)
        
        self.is_trained = True
        logger.info("Model yüklendi: %s", path)
    
    def get_top_features(
        self, 
        n: int = 20,
        class_idx: int = 2,
    ) -> list[tuple[str, float]]:
        """
        En etkili kelimeleri döndür.
        Modeli yorumlamak için — CV'de gösterilebilir!
        """
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi.")

        vectorizer = self._model.named_steps["tfidf"]
        classifier = self._model.named_steps["classifier"]

        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[class_idx]

        top_idx = np.argsort(coefficients)[-n:][::-1]

        return [
            (feature_names[i], coefficients[i])
            for i in top_idx
        ]