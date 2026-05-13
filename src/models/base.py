from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    Tüm modeller için ortak config.
    Her model bunu genişletebilir.
    """
    model_name: str
    num_classes: int = 3
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    random_seed: int = 42
    output_dir: str = "models"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info("Config kaydedildi: %s", path)
    
    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(**data)
    

@dataclass
class ModelResult:
    """
    Model tahmin sonuçlarını taşır.
    """
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None


class BaseModel(ABC):
    """
    Tüm modellerin türeyeceği abstract temel sınıf.
    """
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.is_trained: bool = False
        self._model = None
        self._set_seed()
    
    def _set_seed(self) -> None:
        import random
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        try:
            import torch
            torch.manual_seed(self.config.random_seed)
        except ImportError:
            pass
    
    @abstractmethod
    def train(
        self, 
        X_train: list[str],
        y_train: list[int],
        X_val: Optional[list[str]] = None,
        y_val: Optional[list[int]] = None,
    ) -> dict:
        """
        Modeli eğit.

        Returns:
            dict: Eğitim metrikleri {"loss": ..., "accuracy": ...}
        """
    
    @abstractmethod
    def predict(self, texts: list[str]) -> ModelResult:
        """Tahmin yap, ModelResult döndür."""
    
    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Modeli diske kaydet."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Modeli diskten yükle."""
    
    def evaluate(
        self,
        X_test: list[str],
        y_test: list[str],
    ) -> dict:
        """
        Model performansını değerlendir.
        Abstract değil — tüm modeller için aynı metrik hesaplaması.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model henüz eğitilmedi. Önce train() çağır."
            )
        
        from sklearn.metrics import(
            accuracy_score,
            f1_score,
            classification_report,
        )

        result = self.predict(X_test)
        preds = result.predictions

        present_labels = sorted(set(y_test))
        label_names = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
        target_names = [
            label_names[l] for l in present_labels
        ]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro", labels=present_labels),
            "f1_weighted": f1_score(y_test, preds, average="weighted", labels=present_labels),
            "report": classification_report(
                y_test, preds,
                labels=present_labels,
                target_names=target_names,
                zero_division=0,
            ),
            "num_classes": len(present_labels),
        }

        logger.info(
            "Sınıflar: %s | Accuracy: %.4f | F1 (macro): %.4f",
            target_names,
            metrics["accuracy"], metrics["f1_macro"]
        )

        return metrics
    

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"trained={self.is_trained}, "
            f"config={self.config.model_name})"
        )