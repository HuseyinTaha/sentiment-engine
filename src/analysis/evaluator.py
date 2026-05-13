from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Tek bir modelin değerlendirme sonuçlarını taşır.
    """
    model_name: str
    accuracy: float
    f1_macro: float
    f1_weighted: float
    report: str
    predictions: np.ndarray
    true_labels: np.ndarray
    error_indices: list[int] = field(default_factory=list)
    eroor_texts: list[str] = field(default_factory=list)


class ModelEvaluator:
    """
    Tek bir modeli derinlemesine değerlendirir.

    - Metrik hesaplama
    - Confusion matrix
    - Hata analizi (hangi metinlerde yanılıyor?)
    - Güven skoru analizi
    """

    LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model: BaseModel,
        output_dir: str | Path = "outputs/analysis",
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    
    def evaluate(
        self,
        X_test: list[str],
        y_test: list[int],
        model_name: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Modeli değerlendir, EvaluationResult döndür.
        """
        name = model_name or self.model.config.model_name
        logger.info("'%s' değerlendiriliyor...", name)

        metrics = self.model.evaluate(X_test, y_test)
        result = self.model.predict(X_test)
        preds = result.predictions

        errors = self._find_errors(X_test, y_test, preds)

        eval_result = EvaluationResult(
            model_name=name,
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            f1_weighted=metrics["f1_weighted"],
            report=metrics["report"],
            predictions=preds,
            true_labels=np.array(y_test),
            error_indices=errors["indices"],
            eroor_texts=errors["texts"],
        )

        logger.info(
            "'%s' — Accuracy: %.4f | F1: %.4f | Hata: %d/%d",
            name, eval_result.accuracy,
            eval_result.f1_macro,
            len(errors["indices"]), len(y_test),
        )

        return eval_result
    
    def _find_errors(
        self,
        texts: list[str],
        true_labels: list[int],
        predictions: np.ndarray,
    ) -> dict:
        """Yanlış tahmin edilen örnekleri bul."""
        indices = []
        error_texts = []

        for i, (true, pred) in enumerate(zip(true_labels, predictions)):
            if true != pred:
                indices.append(i)
                error_texts.append(
                    f"[{i}] Gerçek: {self.LABEL_NAMES.get(true, true)} | "
                    f"Tahmin: {self.LABEL_NAMES.get(pred, pred)}\n"
                    f"Metin: {texts[i][:100]}"
                )

        return {"indices": indices, "texts": error_texts}
    
    def plot_confusion_matrix(
        self,
        eval_result: EvaluationResult,
        save: bool = True,
    ) -> Path:
        """Confusion matrix çiz ve kaydet."""
        present_labels = sorted(set(eval_result.true_labels))
        label_names = [
            self.LABEL_NAMES.get(l, str(l)) for l in present_labels
        ]

        cm = confusion_matrix(
            eval_result.true_labels,
            eval_result.predictions,
            labels=present_labels,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_names,
        )
        disp.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )

        ax.set_title(
            f"{eval_result.model_name}\n"
            f"Accuracy: {eval_result.accuracy:.4f} | "
            f"F1: {eval_result.f1_macro:.4f}",
            fontsize=13,
            pad=15,
        )

        plt.tight_layout()

        path = self.output_dir / f"cm_{eval_result.model_name}.png"

        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix kaydedildi: %s", path)
        
        plt.close(fig)
        return path
    

    def print_error_analysis(
        self,
        eval_result: EvaluationResult,
        n: int = 5,
    ) -> None:
        """En ilginç hataları konsola yazdır."""
        errors = eval_result.eroor_texts

        print(f"\n{'='*60}")
        print(f"HATA ANALİZİ — {eval_result.model_name}")
        print(f"Toplam hata: {len(errors)}/{len(eval_result.true_labels)}")
        print(f"{'='*60}")

        for error in errors[:n]:                              
            print(f"\n{error}")
            print("-" * 40)