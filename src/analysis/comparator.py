from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from .evaluator import EvaluationResult

logger = logging.getLogger(__name__)

class ResultComparator:
    """
    Birden fazla modelin sonuçlarını karşılaştırır.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs/analysis",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[EvaluationResult] = []

    def add_result(self, result: EvaluationResult) -> None:
        self.results.append(result)
        logger.info("Sonuç eklendi: %s", result.model_name)
    
    def print_comparison(self) -> None:
        """Tablo formatında karşılaştırma yaz."""
        print(f"\n{'='*60}")
        print("MODEL KARŞILAŞTIRMASI")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
        print("-" * 55)

        for r in sorted(
            self.results, key=lambda x: x.f1_macro, reverse=True
        ):
            print(
                f"{r.model_name:<20} "
                f"{r.accuracy:>10.4f} "
                f"{r.f1_macro:>10.4f} "
                f"{r.f1_weighted:>12.4f}"
            )
        
        best = max(self.results, key=lambda x: x.f1_macro)
        print(f"\n🏆 En iyi model: {best.model_name} (F1: {best.f1_macro:.4f})")

    def plot_comparison(self, save: bool = True) -> Path:
        """Bar chart ile model karşılaştırması."""
        names = [r.model_name for r in self.results]
        accuracies = [r.accuracy for r in self.results]
        f1_macros = [r.f1_macro for r in self.results]
        f1_weighted = [r.f1_weighted for r in self.results]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10,6))

        bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color="#4C72B0")
        bars2 = ax.bar(x, f1_macros, width, label="F1 Macro", color="#DD8452")
        bars3 = ax.bar(x + width, f1_weighted, width, label="F1 Weighted", color="#55A868")

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9,
                )

        ax.set_ylabel("Skor")
        ax.set_title("Model Karşılaştırması")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        path = self.output_dir / "model_comparison.png"
        if save:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Karşılaştırma grafiği kaydedildi: %s", path)
        
        plt.close(fig)
        return path
    