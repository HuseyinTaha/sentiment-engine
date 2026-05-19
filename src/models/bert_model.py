from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from .base import BaseModel, ModelConfig, ModelResult

logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """
    PyTorch Dataset — BERT için metin ve label çiftleri.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:    
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTModel(BaseModel):
    """
    HuggingFace BERT tabanlı sentiment sınıflandırıcı.

    Türkçe: dbmdz/bert-base-turkish-cased
    İngilizce: distilbert-base-uncased
    """

    def __init__(
        self, 
        config: ModelConfig,
        model_name_or_path: str = "dbmdz/bert-base-turkish-cased",
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = self._get_device()
        self._tokenizer = None
        self._label_map = {0: 0, 1: 1, 2: 2}             
        self._reverse_label_map = {0: 0, 1: 1, 2: 2}     
        
        super().__init__(config)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("GPU kullanılıyor: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        logger.warning("GPU bulunamadı, CPU kullanılıyor")
        return torch.device("cpu")
    
    def _setup_model(self) -> None:
        """Tokenizer ve modeli yükle."""
        logger.info("'%s' yükleniyor...", self.model_name_or_path)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.config.num_classes,
            ignore_mismatched_sizes=True,
        )

        self._model.to(self.device)
        logger.info("Model GPU/CPU'ya taşındı")

    def train(
        self, 
        X_train: list[str], 
        y_train: list[int], 
        X_val: Optional[list[str]] = None, 
        y_val: Optional[list[int]] = None,
    ) -> dict:
        self._setup_model()

        unique_labels = sorted(set(y_train))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_train_mapped = [label_map[y] for y in y_train]
        self._label_map = label_map
        self._reverse_label_map = {v: k for k, v in label_map.items()}

        train_dataset = SentimentDataset(
            X_train, y_train_mapped,
            self._tokenizer,
            self.config.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * self.config.num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        metrics = {"epochs": []}

        for epoch in range(self.config.num_epochs):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler
            )

            epoch_metrics = {"epoch": epoch+1, "train_loss": train_loss}

            if X_val and y_val:
                y_val_mapped = [label_map.get(y,y) for y in y_val]
                val_metrics = self._validate(X_val, y_val_mapped)
                epoch_metrics.update(val_metrics)
                logger.info(
                    "Epoch %d/%d — Loss: %.4f | Val Acc: %.4f",
                    epoch + 1, self.config.num_epochs,
                    train_loss, val_metrics["val_accuracy"],
                )
            else:
                logger.info(
                    "Epoch %d/%d — Loss: %.4f",
                    epoch + 1, self.config.num_epochs, train_loss,
                )
            
            metrics["epochs"].append(epoch_metrics)
        
        self.is_trained = True
        return metrics
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scheduler,
    ) -> float:
        """Tek epoch eğitimi — loss döndür."""
        self._model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
        
        return total_loss/len(loader)

    def _validate(
        self,
        X_val: list[str],
        y_val: list[int],
    ) -> dict:
        """Validation seti üzerinde değerlendir."""
        self._model.eval()
        raw_preds, _ = self._predict_raw(X_val)

        from sklearn.metrics import accuracy_score, f1_score
        return {
            "val_accuracy": accuracy_score(y_val, raw_preds),
            "val_f1": f1_score(y_val, raw_preds, average="macro", zero_division=0),
        }
    
    def predict(self, texts: list[str]) -> ModelResult:
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi")
        
        raw_preds, raw_probs = self._predict_raw(texts)

        preds = np.array([
            self._reverse_label_map.get(p,p) for p in raw_preds
        ])

        return ModelResult(
            predictions=preds,
            probabilities=np.array(raw_probs),
        )
    

    def _predict_raw(self, texts: list[str]) -> tuple[list[int], list[list[float]]]:
        """İç kullanım — mapped label ve olasılıkları döndürür."""
        self._model.eval()
        all_preds = []
        all_probs = []

        dataset = SentimentDataset(
            texts,
            [0] * len(texts),
            self._tokenizer,
            self.config.max_length,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy().tolist())
            
        return all_preds, all_probs


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        self.config.save(path / "sentiment_config.json") 

        import json
        with open(path / "label_map.json", "w") as f:
            json.dump(self._label_map, f)

        logger.info("BERT modeli kaydedildi: %s", path)
    
    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)
        self._model.to(self.device)

        import json
        label_map_path = path / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                raw = json.load(f)
                self._label_map = {int(k): int(v) for k, v in raw.items()}
                self._reverse_label_map = {v: k for k, v in self._label_map.items()}

        self.is_trained = True
        logger.info("BERT modeli yüklendi: %s", path)
    
















