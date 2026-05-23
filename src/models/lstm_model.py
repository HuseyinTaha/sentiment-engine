from __future__ import annotations
from pathlib import Path
from typing import Optional
from collections import Counter
import logging
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .base import BaseModel, ModelConfig, ModelResult

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Kelimeleri sayılara, sayıları kelimelere çevirir.
    
    Örnek:
        vocab = Vocabulary()
        vocab.build(["iyi ürün", "kötü ürün"])
        vocab.encode("iyi ürün")  → [2, 3]
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, max_vocab_size: int = 20_000) -> None:
        self.max_vocab_size = max_vocab_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.size = 0

    def build(self, texts: list[str], min_freq: int=2) -> None:
        """
        Metinlerden kelime hazinesi oluştur.
        
        Args:
            texts: Eğitim metinleri
            min_freq: Minimum kelime frekansı
        """
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        for word, freq in counter.most_common(self.max_vocab_size):
            if freq < min_freq:
                break
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
            
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.size = len(self.word2idx)

        logger.info(
            "Vocabulary oluşturuldu: %d kelime (min_freq=%d)",
            self.size, min_freq
        )

    def encode(
        self,
        text: str,
        max_length: int = 128,    
    ) -> list[int]:
        """Metni token ID listesine çevir."""
        tokens = text.split()[:max_length]
        ids = [
            self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in tokens
        ]
        ids += [0] * (max_length - len(ids))
        return ids
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)
        logger.info("Vocabulary kaydedildi: %s", path)
    
    def load(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.size = len(self.word2idx)
        logger.info("Vocabulary yüklendi: %d kelime", self.size)
    

class LSTMDataset(Dataset):
    """LSTM için metin ve label çiftleri."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self)  -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        ids = self.vocab.encode(
            str(self.texts[idx]),
            self.max_length,
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM ile metin sınıflandırma.
    
    Akış:
        input_ids → Embedding → BiLSTM → Dropout → Linear → logits
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )        

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))

        lstm_out, (hidden, _) = self.lstm(embedded)

        hidden_cat = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )
        hidden_cat = self.dropout(hidden_cat)
        logits = self.classifier(hidden_cat)
        return logits


class LSTMModel(BaseModel):
    """
    Bidirectional LSTM tabanlı sentiment sınıflandırıcı.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._vocab: Optional[Vocabulary] = None
        self.device = self._get_device()
        super().__init__(config)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            logger.info("GPU kullanılıyor")
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_model(self, vocab_size: int) -> None:
        self._model = LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=self.config.num_classes,
        ).to(self.device)

    def train(
        self,
        X_train: list[str],
        y_train: list[int],
        X_val: Optional[list[str]] = None,
        y_val: Optional[list[int]] = None,
    ) -> dict:
        
        self._vocab = Vocabulary()
        self._vocab.build(X_train)

        unique_labels = sorted(set(y_train))
        self._label_map = {old: new for new, old in enumerate(unique_labels)}
        self._reverse_label_map = {v: k for k, v in self._label_map.items()}
        y_train_mapped = [self._label_map[y] for y in y_train]

        self._build_model(self._vocab.size)
        logger.info(
            "LSTM oluşturuldu: vocab=%d, params=%d",
            self._vocab.size,
            sum(p.numel() for p in self._model.parameters()), 
        )

        train_dataset = LSTMDataset(
            X_train, y_train_mapped,
            self._vocab, self.config.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
        )

        metrics = {"epochs": []}

        for epoch in range(self.config.num_epochs):
            self._model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self._model(input_ids)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), 1.0
                )
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            }

            if X_val and y_val:
                y_val_mapped = [self._label_map.get(y, y) for y in y_val]
                val_acc = self._validate(X_val, y_val_mapped)
                epoch_metrics["val_accuracy"] = val_acc
                logger.info(
                    "Epoch %d/%d — Loss: %.4f | Val Acc: %.4f",
                    epoch + 1, self.config.num_epochs,
                    avg_loss, val_acc,
                )
            else:
                logger.info(
                    "Epoch %d/%d — Loss: %.4f",
                    epoch + 1, self.config.num_epochs, avg_loss,
                )

            metrics["epochs"].append(epoch_metrics)

        self.is_trained = True
        return metrics
    
    def _validate(
        self, X_val: list[str], y_val: list[int]
    ) -> float:
        from sklearn.metrics import accuracy_score
        preds, _ = self._predict_raw(X_val)
        return accuracy_score(y_val, preds)
    
    def _predict_raw(self, texts: list[str]) -> tuple[list[int], list[list[float]]]:
        self._model.eval()
        all_preds = []
        all_probs = []

        dummy_labels = [0] * len(texts)
        dataset = LSTMDataset(
            texts, dummy_labels,
            self._vocab, self.config.max_length,
        )
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=0,
        )

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                logits = self._model(input_ids)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy().tolist())
        
        return all_preds, all_probs

    
    def predict(self, texts: list[str]) -> ModelResult:
        if not self.is_trained:
            raise RuntimeError("Model eğitilmedi.")
        
        raw_preds, raw_probs = self._predict_raw(texts)

        preds = np.array([
            self._reverse_label_map.get(p, p) for p in raw_preds
        ])

        return ModelResult(
            predictions=preds,
            probabilities=np.array(raw_probs),
        )
    
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "lstm.pt")
        self._vocab.save(path / "vocab.json")
        self.config.save(path / "config.json")

        with open(path / "label_map.json", "w") as f:
            json.dump(self._label_map, f)

        logger.info("LSTM kaydedildi: %s", path)
    
    def load(self, path: str | Path) -> None:
        path = Path(path)

        self._vocab = Vocabulary()
        self._vocab.load(path / "vocab.json")

        with open(path / "label_map.json") as f:
            raw = json.load(f)
            self._label_map = {int(k): int(v) for k, v in  raw.items()}
            self._reverse_label_map = {v: k for k, v in self._label_map.items()}
        
        self._build_model(self._vocab.size)
        self._model.load_state_dict(
            torch.load(path / "lstm.pt", map_location=self.device)
        )

        self.is_trained = True
        logger.info("LSTM yüklendi: %s", path)



