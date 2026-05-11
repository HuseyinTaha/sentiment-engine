from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessConfig:
    """
    Preprocessing adımlarını açıp kapatmak için config.
    Her alan varsayılan olarak True — istersen kapatırsın.
    """
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = True
    remove_emoji: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True
    remove_stopwords: bool = True
    apply_stemming: bool = False
    min_token_length: int = 2
    max_text_length: Optional[int] = None

class BasePreprocessor(ABC):
    """
    Tüm preprocessor'ların türeyeceği abstract temel sınıf.

    Alt sınıflar implement etmek zorunda:
        - clean_text()     : dile özgü temizleme
        - tokenize()       : dile özgü tokenization
        - remove_stopwords_impl() : dile özgü stopword listesi
    """

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self._stopwords: set[str] = set()
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """
        Dile özgü kaynakları yükle.
        (stopword listesi, stemmer, morfoloji aracı vb.)
        """
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Ham metni temizle — dile özgü kurallar burada."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Metni token listesine çevir."""
    
    @abstractmethod
    def remove_stopwords_impl(self, tokens:list[str]) -> list[str]:
        """Dile özgü stopword'leri kaldır."""

    def process(self, text: str) -> str:
        """
        Tam preprocessing pipeline'ı çalıştır.

        Config'e göre adımları sırayla uygular.
        Alt sınıflar bunu override etmez — sadece
        abstract metodları implement eder.
        """

        if not isinstance(text, str):
            logger.warning("String olmayan input: %s", type(text))
            return ""
        
        if self.config.max_text_length:
            text = text[:self.config.max_text_length]

        text = self._remove_urls(text)
        text = self._remove_mentions(text)
        text = self._remove_hashtag_symbols(text)
        text = self._remove_emojis(text)

        text = self.clean_text(text)

        if self.config.lowercase:
            text = text.lower()
        
        tokens = self.tokenize(text)

        if self.config.remove_stopwords:
            tokens = self.remove_stopwords_impl(tokens)

        tokens = [
            t for t in tokens
            if len(t) >= self.config.min_token_length
        ]

        result = " ".join(tokens)
        logger.debug("İşlendi: '%s' → '%s'", text[:50], result[:50])
        return result
    
    def process_batch(self, texts: list[str]) -> list[str]:
        """Bir liste metni toplu işle."""
        return [self.process(t) for t in texts]
    
    def _remove_urls(self, text: str) -> str:
        if not self.config.remove_urls:
            return text
        return re.sub(r"https?://\S+|www\.\S+", "", text)
    
    def _remove_mentions(self, text: str) -> str:
        if not self.config.remove_mentions:
            return text
        return re.sub(r"@\w+", "", text)
    
    def _remove_hashtag_symbols(self, text: str) -> str:
        if not self.config.remove_hashtags:
            return text
        return re.sub(r"#(\w+)", r"\1", text)
    
    def _remove_emojis(self, text: str) -> str:
        if not self.config.remove_emoji:
            return text
        try:
            import emoji
            return emoji.replace_emoji(text, replace="")
        except ImportError:
            return re.sub(r"[^\x00-\x7F\u00C0-\u024F]+", " ", text)
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"config={self.config})"
        )