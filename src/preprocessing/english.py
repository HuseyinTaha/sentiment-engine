from __future__ import annotations
from typing import Optional
import re 
import logging

from .base import BasePreprocessor, PreprocessConfig

logger = logging.getLogger(__name__)

class EnglishPreprocessor(BasePreprocessor):
    """
    İngilizce metinler için preprocessing.

    İngilizce'ye özgü:
        - WordNet Lemmatizer
        - NLTK İngilizce stopword listesi
        - Contraction genişletme (don't → do not)
    """

    def __init__(
        self, 
        config: Optional[PreprocessConfig] = None,
    ) -> None:
        self._lemmatizer = None
        super().__init__(config)

    def _setup(self) -> None:
        """İngilizce kaynakları yükle."""
        try:
            from nltk.stem import WordNetLemmatizer
            import nltk
            nltk.download("wordnet", quiet=True)
            self._lemmatizer = WordNetLemmatizer()
            logger.info("WordNet Lemmatizer yüklendi")
        except Exception as e:
            logger.warning("Lemmatizer yüklenemedi: %s", e)
        
        self._stopwords = self._load_stopwords()
        logger.info("%d İngilizce stopword yüklendi", len(self._stopwords))

    def _load_stopwords(self) -> set[str]:
        try:
            from nltk.corpus import stopwords
            words = set(stopwords.words("english"))
            keep = {"not", "no", "nor", "neither", "never", 
                "none", "nothing", "nobody", "nowhere",
                "but", "however", "although", "yet"}
            words = words - keep

            return words
        except Exception:
            logger.warning("NLTK İngilizce stopword yüklenemedi")
            return set()
        
    def clean_text(self, text: str) -> str:
        """İngilizce'ye özgü metin temizleme."""
        text = self._expand_contractions(text)
        text = re.sub(r"(.)\1{2,}", r"\1", text)

        if self.config.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def tokenize(self, text: str) -> list[str]:
        """Metni token listesine ayır."""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            return text.split()
    
    def remove_stopwords_impl(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._stopwords]
    
    def lemmatize(self, token: str) -> str:
        """Tek kelimeyi lemmatize et."""
        if self._lemmatizer:
            return self._lemmatizer.lemmatize(token)
        return token
    
    def _expand_contractions(self, text: str) -> str:
        """
        İngilizce kısaltmaları genişlet.
        don't → do not, I'm → I am
        """
        contractions = {
            "don't": "do not", "won't": "will not",
            "can't": "cannot", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not",
            "haven't": "have not", "hadn't": "had not",
            "doesn't": "does not", "didn't": "did not",
            "i'm": "i am", "i've": "i have",
            "i'll": "i will", "i'd": "i would",
            "it's": "it is", "that's": "that is",
            "there's": "there is", "they're": "they are",
            "we're": "we are", "you're": "you are",
            "he's": "he is", "she's": "she is",
        }

        for contraction, expanded in contractions.items():
            text = re.sub(
                contraction, expanded, text, flags=re.IGNORECASE
            )

        return text


    

        
        