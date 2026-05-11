from __future__ import annotations
from typing import Optional
import re
import logging

from .base import BasePreprocessor, PreprocessConfig

logger = logging.getLogger(__name__)

class TurkishPreprocessor(BasePreprocessor):
    """
    Türkçe metinler için preprocessing.

    Türkçe'ye özgü:
        - Özel karakter normalizasyonu (I → ı, İ → i)
        - Zemberek ile morfolojik analiz (opsiyonel)
        - Snowball Türkçe stemmer (fallback)
        - NLTK Türkçe stopword listesi + özel ekler
    """

    def __init__(
        self, 
        config: Optional[PreprocessConfig] = None,
        use_zemberek: bool = False,
    ) -> None:
        self.use_zemberek = use_zemberek
        self._stemmer = None
        self._zemberek = None
        super().__init__(config)

    def _setup(self) -> None:
        """Türkçe kaynakları yükle: stemmer, stopwords."""
        try:
            from snowballstemmer import TurkishStemmer
            self._stemmer = TurkishStemmer()
            logger.info("Snowball Türkçe stemmer yüklendi")
        
        except ImportError:
            logger.warning("snowballstemmer bulunamadı, stemming devre dışı")

        if self.use_zemberek:
            try:
                from zemberek import TurkishMorphology
                self._zemberek = TurkishMorphology.create_with_defaults()
                logger.info("Zemberek morfoloji motoru yüklendi")
            except Exception as e:
                logger.warning("Zemberek yüklenemedi: %s", e)
                self._zemberek = None
        
        self._stopwords = self._load_stopwords()
        logger.info("%d Türkçe stopword yüklendi", len(self._stopwords))
    
    def _load_stopwords(self) -> set[str]:
        """NLTK + özel Türkçe stopword listesini birleştir."""
        words = set()

        try:
            from nltk.corpus import stopwords
            words.update(stopwords.words("turkish"))
        except Exception:
            logger.warning("NLTK Türkçe stopword yüklenemedi")

        custom = {
            "bir", "bu", "şu", "o", "ve", "ile", "de",
            "da", "ki", "mi", "mu", "mü", "mı", "ya",
            "ama", "fakat", "lakin", "çünkü", "eğer",
            "için", "gibi", "kadar", "daha", "en", "çok",
            "hiç", "her", "bazı", "tüm", "bütün", "hangi",
            "ne", "nasıl", "neden", "nerede", "kim", "ben",
            "sen", "biz", "siz", "onlar", "var", "yok",
        }

        words.update(custom)
        return words
    
    def clean_text(self, text: str) -> str:
        """Türkçe'ye özgü metin temizleme."""
        text = text.replace("I", "ı").replace("İ", "i")
        # Tekrar eden karakterleri normalize et          
        # "çooook" → "çok", "harikaaa" → "harika"
        text = re.sub(r"(.)\1{2,}", r"\1", text)

        if self.config.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text) 
        
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
    def tokenize(self, text: str) -> list[str]:
        """Metni kelimelerine ayır."""
        return text.split()
    
    def remove_stopwords_impl(self, tokens: list[str]) -> list[str]:
        """Türkçe stopword'leri kaldır."""
        return [t for t in tokens if t not in self._stopwords]
    
    def stem(self, token: str) -> str:
        """Tek kelimeyi stemle."""
        if self._zemberek:
            try:
                results = self._zemberek.analyze(token)
                if results:
                    return results[0].get_stem()
            except Exception:
                pass
        
        if self._stemmer:
            return self._stemmer.stemWord(token)
        
        return token
    
    def apply_stemming_to_tokens(
            self, tokens: list[str]
    ) -> list[str]:
        if not self.config.apply_stemming:
            return tokens
        return [self.stem(t) for t in tokens]