from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    name: str
    language: str
    num_samples: int = 0
    labels: list[str] = field(default_factory=list)
    source_path: Optional[Path] = None

    def __post_init__(self):
        if self.language not in ("tr", "en", "multilingual"):
            raise ValueError(
                f"Geçersiz dil: '{self.language}'. "
                f"Beklenen: 'tr', 'en' veya 'multilingual'"
            )
    
class BaseDatasetLoader(ABC):
    """
    Tüm dataset loader'ların türeyeceği abstract temel sınıf.

    Alt sınıflar `load()` ve `validate()` metodlarını
    implement etmek zorundadır.
    """                          

    def __init__(self, data_dir: str| Path) -> None:
        self.data_dir = Path(data_dir)
        self._dataset_info: Optional[DatasetInfo] = None
        self.COLUMN_ALIASES: dict[str, list[str]] = {   # (1)
            "text":  ["sentence", "review", "content", "comment", "tweet"],
            "label": ["sentiment", "target", "class", "polarity"],
        }

        if not self.data_dir.exists():      # (20)
            logger.warning(
                "Veri dizini bulunamadı: %s. Oluşturuluyor...",
                self.data_dir
            )
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def load(self) -> dict:
        """Ham veriyi yükle ve döndür"""
    
    @abstractmethod
    def validate(self) -> bool:
        """Yüklenen verinin bütünlüğünü kontrol et."""
    
    @property
    def dataset_info(self) -> Optional[DatasetInfo]:
        return self._dataset_info

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{self.data_dir}')"
        )

from datasets import load_dataset as hf_load_dataset
import pandas as pd

class HuggingFaceLoader(BaseDatasetLoader):
    """
    HuggingFace Hub'dan dataset yükleyen sınıf.

    Örnek kullanım:
        loader = HuggingFaceLoader(
            dataset_name="mteb/tweet_sentiment_multilingual",
            language="multilingual",
            data_dir="data/raw"
        )
        data = loader.load()
    """

    def __init__(
        self,
        dataset_name: str,
        language: str,
        data_dir: str | Path = "data/raw",
        subset: Optional[str] = None,
        max_samples: Optional[int] = None,
                 
    ) -> None:
        super().__init__(data_dir)
        self.dataset_name = dataset_name
        self.language = language
        self.subset = subset
        self.max_samples = max_samples
        self._data: Optional[pd.DataFrame] = None

        logger.info(
            "HuggingFaceLoader oluşturuldu: '%s'", dataset_name
        )

    def normalize_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Farklı dataset'lerdeki kolon isimlerini standart hale getirir.
        'sentence' → 'text', 'sentiment' → 'label' gibi.

        Args:
            df: Ham DataFrame

        Returns:
            Kolon isimleri normalize edilmiş DataFrame
        """
        rename_map = {}  
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            if standard_name in df.columns:               
                continue

            for alias in aliases:
                if alias in df.columns:
                    rename_map[alias] = standard_name
                    logger.info(
                        "Kolon normalize edildi: '%s' → '%s'",
                        alias, standard_name
                    )
                    break 
        
        if rename_map:
            df = df.rename(columns=rename_map)

        keep = [c for c in ["text", "label"] if c in df.columns]
        return df[keep]
    

    def load(self) -> dict:
        """
        HuggingFace'den veriyi indir, DataFrame'e çevir.

        Returns:
            dict: {"train": DataFrame, "test": DataFrame}
        """
        logger.info("'%s' yükleniyor...", self.dataset_name)

        try:
            raw = hf_load_dataset(
                self.dataset_name,
                self.subset,
            )
        except Exception as e:
            logger.error("Dataset yüklenemedi: %s", e)
            raise

        result = {}

        for split in raw.keys():
            df = raw[split].to_pandas()
            if self.max_samples:
                df = df.head(self.max_samples)
            
            result[split] = df
            normalized_df = self.normalize_columns(df)
            normalizer = LabelNormalizer()
            normalized_df["label"] = normalizer.normalize(
                normalized_df["label"]
            )
            result[split] = normalized_df
            
            logger.info(
                "%s split: %d satır yüklendi", split, len(df)
            )
        
        self._data = result.get("train")
        self._dataset_info = DatasetInfo(
            name=self.dataset_name,
            language=self.language,
            num_samples=len(self._data) if self._data is not None else 0,
        )

        return result
    
    def validate(self) -> bool:
        """
        Yüklenen verinin kullanılabilir olduğunu kontrol et.

        Returns:
            bool: Veri geçerliyse True
        """

        if self._data is None:
            logger.warning("Veri henüz yüklenmedi.")
            return False
        
        if len(self._data) == 0:
            logger.warning("Dataset boş.")
            return False
        
        required_columns = {"text", "label"}
        actual_columns = set(self._data.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            logger.error("Eksik kolonlar: %s", missing)
            return False

        null_counts = self._data[["text", "label"]].isnull().sum()
        if null_counts.any():
            logger.warning("Null değerler var:\n%s", null_counts)
        
        logger.info("Validasyon başarılı.")
        return True

@dataclass
class LabelNormalizer:
    """
    Farklı formatlardaki label'ları standart sayısal forma çevirir.

    Standart format:
        0 = negative
        1 = neutral
        2 = positive
    """
    LABEL_MAPS: dict = None

    def __post_init__(self):
        if self.LABEL_MAPS is None:
            self.LABEL_MAPS = {
                "positive": 2, "pozitif": 2,
                "negative": 0, "negatif": 0,
                "neutral":  1, "notr": 1, "nötr": 1,
            }
    
    def normalize(self, series: pd.Series) -> pd.Series:
        """
        Bir pandas Series'deki label'ları normalize eder.

        Args:
            series: Ham label serisi (string veya int)

        Returns:
            0/1/2 değerlerinden oluşan normalize edilmiş seri
        """

        first = series.dropna().iloc[0]

        if self._is_binary_int(series):
            logger.info("Binary int label tespit edildi (0/1) → 0/2'ye çevriliyor")
            return series.map({0: 0, 1: 2})     

        if pd.api.types.is_integer_dtype(series):
            logger.info("Integer label, dokunulmadı")
            return series
        
        logger.info("String label tespit edildi → sayısala çevriliyor")
        return series.str.lower().str.strip().map(
            lambda x: self.LABEL_MAPS.get(x, -1)
        )
    
    def _is_binary_int(self, series: pd.Series) -> bool:
        """Serinin yalnızca 0 ve 1'den oluşup oluşmadığını kontrol et."""
        unique = set(series.dropna().unique())
        return unique.issubset({0,1}) and pd.api.types.is_integer_dtype(series)
    