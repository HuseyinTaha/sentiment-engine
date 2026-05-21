import logging
import argparse
logging.basicConfig(level=logging.INFO)

from src.data.loader import HuggingFaceLoader
from src.preprocessing.turkish import TurkishPreprocessor
from src.preprocessing.english import EnglishPreprocessor
from src.preprocessing.base import PreprocessConfig
from src.models.tfidf_model import TFIDFModel
from src.models.bert_model import BERTModel
from src.models.base import ModelConfig
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
        description="Sentiment Model Eğitimi"
    )
    parser.add_argument(
        "--lang",
        choices=["tr","en","all"],
        default="all",
        help="Hangi dil için eğitim yapılacak",
    )
    parser.add_argument(
        "--model",
        choices=["tfidf","bert", "all"],
        default="all",
        help="Hangi model eğitilecek",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Hızlı test için örnek sayısı (varsayılan: tüm veri)",
    )
    return parser.parse_args()


# ── 1. VERİ ──────────────────────────────────────────

def prepare_tr_data(max_samples=None):
    print("Türkçe Veri yükleniyor...")

    tr_loader = HuggingFaceLoader(
        dataset_name="winvoker/turkish-sentiment-analysis-dataset",
        language="tr",
        max_samples=max_samples,       # ← tam veri
    )
    tr_data = tr_loader.load()

    config = PreprocessConfig(apply_stemming=False)
    preprocessor = TurkishPreprocessor(config=config)

    print("Preprocessing uygulanıyor...")
    tr_data["train"]["clean_text"] = (
        tr_data["train"]["text"].apply(preprocessor.process)
    )

    df = tr_data["train"].dropna(subset=["clean_text", "label"])
    df = df[df["clean_text"].str.strip() != ""]

    X = df["clean_text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def prepare_en_data(max_samples=None):
    print("\nİngilizce veri yükleniyor...")
    en_loader = HuggingFaceLoader(
        dataset_name="stanfordnlp/sst2",
        language="en",
        max_samples=max_samples,
    )
    data = en_loader.load()

    preprocessor = EnglishPreprocessor()
    data["train"]["clean_text"] = (
        data["train"]["text"].apply(preprocessor.process)
    )

    df = data["train"].dropna(subset=["clean_text", "label"])
    df = df[df["clean_text"].str.strip() != ""]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"].tolist()
    )
    print(f"EN — Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ── 2. TF-IDF ────────────────────────────────────────

def train_tfidf_tr(X_train, X_test, y_train, y_test):
    print("\nTürkçe TF-IDF eğitiliyor...")

    tfidf_config = ModelConfig(
        model_name="tfidf-tr",
        num_classes=3,
    )
    tfidf_model = TFIDFModel(config=tfidf_config)
    tfidf_model.train(X_train, y_train, X_test, y_test)

    tfidf_model.save("models/saved/tfidf_tr.pkl")
    print("TF-IDF kaydedildi → models/saved/tfidf_tr.pkl")
    return tfidf_model

def train_tfidf_en(X_train, X_test, y_train, y_test):
    print("\nİngilizce TF-IDF eğitiliyor...")
    tfidf_config = ModelConfig(
        model_name="tfidf-en",
        num_classes=2,
    )
    tfidf_model = TFIDFModel(config=tfidf_config)
    tfidf_model.train(X_train, y_train, X_test, y_test) 
    tfidf_model.save("models/saved/tfidf_en.pkl")
    print("Kaydedildi → models/saved/tfidf_en.pkl")
    return tfidf_model


# ── 3. BERT ──────────────────────────────────────────

def train_bert_tr(X_train, X_test, y_train, y_test):
    print("\nBERT tr eğitiliyor (bu biraz uzun sürer)...")

    bert_config = ModelConfig(
        model_name="bert-tr",
        num_classes=3,
        max_length=64,
        batch_size=16,          
        learning_rate=2e-5,
        num_epochs=3,
    )
    bert_model = BERTModel(
        config=bert_config,
        model_name_or_path="dbmdz/bert-base-turkish-cased",
    )
    bert_model.train(X_train, y_train, X_test, y_test)

    bert_model.save("models/saved/bert_tr")
    print("BERT kaydedildi → models/saved/bert_tr/")
    return bert_model

def train_bert_en(X_train, X_test, y_train, y_test):
    print("\nBERT en eğitiliyor (bu biraz uzun sürer)...")
    bert_config = ModelConfig(
        model_name="bert-en",
        num_classes=2,
        max_length=64,
        batch_size=16,          
        learning_rate=2e-5,
        num_epochs=3,
    )
    bert_model = BERTModel(
        config=bert_config,
        model_name_or_path="distilbert-base-uncased",
    )
    bert_model.train(X_train, y_train, X_test, y_test)
    bert_model.save("models/saved/bert_en")
    print("Kaydedildi → models/saved/bert_en/")
    return bert_model


def main():
    args = get_args()

    do_tr = args.lang in ("tr", "all")
    do_en = args.lang in ("en", "all")
    do_tfidf = args.model in ("tfidf", "all")
    do_bert = args.model in ("bert", "all")

    tr_data = prepare_tr_data(args.max_samples) if do_tr else None
    en_data = prepare_en_data(args.max_samples) if do_en else None

    if do_tr and do_tfidf:
        train_tfidf_tr(*tr_data)
    
    if do_en and do_tfidf:
        train_tfidf_en(*en_data)
    
    if do_tr and do_bert:
        train_bert_tr(*tr_data)
    
    if do_en and do_bert:
        train_bert_en(*en_data)
    
    print("\nTüm eğitimler tamamlandı.")

if __name__ == "__main__":
    main()
    