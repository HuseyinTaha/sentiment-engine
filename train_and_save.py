import logging
logging.basicConfig(level=logging.INFO)

from src.data.loader import HuggingFaceLoader
from src.preprocessing.turkish import TurkishPreprocessor
from src.preprocessing.base import PreprocessConfig
from src.models.tfidf_model import TFIDFModel
from src.models.bert_model import BERTModel
from src.models.base import ModelConfig
from sklearn.model_selection import train_test_split

# ── 1. VERİ ──────────────────────────────────────────
print("Veri yükleniyor...")

tr_loader = HuggingFaceLoader(
    dataset_name="winvoker/turkish-sentiment-analysis-dataset",
    language="tr",
    max_samples=None,       # ← tam veri
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

# ── 2. TF-IDF ────────────────────────────────────────
print("\nTF-IDF eğitiliyor...")

tfidf_config = ModelConfig(
    model_name="tfidf-tr",
    num_classes=3,
)
tfidf_model = TFIDFModel(config=tfidf_config)
tfidf_model.train(X_train, y_train, X_test, y_test)

tfidf_model.save("models/saved/tfidf_tr.pkl")
print("TF-IDF kaydedildi → models/saved/tfidf_tr.pkl")

# ── 3. BERT ──────────────────────────────────────────
print("\nBERT eğitiliyor (bu biraz uzun sürer)...")

bert_config = ModelConfig(
    model_name="bert-tr",
    num_classes=3,
    max_length=64,
    batch_size=16,          # ← tam eğitimde 16
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

# ── 4. SONUÇLAR ──────────────────────────────────────
print("\n" + "="*50)
print("SONUÇLAR")
print("="*50)

from src.analysis.evaluator import ModelEvaluator
from src.analysis.comparator import ResultComparator

tfidf_eval = ModelEvaluator(tfidf_model).evaluate(X_test, y_test, "TF-IDF")
bert_eval  = ModelEvaluator(bert_model).evaluate(X_test, y_test, "BERT")

comp = ResultComparator()
comp.add_result(tfidf_eval)
comp.add_result(bert_eval)
comp.print_comparison()

print("\nModeller kaydedildi, Streamlit'e geçilebilir.")