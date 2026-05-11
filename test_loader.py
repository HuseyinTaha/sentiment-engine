import logging
logging.basicConfig(level=logging.INFO)

from src.data.loader import HuggingFaceLoader

# --- TEST 1: Türkçe dataset ---
print("\n" + "="*50)
print("TÜRKÇE DATASET TESTİ")
print("="*50)

tr_loader = HuggingFaceLoader(
    dataset_name="winvoker/turkish-sentiment-analysis-dataset",
    language="tr",
    data_dir="data/raw",
    max_samples=200                        
)

tr_data = tr_loader.load()
print("Kolonlar:", tr_data["train"].columns.tolist())
print("İlk 3 satır:\n", tr_data["train"].head(3))
print("Validate:", tr_loader.validate())


# --- TEST 2: İngilizce dataset ---
print("\n" + "="*50)
print("İNGİLİZCE DATASET TESTİ")
print("="*50)

en_loader = HuggingFaceLoader(
    dataset_name="stanfordnlp/sst2",
    language="en",
    data_dir="data/raw",
    max_samples=200
)

en_data = en_loader.load()
print("Kolonlar:", en_data["train"].columns.tolist())
print("İlk 3 satır:\n", en_data["train"].head(3))
print("Validate:", en_loader.validate())

# Her iki loader'da normalize_columns test et
print("\n--- Normalize sonrası kolonlar ---")
print("TR:", tr_data["train"].columns.tolist())
print("EN:", en_data["train"].columns.tolist())

# Label dağılımını gör
print("\n--- Label dağılımı ---")
print("TR:\n", tr_data["train"]["label"].value_counts())
print("EN:\n", en_data["train"]["label"].value_counts())

print("\n--- Normalize sonrası label dağılımı ---")
print("TR (0=neg, 1=nötr, 2=pos):")
print(tr_data["train"]["label"].value_counts().sort_index())

print("\nEN (0=neg, 2=pos):")
print(en_data["train"]["label"].value_counts().sort_index())

# Preprocessing testi
print("\n" + "="*50)
print("TURKISH PREPROCESSOR TESTİ")
print("\n" + "="*50)

from src.preprocessing.turkish import TurkishPreprocessor
from src.preprocessing.base import PreprocessConfig

config = PreprocessConfig(apply_stemming=False)
preprocessor = TurkishPreprocessor(config=config)

test_metinler = [
    "Ürün çooook güzeldi!! 😍 http://link.com @user #harika kesinlikle tavsiye ederim",
    "Bu kadar kötü bir ürün görmedim, param çöpe gitti!!!",
    "İdare eder, ne iyi ne kötü bir ürün.",
]

for metin in test_metinler:
    print(f"\nHam : {metin}")
    print(f"Temiz: {preprocessor.process(metin)}")

print("\n" + "="*50)
print("ENGLISH PREPROCESSOR TESTİ")
print("="*50)

from src.preprocessing.english import EnglishPreprocessor

en_preprocessor = EnglishPreprocessor()

test_texts = [
    "This product is amazinggg!! 😍 http://link.com @user #great totally recommend it",
    "I don't like this at all, it's a waste of money!!!",
    "It's okay, not great but isn't terrible either.",
]

for text in test_texts:
    print(f"\nHam  : {text}")
    print(f"Temiz: {en_preprocessor.process(text)}")

