import logging
logging.basicConfig(level=logging.INFO)

from src.data.loader import HuggingFaceLoader
from src.preprocessing import TurkishPreprocessor, EnglishPreprocessor, PreprocessConfig

# 1. Türkçe veri yükle
print("="*60)
print("TÜRKÇE PIPELINE TESTİ")
print("="*60)

tr_loader = HuggingFaceLoader(
    dataset_name="winvoker/turkish-sentiment-analysis-dataset",
    language="tr",
    max_samples=100  # test için 100 satır yeterli
)

tr_data = tr_loader.load()
print(f"✓ {len(tr_data['train'])} Türkçe satır yüklendi")

# 2. Preprocessing uygula
config = PreprocessConfig(apply_stemming=False)
tr_preprocessor = TurkishPreprocessor(config=config)

tr_data["train"]["clean_text"] = tr_data["train"]["text"].apply(
    tr_preprocessor.process
)

print("\n--- İlk 3 satır karşılaştırması ---")
for i in range(3):
    print(f"\nHAM  : {tr_data['train'].iloc[i]['text'][:80]}...")
    print(f"TEMİZ: {tr_data['train'].iloc[i]['clean_text']}")
    print(f"LABEL: {tr_data['train'].iloc[i]['label']}")

# 3. İngilizce aynı şekilde
print("\n" + "="*60)
print("İNGİLİZCE PIPELINE TESTİ")
print("="*60)

en_loader = HuggingFaceLoader(
    dataset_name="stanfordnlp/sst2",
    language="en",
    max_samples=100
)

en_data = en_loader.load()
print(f"✓ {len(en_data['train'])} İngilizce satır yüklendi")

en_preprocessor = EnglishPreprocessor()
en_data["train"]["clean_text"] = en_data["train"]["text"].apply(
    en_preprocessor.process
)

print("\n--- İlk 3 satır karşılaştırması ---")
for i in range(3):
    print(f"\nHAM  : {en_data['train'].iloc[i]['text']}")
    print(f"TEMİZ: {en_data['train'].iloc[i]['clean_text']}")
    print(f"LABEL: {en_data['train'].iloc[i]['label']}")

print("\n" + "="*60)
print("PIPELINE BAŞARILI - Model katmanına geçilebilir")
print("="*60)