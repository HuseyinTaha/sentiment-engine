# 🧠 SentimentEngine

Türkçe ve İngilizce metinler için çok dilli duygu analizi sistemi.  
TF-IDF, LSTM ve BERT modellerini karşılaştıran, production'a deploy edilmiş NLP projesi.

🔗 **[Canlı Demo](https://sentiment-engine-kkzmak7zx8txreqpu9i9vc.streamlit.app/#sentiment-engine)** — metni yaz, 3 modeli karşılaştır

---

## 🎯 Proje Hakkında

Sosyal medya yorumları, ürün değerlendirmeleri veya herhangi bir metin için duygu analizi yapar.  
Üç farklı makine öğrenmesi yaklaşımını karşılaştırarak hangi modelin neden daha iyi çalıştığını gösterir.

**Desteklenen diller:** Türkçe 🇹🇷 · İngilizce 🇬🇧  
**Sınıflar:** Negatif · Nötr · Pozitif

---

## 📊 Model Performansı

| Model | Dil | Accuracy | F1 Macro | Eğitim Süresi |
|-------|-----|----------|----------|---------------|
| TF-IDF + LR | TR | 0.84 | 0.74 | ~5 sn |
| TF-IDF + LR | EN | 0.90 | 0.90 | ~3 sn |
| BiLSTM | TR | 0.85+ | 0.80+ | ~10 dk |
| BiLSTM | EN | 0.87+ | 0.87+ | ~8 dk |
| BERT (fine-tuned) | TR | **0.94** | **0.89** | ~25 dk |
| BERT (fine-tuned) | EN | **0.93** | **0.93** | ~20 dk |

> Tüm modeller RTX 3050 Ti üzerinde eğitilmiştir.

---

## 🏗️ Mimari

```
sentiment_engine/
├── src/
│   ├── data/               # Data Layer
│   │   └── loader.py       # HuggingFace dataset yükleme, normalizasyon
│   ├── preprocessing/      # Preprocessing Layer
│   │   ├── base.py         # Abstract BasePreprocessor
│   │   ├── turkish.py      # Türkçe: stopwords, stemming, karakter norm.
│   │   └── english.py      # İngilizce: lemmatization, contraction expansion
│   ├── models/             # Model Layer
│   │   ├── base.py         # Abstract BaseModel, ModelConfig, ModelResult
│   │   ├── tfidf_model.py  # TF-IDF + Logistic Regression (sklearn Pipeline)
│   │   ├── lstm_model.py   # Bidirectional LSTM (PyTorch) + Vocabulary
│   │   └── bert_model.py   # BERT fine-tuning (HuggingFace Transformers)
│   └── analysis/           # Analysis Layer
│       ├── evaluator.py    # ModelEvaluator, confusion matrix, hata analizi
│       └── comparator.py   # ResultComparator, model karşılaştırma grafikleri
├── app.py                  # Streamlit arayüzü
└── train_and_save.py       # CLI ile model eğitimi
```

### Tasarım Prensipleri

- **OOP + SOLID:** Her katman abstract base class ile tanımlanmış, alt sınıflar implement eder
- **Template Method Pattern:** `BasePreprocessor.process()` adım sırasını belirler, dile özgü detayları alt sınıflar doldurur
- **Single Responsibility:** `LabelNormalizer`, `Vocabulary`, `ModelEvaluator` her biri tek iş yapar
- **Open/Closed:** Yeni dil veya model eklemek mevcut kodu değiştirmez, sadece extend eder

---

## 🚀 Kurulum

```bash
git clone https://github.com/HusoPasha/sentiment_engine.git
cd sentiment_engine

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### NLTK veri paketleri

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## ⚙️ Kullanım

### Streamlit arayüzü

```bash
streamlit run app.py
```

### Model eğitimi (CLI)

```bash
# Sadece Türkçe TF-IDF
python train_and_save.py --lang tr --model tfidf

# Sadece İngilizce BERT
python train_and_save.py --lang en --model bert

# Hızlı test (500 örnek)
python train_and_save.py --lang tr --model lstm --max-samples 500

# Hepsini eğit
python train_and_save.py --lang all --model all
```

### Python API

```python
from src.data.loader import HuggingFaceLoader
from src.preprocessing.turkish import TurkishPreprocessor
from src.models.bert_model import BERTModel
from src.models.base import ModelConfig

# Veri yükle
loader = HuggingFaceLoader(
    dataset_name="winvoker/turkish-sentiment-analysis-dataset",
    language="tr",
    max_samples=1000,
)
data = loader.load()

# Preprocessing
preprocessor = TurkishPreprocessor()
clean_texts = [preprocessor.process(t) for t in data["train"]["text"]]

# Tahmin
model = BERTModel(
    config=ModelConfig(model_name="bert-tr", num_classes=3),
    model_name_or_path="HusoPasha/sentiment-engine-bert-tr",
)
model.load("HusoPasha/sentiment-engine-bert-tr")
result = model.predict(["Ürün çok kaliteliydi, kesinlikle tavsiye ederim"])
# → ModelResult(predictions=[2])  # 2 = Pozitif
```

---

## 🗄️ Kullanılan Veri Setleri

| Dataset | Dil | Boyut | Kaynak |
|---------|-----|-------|--------|
| turkish-sentiment-analysis | TR | ~490K | [HuggingFace](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset) |
| SST-2 (Stanford) | EN | ~67K | [HuggingFace](https://huggingface.co/datasets/stanfordnlp/sst2) |

---

## 🤖 Modeller (HuggingFace Hub)

| Model | Link |
|-------|------|
| TF-IDF TR | [sentiment-engine-tfidf-tr](https://huggingface.co/HusoPasha/sentiment-engine-tfidf-tr) |
| TF-IDF EN | [sentiment-engine-tfidf-en](https://huggingface.co/HusoPasha/sentiment-engine-tfidf-en) |
| LSTM TR | [sentiment-engine-lstm-tr](https://huggingface.co/HusoPasha/sentiment-engine-lstm-tr) |
| LSTM EN | [sentiment-engine-lstm-en](https://huggingface.co/HusoPasha/sentiment-engine-lstm-en) |
| BERT TR | [sentiment-engine-bert-tr](https://huggingface.co/HusoPasha/sentiment-engine-bert-tr) |
| BERT EN | [sentiment-engine-bert-en](https://huggingface.co/HusoPasha/sentiment-engine-bert-en) |

---

## 🛠️ Teknoloji Stack

**ML / DL:** PyTorch · HuggingFace Transformers · scikit-learn  
**NLP:** NLTK · Snowball Stemmer  
**Data:** pandas · HuggingFace Datasets  
**Arayüz:** Streamlit · Plotly  
**Model Depolama:** HuggingFace Hub (Git LFS)  
**Versiyon Kontrol:** Git + Conventional Commits  

---

## 📈 Neden 3 Farklı Model?

| | TF-IDF | LSTM | BERT |
|--|--------|------|------|
| Kelime sırası | ❌ | ✅ | ✅ |
| Bağlam anlayışı | ❌ | Kısmi | ✅ |
| Transfer learning | ❌ | ❌ | ✅ |
| Eğitim süresi | Saniyeler | Dakikalar | Saatler |
| GPU gereksinimi | ❌ | Opsiyonel | Önerilen |
| Yorumlanabilirlik | ✅ Yüksek | Orta | ❌ Düşük |

> "Neden BERT daha iyi?" sorusunun cevabı bu tabloda.  
> "Neden TF-IDF hâlâ değerli?" sorusunun cevabı da.

---

## 👤 Geliştirici

**Hüseyin** — Yazılım Mühendisi
[GitHub](https://github.com/HuseyinTaha) · [LinkedIn](www.linkedin.com/in/hüseyin-taha-danış-0897002b6)
