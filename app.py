import streamlit as st
import torch
import pickle
from pathlib import Path
from langdetect import detect

from src.preprocessing.turkish import TurkishPreprocessor
from src.preprocessing.english import EnglishPreprocessor
from src.preprocessing.base import PreprocessConfig
from src.models.tfidf_model import TFIDFModel
from src.models.lstm_model import LSTMModel
from src.models.bert_model import BERTModel
from src.models.base import ModelConfig

from huggingface_hub import hf_hub_download, snapshot_download
import os

HF_USERNAME = "HusoPasha"

st.set_page_config(
    page_title="SentimentEngine",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
.result-card {
    padding: 1rem 1.25rem;
    border-radius: 12px;
    border: 0.5px solid #e0e0e0;
    margin-bottom: 1rem;
}
.label-pos { color: #085041; background: #E1F5EE; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 500; }
.label-neg { color: #791F1F; background: #FCEBEB; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 500; }
.label-neu { color: #633806; background: #FAEEDA; padding: 3px 10px; border-radius: 6px; font-size: 13px; font-weight: 500; }
.meta-pill { display: inline-block; background: #f5f5f5; color: #555; border-radius: 99px; padding: 4px 12px; font-size: 12px; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_tfidf():
    path = hf_hub_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-tfidf-tr",
        filename="tfidf_tr.pkl"
    )
    config = ModelConfig(model_name="tfidf-tr", num_classes=3)
    model = TFIDFModel(config=config)
    model.load(path)
    return model

@st.cache_resource
def load_tfidf_en():
    config = ModelConfig(model_name="tfidf-en", num_classes=2)
    model = TFIDFModel(config=config)
    path = hf_hub_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-tfidf-en",
        filename="tfidf_en.pkl",
    )
    model.load(path)
    return model

@st.cache_resource
def load_bert_en():
    config = ModelConfig(
        model_name="bert-en",
        num_classes=2,
        max_length=64,
        batch_size=16,
    )
    model = BERTModel(
        config=config,
        model_name_or_path="models/saved/bert_en",
    )
    path = snapshot_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-bert-en",
    )
    model.load(path)
    return model

@st.cache_resource
def load_lstm_tr():
    path = snapshot_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-lstm-tr",
    )
    config = ModelConfig(
        model_name="lstm-tr",
        num_classes=3,
        max_length=128,
        batch_size=64,
    )
    model = LSTMModel(config=config)
    model.load(path)
    return model

@st.cache_resource
def load_lstm_en():
    path = snapshot_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-lstm-en",
    )
    config = ModelConfig(
        model_name="lstm-en",
        num_classes=2,
        max_length=128,
        batch_size=64,
    )
    model = LSTMModel(config=config)
    model.load(path)
    return model

@st.cache_resource
def load_bert():
    path = snapshot_download(
        repo_id=f"{HF_USERNAME}/sentiment-engine-bert-tr",
    )
    config = ModelConfig(
        model_name="bert-tr",
        num_classes=3,
        max_length=64,
        batch_size=16,
    )
    model = BERTModel(
        config=config,
        model_name_or_path=path,
    )
    model.load(path)
    return model

@st.cache_resource
def load_preprocessors():
    cfg = PreprocessConfig(apply_stemming=False)
    return {
        "tr": TurkishPreprocessor(config=cfg),
        "en": EnglishPreprocessor(),
    }

LABEL_MAP = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
EMOJI_MAP = {0: "😞", 1: "😐", 2: "😊"}
BADGE_MAP = {0: "label-neg", 1: "label-neu", 2: "label-pos"}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "tr" if lang == "tr" else "en"
    except Exception:
        return "tr"

def run_model(model, preprocessor, text: str) -> dict:
    clean = preprocessor.process(text)
    result = model.predict([clean])
    pred = int(result.predictions[0])
    probs = result.probabilities[0] if result.probabilities is not None else None

    num_classes = model.config.num_classes

    if num_classes == 2:
        prob_labels = ["Negatif", "Pozitif"]
        prob_colors = ["#E24B4A", "#1D9E75"]
    else:
        prob_labels = ["Negatif", "Nötr", "Pozitif"]
        prob_colors = ["#E24B4A", "#EF9F27", "#1D9E75"]
        
    return {
        "label": pred, 
        "probs": probs, 
        "clean": clean, 
        "prob_labels": prob_labels,
        "prob_colors": prob_colors,
    }

tab1, tab2 = st.tabs(["Tek Metin Analizi", "Toplu CSV Analizi"])

with tab1:
    st.markdown("## 🧠 SentimentEngine")
    st.markdown(
        "Türkçe & İngilizce metinler için duygu analizi. "
        "TF-IDF ve BERT modellerini karşılaştır."
    )
    st.divider()

    st.markdown("#### Model seç")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_tfidf = st.checkbox("TF-IDF", value=True)
    with col2:
        use_lstm = st.checkbox("LSTM", value=True)
    with col3:
        use_bert = st.checkbox("BERT", value=True)

    if not use_tfidf and not use_lstm and not use_bert:
        st.warning("En az bir model seçmelisin.")
        st.stop()

    st.markdown("#### Metin gir")
    text = st.text_area(
        label="metin",
        placeholder="Türkçe veya İngilizce bir metin yaz...",
        height=120,
        label_visibility="collapsed",
    )

    ornek_metinler = [
        "Ürün gerçekten çok kaliteli, kesinlikle tavsiye ederim!",
        "Bu kadar kötü bir ürün görmedim, param çöpe gitti.",
        "İdare eder, ne iyi ne kötü.",
        "This product is absolutely amazing, best purchase ever!",
        "Terrible quality, do not waste your money.",
    ]

    with st.expander("Ornek metinler"):
        for ornek in ornek_metinler:
            if st.button(ornek[:60] + "...", key=ornek):
                text = ornek
                st.rerun()

    analyze = st.button("Analiz et", type="primary", use_container_width=True)

    if analyze and text.strip():
        lang = detect_language(text)
        token_count = len(text.split())

        preprocessors = load_preprocessors()
        preprocessor = preprocessors[lang]

        results = {}

        with st.spinner("Analiz ediliyor..."):
            if lang == "tr":
                tfidf_fn = load_tfidf
                lstm_fn  = load_lstm_tr
                bert_fn = load_bert
            else:
                tfidf_fn = load_tfidf_en
                lstm_fn  = load_lstm_en
                bert_fn = load_bert_en

            if use_tfidf:
                results["TF-IDF"] = run_model(tfidf_fn(), preprocessor, text)

            if use_lstm:
                results["LSTM"] = run_model(lstm_fn(), preprocessor, text)

            if use_bert:
                results["BERT"] = run_model(bert_fn(), preprocessor, text)


        lang_label = "Türkçe" if lang == "tr" else "İngilizce"
        agree = len(set(r["label"] for r in results.values())) == 1

        st.divider()
        st.markdown("### Sonuçlar")

        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.metric("Dil", lang_label)
        with meta_col2:
            st.metric("Token sayısı", token_count)
        with meta_col3:
            if len(results) > 1:
                st.metric(
                    "Modeller aynı fikirde",
                    "Evet ✓" if agree else "Hayır ✗",
                )
            else:
                st.metric("Aktif model", list(results.keys())[0])

        st.divider()

        cols = st.columns(len(results))

        for col, (model_name, res) in zip(cols, results.items()):
            with col:
                label = res["label"]
                probs = res["probs"]

                st.markdown(f"**{model_name}**")
                st.markdown(
                    f"<span class='{BADGE_MAP[label]}'>"
                    f"{EMOJI_MAP[label]} {LABEL_MAP[label]}</span>",
                    unsafe_allow_html=True,
                )
                st.write("")

                if probs is not None:
                    import plotly.graph_objects as go

                    fig = go.Figure(go.Bar(
                        x=res["prob_labels"],
                        y=[round(p * 100, 1) for p in probs],
                        marker_color=res["prob_colors"],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=260,
                        margin=dict(t=20, b=10, l=10, r=10),
                        yaxis=dict(range=[0, 115], showticklabels=False),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(
                        fig, 
                        use_container_width=True,
                        key=f"chart_{model_name}_{hash(text)}",
                    )


        with st.expander("Preprocessing sonucu"):
            for model_name, res in results.items():
                st.markdown(f"**{model_name}: `{res['clean']}`")

    elif analyze and not text.strip():
        st.warning("Lütfen bir metin gir.")


with tab2:
    st.markdown("#### CSV dosyası yükle")
    st.markdown(
        "CSV dosyanızda analiz edilecek metinleri içeren bir kolon olmalı. "
        "Dosya yüklendikten sonra hangi kolonu analiz etmek istediğinizi seçebilirsiniz."
    )

    uploaded_file = st.file_uploader(              # (2)
        "CSV dosyası seç",
        type=["csv"],
        help="UTF-8 formatında CSV dosyası yükleyin"
    )

    if uploaded_file:
        import pandas as pd
        import io

        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin-1")
        
        st.success(f"{len(df)} satır yüklendi")
        st.dataframe(df.head(3), use_container_width=True)

        text_col = st.selectbox(
            "Hangi kolon analiz edilsin?",
            options=df.columns.tolist(),
        )

        st.markdown("#### Model seç")
        c1, c2, c3 = st.columns(3)
        with c1:
            csv_tfidf = st.checkbox("TF-IDF", value=True, key="csv_tfidf")
        with c2:
            csv_lstm = st.checkbox("LSTM", value=False, key="csv_lstm")
        with c3:
            csv_bert = st.checkbox("BERT", value=False, key="csv_bert")

        min_slider = min(10, len(df))                          
        max_slider = min(1000, len(df))

        if min_slider >= max_slider:                           
            max_rows = len(df)
            st.info(f"Tüm {len(df)} satır analiz edilecek")
        else:
            max_rows = st.slider(
            "Maksimum satır sayısı",
            min_value=min_slider,
            max_value=max_slider,
            value=min_slider,
            step=10,
        )

        if st.button("Toplu analiz başlat", type="primary", use_container_width=True):
            df_analyze = df[[text_col]].head(max_rows).copy()
            df_analyze[text_col] = df_analyze[text_col].fillna("").astype(str)

            texts = df_analyze[text_col].tolist()

            sample = " ".join(texts[:5])
            lang = detect_language(sample)
            lang_label = "Türkçe" if lang == "tr" else "İngilizce"
            st.info(f"Dil tespiti: {lang_label}")

            preprocessors = load_preprocessors()
            preprocessor = preprocessors[lang]

            if lang == "tr":
                tfidf_fn = load_tfidf
                lstm_fn  = load_lstm_tr
                bert_fn  = load_bert
            else:
                tfidf_fn = load_tfidf_en
                lstm_fn  = load_lstm_en
                bert_fn  = load_bert_en
            
            progress = st.progress(0, text="Analiz başlıyor...")
            clean_texts = []

            for i, text in enumerate(texts):
                clean_texts.append(preprocessor.process(text))
                if i % 10 == 0:
                    progress.progress(
                        int((i / len(texts)) * 40),
                        text=f"Metin temizleniyor: {i}/{len(texts)}"
                    )
            
            active_models = {}
            if csv_tfidf:
                active_models["TF-IDF"] = tfidf_fn()
            if csv_lstm:
                active_models["LSTM"] = lstm_fn()
            if csv_bert:
                active_models["BERT"] = bert_fn()

            progress.progress(50, text="Modeller yüklendi, tahmin yapılıyor...")

            for model_name, model in active_models.items():
                result = model.predict(clean_texts)          
                preds = result.predictions

                df_analyze[f"{model_name}_label"] = [
                    LABEL_MAP.get(int(p), str(p)) for p in preds
                ]
                df_analyze[f"{model_name}_emoji"] = [
                    EMOJI_MAP.get(int(p), "❓") for p in preds
                ]

                if result.probabilities is not None:
                    df_analyze[f"{model_name}_confidence"] = [
                        round(float(max(prob)) * 100, 1)       
                        for prob in result.probabilities
                    ]

            progress.progress(100, text="Tamamlandı!")

            st.markdown("#### Sonuçlar")
            st.dataframe(df_analyze, use_container_width=True)

            st.markdown("#### Dağılım")
            label_cols = [c for c in df_analyze.columns if c.endswith("_label")]

            dist_cols = st.columns(len(label_cols))
            for col, label_col in zip(dist_cols, label_cols):
                with col:
                    model_name = label_col.replace("_label", "")
                    counts = df_analyze[label_col].value_counts()

                    import plotly.graph_objects as go
                    fig = go.Figure(go.Pie(                     
                        labels=counts.index.tolist(),
                        values=counts.values.tolist(),
                        marker_colors=["#E24B4A", "#EF9F27", "#1D9E75"],
                        hole=0.4,
                    ))
                    fig.update_layout(
                        title=model_name,
                        height=250,
                        margin=dict(t=40, b=10, l=10, r=10),
                        showlegend=True,
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"pie_{model_name}",
                    )

            st.markdown("#### Sonuçları indir")
            csv_buffer = io.StringIO()
            df_analyze.to_csv(csv_buffer, index=False, encoding="utf-8")

            st.download_button(
                label="Sonuçları CSV olarak indir",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
                use_container_width=True,
            )