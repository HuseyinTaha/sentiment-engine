import streamlit as st
import torch
import pickle
from pathlib import Path
from langdetect import detect

from src.preprocessing.turkish import TurkishPreprocessor
from src.preprocessing.english import EnglishPreprocessor
from src.preprocessing.base import PreprocessConfig
from src.models.tfidf_model import TFIDFModel
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
    return {"label": pred, "probs": probs, "clean": clean}

st.markdown("## 🧠 SentimentEngine")
st.markdown(
    "Türkçe & İngilizce metinler için duygu analizi. "
    "TF-IDF ve BERT modellerini karşılaştır."
)
st.divider()

st.markdown("#### Model seç")
col1, col2 = st.columns(2)
with col1:
    use_tfidf = st.checkbox("TF-IDF", value=True)
with col2:
    use_bert = st.checkbox("BERT", value=True)

if not use_tfidf and not use_bert:
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
        if use_tfidf:
            tfidf = load_tfidf()
            results["TF-IDF"] = run_model(tfidf, preprocessor, text)

        if use_bert:
            bert = load_bert()
            results["BERT"] = run_model(bert, preprocessor, text)

    
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
        st.metric(
            "Modeller aynı fikirde",
            "Evet ✓" if agree else "Hayır ✗",
        )
    
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

                present = sorted(set([0, 1, 2]) & set(range(len(probs))))
                labels = [LABEL_MAP[i] for i in range(len(probs))]
                colors = ["#E24B4A", "#EF9F27", "#1D9E75"]

                fig = go.Figure(go.Bar(
                    x=labels,
                    y=[round(p * 100, 1) for p in probs],
                    marker_color=colors[:len(probs)],
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
                st.plotly_chart(fig, use_container_width=True)


    with st.expander("Preprocessing sonucu"):
        for model_name, res in results.items():
            st.markdown(f"**{model_name}: `{res['clean']}`")
    
elif analyze and not text.strip():
    st.warning("Lütfen bir metin gir.")















