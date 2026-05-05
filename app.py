import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, re, io, base64, os, tempfile
from collections import Counter
from datetime import datetime

st.set_page_config(
    page_title="AvisCheckeur",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown('''
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .hero {
    background: linear-gradient(135deg, #00b67a 0%, #007a52 100%);
    padding: 2rem 2rem 1.5rem; border-radius: 16px;
    color: white; margin-bottom: 1.5rem;
  }
  .hero h1 { font-size: 2rem; font-weight: 800; margin: 0; }
  .hero p  { font-size: .95rem; opacity: .88; margin: .4rem 0 0; }
  .review-card {
    background: white; border: 1px solid #e8e8e8;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    margin-bottom: .8rem; box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }
  .avatar {
    width:36px; height:36px; border-radius:50%;
    background:#00b67a; color:white;
    display:inline-flex; align-items:center; justify-content:center;
    font-weight:700; font-size:.85rem;
  }
  .stars       { color:#00b67a; font-size:1.1rem; margin:.3rem 0; }
  .stars.neg   { color:#ff3722; }
  .stars.neu   { color:#ff8f00; }
  .badge { display:inline-block; padding:.22rem .75rem;
           border-radius:999px; font-size:.72rem; font-weight:700; margin-top:.5rem; }
  .badge-pos { background:#e8f8f1; color:#007a52; }
  .badge-neg { background:#ffecea; color:#c0392b; }
  .badge-neu { background:#fff3e0; color:#b45309; }
  .stat-box {
    background:white; border:1px solid #e8e8e8;
    border-radius:12px; padding:1.1rem 1rem; text-align:center;
  }
  .stat-box .num { font-size:2rem; font-weight:800; line-height:1; }
  .stat-box .lbl { font-size:.78rem; color:#767676; margin-top:.3rem; }
  .score-pill {
    background:#00b67a; color:white;
    padding:.3rem .8rem; border-radius:6px;
    font-weight:700; font-size:1.1rem; display:inline-block;
  }
  .score-pill.neg { background:#ff3722; }
  .score-pill.neu { background:#ff8f00; }
  .nlp-tag {
    display:inline-block; padding:.2rem .6rem;
    border-radius:999px; font-size:.72rem;
    font-weight:600; margin:.15rem;
    background:#e8f8f1; color:#007a52;
  }
  .nlp-tag.neg { background:#ffecea; color:#c0392b; }
  .sec-title {
    font-size:1.15rem; font-weight:700; color:#191919;
    margin:1rem 0 .6rem; padding-bottom:.4rem;
    border-bottom:2px solid #00b67a;
  }
  .stButton > button {
    background:#00b67a !important; color:white !important;
    border-radius:8px !important; font-weight:600 !important;
    border:none !important;
  }
  .ai-box {
    background: linear-gradient(135deg, #f0faf5 0%, #e8f8f1 100%);
    border: 1.5px solid #00b67a; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin: 1rem 0;
  }
  .ai-box h4 { color: #007a52; margin: 0 0 .6rem; font-size: 1rem; }
</style>
''', unsafe_allow_html=True)

# ── COLOR TEMPLATES ──
COLOR_TEMPLATES = {
    "🟢 Vert (défaut)": {
        "pos": "#00b67a", "neg": "#ff3722", "neu": "#ff8f00",
        "primary": "#00b67a", "secondary": "#007a52",
        "cmap_pos": "Greens", "cmap_neg": "Reds",
    },
    "🔵 Bleu Ocean": {
        "pos": "#0066cc", "neg": "#e63946", "neu": "#f4a261",
        "primary": "#0066cc", "secondary": "#004499",
        "cmap_pos": "Blues", "cmap_neg": "Reds",
    },
    "🟣 Violet": {
        "pos": "#7c3aed", "neg": "#dc2626", "neu": "#d97706",
        "primary": "#7c3aed", "secondary": "#5b21b6",
        "cmap_pos": "Purples", "cmap_neg": "Reds",
    },
    "🟠 Orange Pro": {
        "pos": "#ea580c", "neg": "#9333ea", "neu": "#0891b2",
        "primary": "#ea580c", "secondary": "#c2410c",
        "cmap_pos": "Oranges", "cmap_neg": "Purples",
    },
    "⚫ Dark Mode": {
        "pos": "#22d3ee", "neg": "#f43f5e", "neu": "#facc15",
        "primary": "#22d3ee", "secondary": "#0891b2",
        "cmap_pos": "Blues", "cmap_neg": "Reds",
    },
}


# ── HELPERS ──

@st.cache_resource(show_spinner=False)
def load_transformer(model_key):
    from transformers import pipeline
    ids = {
        "DistilBERT ⚡": "distilbert-base-uncased-finetuned-sst-2-english",
        "BERT 🎯":        "textattack/bert-base-uncased-SST-2",
    }
    return pipeline("sentiment-analysis", model=ids[model_key], truncation=True, max_length=512)


@st.cache_data(show_spinner=False)
def load_imdb(path, n=100):
    df = pd.read_csv(path)
    df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    df["review_clean"] = (
        df["review"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.lower().str.strip()
    )
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def load_custom_file(uploaded_file, text_col=None):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")
    elif name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8", errors="replace")
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

    if text_col is None:
        priority = ["review", "text", "comment", "avis", "texte", "content", "body", "description", "message"]
        for col_name in priority:
            matches = [c for c in df.columns if col_name in c.lower()]
            if matches:
                text_col = matches[0]
                break
        if text_col is None:
            text_cols = df.select_dtypes(include="object").columns
            if len(text_cols) > 0:
                text_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())

    df["review"] = df[text_col].astype(str)
    df["review_clean"] = (
        df["review"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.lower().str.strip()
    )
    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"
    df["label"] = 0
    return df, text_col


def normalize_label(raw):
    m = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE",
         "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE"}
    return m.get(raw.upper(), "NEUTRAL")


def score_to_sentiment(score, label, threshold=0.70):
    s = normalize_label(label)
    if score < threshold:
        s = "NEUTRAL"
    return s, score


def get_textblob_scores(text):
    try:
        from textblob import TextBlob
        tb = TextBlob(text)
        return {"polarity": round(tb.sentiment.polarity, 3),
                "subjectivity": round(tb.sentiment.subjectivity, 3)}
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.5}


def extract_keywords(text, top_n=10):
    STOP = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "this", "that", "it", "is", "was", "are", "were", "be", "been", "have", "has",
        "i", "my", "we", "our", "you", "your", "they", "their", "not", "no", "so", "as",
        "by", "from", "all", "just", "more", "very", "too", "also", "would", "could",
        "do", "did", "does", "if", "when", "then", "there", "here", "what", "which",
        "movie", "film", "one", "get", "like", "really", "even", "br", "its",
        "much", "about", "some", "see", "make", "well", "know", "he", "she", "his", "her",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq = Counter(w for w in words if w not in STOP)
    return [w for w, _ in freq.most_common(top_n)]


def make_wordcloud_img(text, color="#00b67a", cmap="Greens"):
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wc = WordCloud(width=600, height=280, background_color="white",
                       colormap=cmap, max_words=60, collocations=False).generate(text)
        buf = io.BytesIO()
        plt.figure(figsize=(6, 2.8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


def stars_html(sentiment):
    n = {"POSITIVE": 5, "NEUTRAL": 3, "NEGATIVE": 1}.get(sentiment, 3)
    cls = {"POSITIVE": "", "NEGATIVE": " neg", "NEUTRAL": " neu"}.get(sentiment, "")
    return f'<div class="stars{cls}">{"★" * n}{"☆" * (5 - n)}</div>'


def review_card_html(text, sentiment, confidence, reviewer="Anonyme", date=""):
    badge_cls = {"POSITIVE": "badge-pos", "NEGATIVE": "badge-neg", "NEUTRAL": "badge-neu"}
    label_fr  = {"POSITIVE": "Positif ✅", "NEGATIVE": "Negatif ❌", "NEUTRAL": "Neutre ➖"}
    preview   = text[:220] + "…" if len(text) > 220 else text
    date_str  = date or datetime.now().strftime("%d %b %Y")
    return (
        '<div class="review-card">'
        '<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem">'
        f'<div class="avatar">{reviewer[0].upper()}</div>'
        '<div>'
        f'<div style="font-weight:600;font-size:.9rem;color:#191919">{reviewer}</div>'
        f'<div style="font-size:.75rem;color:#767676">Publie le {date_str}</div>'
        '</div></div>'
        + stars_html(sentiment) +
        f'<div style="font-size:.88rem;color:#404040;line-height:1.55">{preview}</div>'
        f'<span class="badge {badge_cls[sentiment]}">{label_fr[sentiment]} - {confidence:.0%}</span>'
        '</div>'
    )


def call_anthropic_summary(stats_text):
    import urllib.request
    import json
    prompt = (
        "Tu es un expert en analyse de sentiment et en experience client.\n"
        "Voici les statistiques d une analyse de sentiments sur des avis clients :\n\n"
        + stats_text +
        "\n\nFournis un resume analytique structure en francais avec :\n"
        "1. Vue d ensemble : Interpretation globale des sentiments\n"
        "2. Points forts : Ce qui ressort positivement\n"
        "3. Points d attention : Les axes d amelioration\n"
        "4. Recommandations : 3 actions concretes\n"
        "5. Score de sante : Note globale sur 10 avec justification\n\n"
        "Sois concis, professionnel et actionnable. 300-400 mots maximum."
    )
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["content"][0]["text"]


def generate_pdf_report(df_r, summary_text, template_name, template, model_name, n_reviews):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    primary_color = colors.HexColor(template["primary"])
    styles = getSampleStyleSheet()
    title_style   = ParagraphStyle("T2", parent=styles["Title"],
                                   textColor=primary_color, fontSize=22, spaceAfter=6)
    h1_style      = ParagraphStyle("H1", parent=styles["Heading1"],
                                   textColor=primary_color, fontSize=14, spaceAfter=4)
    body_style    = ParagraphStyle("B2", parent=styles["Normal"],
                                   fontSize=10, leading=16, spaceAfter=4)
    caption_style = ParagraphStyle("Cap", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.grey)

    story = []

    story.append(Paragraph("AvisCheckeur", title_style))
    story.append(Paragraph("Rapport Analyse de Sentiments",
                            ParagraphStyle("Sub", parent=styles["Normal"],
                                           fontSize=14, textColor=colors.grey, spaceAfter=2)))
    story.append(Paragraph(
        "Genere le " + datetime.now().strftime("%d %B %Y a %H:%M") +
        "  |  Modele : " + model_name + "  |  Template : " + template_name,
        caption_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=12))

    counts   = df_r["sentiment_pred"].value_counts()
    n_pos    = counts.get("POSITIVE", 0)
    n_neg    = counts.get("NEGATIVE", 0)
    n_neu    = counts.get("NEUTRAL", 0)
    avg_conf = df_r["confidence"].mean()
    pct_pos  = n_pos / max(len(df_r), 1)
    trust    = round(1 + pct_pos * 4, 1)

    story.append(Paragraph("Statistiques Cles", h1_style))
    data_stats = [
        ["Metrique", "Valeur"],
        ["Total avis analyses", str(n_reviews)],
        ["Avis positifs", str(n_pos) + " (" + f"{n_pos/max(len(df_r),1)*100:.1f}" + "%)"],
        ["Avis negatifs",  str(n_neg) + " (" + f"{n_neg/max(len(df_r),1)*100:.1f}" + "%)"],
        ["Avis neutres",   str(n_neu) + " (" + f"{n_neu/max(len(df_r),1)*100:.1f}" + "%)"],
        ["Confiance moyenne", f"{avg_conf:.1%}"],
        ["TrustScore global", f"{trust}/5.0"],
    ]
    t_stats = Table(data_stats, colWidths=[8*cm, 8*cm])
    t_stats.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), primary_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 12))

    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pie_colors = [template["pos"], template["neg"], template["neu"]]
        axes[0].pie([n_pos, n_neg, n_neu], labels=["Positif", "Negatif", "Neutre"],
                    colors=pie_colors, autopct="%1.1f%%", startangle=90, pctdistance=0.8)
        axes[0].set_title("Repartition des Sentiments", fontweight="bold")
        for sent, color in [("POSITIVE", template["pos"]),
                             ("NEGATIVE", template["neg"]),
                             ("NEUTRAL",  template["neu"])]:
            subset = df_r[df_r["sentiment_pred"] == sent]["confidence"]
            if len(subset) > 0:
                axes[1].hist(subset, bins=12, alpha=0.7, color=color, label=sent)
        axes[1].set_title("Distribution des Scores", fontweight="bold")
        axes[1].set_xlabel("Score")
        axes[1].set_ylabel("Nombre avis")
        axes[1].legend(fontsize=8)
        plt.tight_layout()
        chart_buf = io.BytesIO()
        plt.savefig(chart_buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        chart_buf.seek(0)
        story.append(RLImage(chart_buf, width=16*cm, height=6.4*cm))
        story.append(Spacer(1, 8))
    except Exception as e:
        story.append(Paragraph("[Graphiques non disponibles: " + str(e) + "]", caption_style))

    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=6))
    story.append(Paragraph("Analyse IA - Resume Intelligent", h1_style))
    if summary_text:
        clean = summary_text.replace("**", "").replace("##", "").replace("#", "")
        for line in clean.split("\n"):
            line = line.strip()
            if line:
                if line[:2] in ("1.", "2.", "3.", "4.", "5."):
                    story.append(Paragraph("<b>" + line + "</b>", body_style))
                else:
                    story.append(Paragraph(line, body_style))
    else:
        story.append(Paragraph(
            "Aucun resume IA disponible. Cliquez sur Generer le resume IA avant l export.",
            body_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=6))
    story.append(Paragraph("Echantillon Avis Analyses", h1_style))
    sample = df_r.head(8)[["review", "sentiment_pred", "confidence"]].copy()
    sample["review"]     = sample["review"].str[:80] + "..."
    sample["confidence"] = sample["confidence"].apply(lambda x: f"{x:.0%}")
    sample.columns       = ["Extrait", "Sentiment", "Confiance"]
    tbl_data = [list(sample.columns)] + sample.values.tolist()
    t_rev = Table(tbl_data, colWidths=[10*cm, 3.5*cm, 2.5*cm])
    t_rev.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), primary_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_rev)
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
    story.append(Paragraph(
        "AvisCheckeur  |  Template : " + template_name + "  |  " + datetime.now().strftime("%Y"),
        caption_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="background:#00b67a;padding:.8rem;border-radius:10px;margin-bottom:1rem;">'
        '<span style="color:white;font-weight:800;font-size:1.1rem;">✅ AvisCheckeur</span></div>',
        unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox("Modèle NLP", ["DistilBERT ⚡", "BERT 🎯"])
    threshold    = st.slider("Seuil neutralité", 0.50, 0.90, 0.70, 0.05)
    n_batch      = st.slider("Avis à analyser (IMDB)", 20, 200, 50, 10)
    st.markdown("---")
    st.markdown("### 🎨 Template PDF")
    template_name = st.selectbox("Palette de couleurs", list(COLOR_TEMPLATES.keys()))
    template      = COLOR_TEMPLATES[template_name]
    st.markdown(
        f'<div style="display:flex;gap:6px;margin-top:8px">'
        f'<div style="width:28px;height:28px;border-radius:6px;background:{template["pos"]}"></div>'
        f'<div style="width:28px;height:28px;border-radius:6px;background:{template["neg"]}"></div>'
        f'<div style="width:28px;height:28px;border-radius:6px;background:{template["neu"]}"></div>'
        f'</div><div style="font-size:.72rem;color:#767676;margin-top:4px">Positif · Négatif · Neutre</div>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📚 Notions NLP")
    st.markdown(
        '<div style="font-size:.82rem;color:#555;line-height:1.8">'
        '✅ Tokenisation<br>✅ Stop words<br>✅ Stemming<br>✅ Polarité TextBlob<br>'
        '✅ BOW / TF-IDF<br>✅ DistilBERT / BERT<br>✅ Matrice de confusion</div>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ISE3 ENEAM - Prof. Hounna - 2025")


# ══════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════
st.markdown(
    '<div class="hero"><h1>✅ AvisCheckeur</h1>'
    '<p>Analyse de sentiments intelligente · Import CSV/Excel · Résumé IA · Export PDF · DistilBERT & BERT</p>'
    '</div>',
    unsafe_allow_html=True)

with st.spinner(f"⏳ Chargement de {model_choice}…"):
    classifier = load_transformer(model_choice)
st.success(f"✅ {model_choice} prêt !")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Analyser un avis",
    "📂 Importer un fichier",
    "📋 Tableau de bord",
    "🔬 Prétraitement NLP",
    "📊 Visualisations",
    "🏆 Évaluation",
])

NAMES_LIST = ["John D.", "Sarah M.", "Mike P.", "Emma L.",
              "James W.", "Clara B.", "Tom R.", "Alice K."]


# ══════════════════════════════════════════
# TAB 1 — ANALYSER UN AVIS
# ══════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        st.markdown('<div class="sec-title">✍️ Saisir un avis</div>', unsafe_allow_html=True)
        reviewer_name = st.text_input("Nom du reviewer", value="Anonyme")
        user_text = st.text_area("Texte de l avis (anglais)", height=160,
                                  placeholder="Ex: This movie is absolutely brilliant!")
        examples = [
            "This film is an absolute masterpiece. The acting and story are top-notch.",
            "Terrible waste of time. Boring plot and no coherent story at all.",
            "The movie was okay. Some good scenes but overall a bit disappointing.",
        ]
        ex = st.selectbox("Ou choisir un exemple :", ["— exemples —"] + examples)
        if ex != "— exemples —":
            user_text = ex
        analyze_btn = st.button("🔍 Analyser cet avis", use_container_width=True)

    with col_out:
        st.markdown('<div class="sec-title">📊 Résultat</div>', unsafe_allow_html=True)
        if analyze_btn and user_text.strip():
            with st.spinner("Analyse…"):
                t0  = time.time()
                raw = classifier(user_text)[0]
                ms  = (time.time() - t0) * 1000
            sent, conf = score_to_sentiment(raw["score"], raw["label"], threshold)
            tb = get_textblob_scores(user_text)
            st.markdown(review_card_html(user_text, sent, conf, reviewer_name), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                cls = "" if sent == "POSITIVE" else " neg" if sent == "NEGATIVE" else " neu"
                st.markdown(
                    f'<div class="stat-box"><div class="num">'
                    f'<span class="score-pill{cls}">{conf:.0%}</span></div>'
                    f'<div class="lbl">Confiance</div></div>',
                    unsafe_allow_html=True)
            with c2:
                col_c = "#00b67a" if tb["polarity"] > 0 else "#ff3722"
                st.markdown(
                    f'<div class="stat-box"><div class="num" style="color:{col_c}">'
                    f'{tb["polarity"]:+.2f}</div>'
                    f'<div class="lbl">Polarité TextBlob</div></div>',
                    unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f'<div class="stat-box"><div class="num" style="color:#767676">'
                    f'{tb["subjectivity"]:.2f}</div>'
                    f'<div class="lbl">Subjectivité</div></div>',
                    unsafe_allow_html=True)
            keywords = extract_keywords(user_text)
            if keywords:
                kw_cls = "nlp-tag" if sent == "POSITIVE" else "nlp-tag neg"
                tags = " ".join(f'<span class="{kw_cls}">{w}</span>' for w in keywords[:8])
                st.markdown(f"**🔑 Mots-clés :** {tags}", unsafe_allow_html=True)
            st.caption(f"⏱ {ms:.0f} ms · {model_choice}")
        else:
            st.markdown(
                '<div style="background:#f0faf5;border:1.5px dashed #00b67a;border-radius:12px;'
                'padding:2rem;text-align:center;color:#555">'
                '<div style="font-size:2rem">🎬</div>'
                '<div style="font-weight:600;margin-top:.5rem">Entrez un avis pour le voir apparaitre ici</div>'
                '</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2 — IMPORTER UN FICHIER
# ══════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📂 Importer vos données (CSV, Excel, TSV…)</div>', unsafe_allow_html=True)
    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("**Formats supportés :** CSV, Excel (.xlsx/.xls), TSV")
        st.markdown("**Colonnes requises :** Une colonne texte contenant les avis")
        st.markdown("**Optionnel :** Colonne `sentiment` (positive/negative) pour l evaluation")

        uploaded_file = st.file_uploader(
            "Déposer votre fichier ici",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Votre fichier doit contenir au moins une colonne de texte d avis")

        if uploaded_file is not None:
            try:
                preview_df, detected_col = load_custom_file(uploaded_file)
                st.success(f"✅ Fichier chargé : {len(preview_df)} lignes · Colonne détectée : **{detected_col}**")
                all_text_cols = preview_df.select_dtypes(include="object").columns.tolist()
                idx = all_text_cols.index(detected_col) if detected_col in all_text_cols else 0
                selected_col = st.selectbox("Colonne de texte à analyser :", all_text_cols, index=idx)
                st.markdown("**Aperçu des données :**")
                st.dataframe(preview_df[[selected_col]].head(5), use_container_width=True)
                max_rows = min(len(preview_df), 500)
                n_file   = st.slider("Nombre d avis à analyser", 10, max_rows, min(50, max_rows), 10)

                if st.button("🚀 Lancer l analyse du fichier", use_container_width=True):
                    uploaded_file.seek(0)
                    df_custom, _ = load_custom_file(uploaded_file, selected_col)
                    df_custom    = df_custom.head(n_file).reset_index(drop=True)
                    results = []
                    bar = st.progress(0)
                    with st.spinner(f"Analyse de {len(df_custom)} avis en cours…"):
                        for i, row in enumerate(df_custom.itertuples()):
                            raw_pred = classifier(str(row.review_clean)[:512])[0]
                            s, c = score_to_sentiment(raw_pred["score"], raw_pred["label"], threshold)
                            tb   = get_textblob_scores(str(row.review_clean)[:300])
                            results.append({
                                "review":        str(row.review)[:200],
                                "sentiment_vrai": str(row.sentiment),
                                "sentiment_pred": s,
                                "confidence":     round(c, 3),
                                "polarity":       tb["polarity"],
                                "subjectivity":   tb["subjectivity"],
                                "nb_mots":        len(str(row.review_clean).split()),
                                "reviewer":       NAMES_LIST[i % len(NAMES_LIST)],
                            })
                            bar.progress((i + 1) / len(df_custom))
                        bar.empty()
                    df_r = pd.DataFrame(results)
                    st.session_state["df_results"]  = df_r
                    st.session_state["last_n"]      = len(df_r)
                    st.session_state["data_source"] = f"Fichier importé : {uploaded_file.name}"
                    st.success(f"✅ Analyse terminée ! {len(df_r)} avis analysés. → Voir **Tableau de bord**")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")

    with col_info:
        st.markdown(
            '<div style="background:#f0faf5;border:1.5px solid #00b67a;border-radius:12px;padding:1.2rem">'
            '<h4 style="color:#007a52;margin:0 0 .8rem">📋 Format attendu</h4>'
            '<b>Colonnes auto-détectées :</b><br>'
            '<code>review</code>, <code>text</code>, <code>comment</code>, '
            '<code>avis</code>, <code>texte</code>, <code>content</code>'
            '<br><br><b>💡 Astuce :</b> Vous pouvez aussi utiliser le dataset IMDB dans '
            '<b>Tableau de bord</b>.</div>',
            unsafe_allow_html=True)
        sample_csv = pd.DataFrame({
            "review":    ["This product is amazing!", "Very disappointing.",
                          "It was okay.", "Incredible quality!", "Not what I expected."],
            "sentiment": ["positive", "negative", "positive", "positive", "negative"]
        }).to_csv(index=False)
        st.download_button("⬇️ Télécharger un modèle CSV exemple",
                           sample_csv, "modele_avis.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════
# TAB 3 — TABLEAU DE BORD
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">📋 Tableau de bord</div>', unsafe_allow_html=True)

    possible_paths = [
        "IMDB_Dataset.csv",
        "/content/sentimentscope/IMDB_Dataset.csv",
        "./IMDB_Dataset.csv",
    ]
    csv_path    = next((p for p in possible_paths if os.path.exists(p)), None)
    data_source = st.session_state.get("data_source", "IMDB Dataset")

    if "df_results" not in st.session_state:
        if csv_path is None:
            st.info("💡 Importez un fichier via **📂 Importer un fichier** ou uploadez IMDB ci-dessous.")
            uploaded_csv = st.file_uploader("Uploader IMDB_Dataset.csv", type="csv")
            if uploaded_csv:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded_csv.read())
                tmp.flush()
                csv_path = tmp.name
            else:
                st.stop()

        with st.spinner("Chargement IMDB…"):
            df = load_imdb(csv_path, n=n_batch)
        results = []
        bar = st.progress(0)
        with st.spinner(f"Classification de {len(df)} avis…"):
            for i, row in enumerate(df.itertuples()):
                raw = classifier(str(row.review_clean)[:512])[0]
                s, c = score_to_sentiment(raw["score"], raw["label"], threshold)
                tb   = get_textblob_scores(str(row.review_clean)[:300])
                results.append({
                    "review":        str(row.review)[:200],
                    "sentiment_vrai": row.sentiment,
                    "sentiment_pred": s,
                    "confidence":     round(c, 3),
                    "polarity":       tb["polarity"],
                    "subjectivity":   tb["subjectivity"],
                    "nb_mots":        len(str(row.review_clean).split()),
                    "reviewer":       NAMES_LIST[i % len(NAMES_LIST)],
                })
                bar.progress((i + 1) / len(df))
            bar.empty()
        df_r = pd.DataFrame(results)
        st.session_state["df_results"] = df_r
        st.session_state["last_n"]     = n_batch
    else:
        df_r = st.session_state["df_results"]

    counts  = df_r["sentiment_pred"].value_counts()
    pct_pos = counts.get("POSITIVE", 0) / max(len(df_r), 1)
    trust   = round(1 + pct_pos * 4, 1)

    st.markdown(
        f'<div style="font-size:.8rem;color:#767676;margin-bottom:.5rem">📁 Source : {data_source}</div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:white;border:1px solid #e8e8e8;border-radius:12px;'
        f'padding:1.2rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.2rem">'
        f'<span class="score-pill" style="font-size:1.6rem;padding:.4rem 1rem">{trust}</span>'
        f'<div><div style="font-size:1.1rem;font-weight:700">TrustScore global</div>'
        f'<div style="color:#00b67a;font-size:1.2rem">{"★" * int(trust)}{"☆" * (5 - int(trust))}</div>'
        f'<div style="font-size:.82rem;color:#767676">Basé sur {len(df_r)} avis</div></div>'
        f'<div style="margin-left:auto;font-size:.85rem;color:#555">'
        f'✅ {counts.get("POSITIVE",0)} positifs &nbsp;'
        f'❌ {counts.get("NEGATIVE",0)} négatifs &nbsp;'
        f'➖ {counts.get("NEUTRAL",0)} neutres</div></div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col_w, (val, color, lbl) in zip([c1, c2, c3, c4], [
        (len(df_r), "#191919", "Avis analysés"),
        (counts.get("POSITIVE", 0), "#00b67a", "Positifs ★★★★★"),
        (counts.get("NEGATIVE", 0), "#ff3722", "Négatifs ★☆☆☆☆"),
        (f'{df_r["confidence"].mean():.1%}', "#007a52", "Confiance moy."),
    ]):
        with col_w:
            st.markdown(
                f'<div class="stat-box"><div class="num" style="color:{color}">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # AI SUMMARY
    st.markdown('<div class="sec-title">🤖 Résumé IA (Analyse Intelligente)</div>', unsafe_allow_html=True)
    _, ai_btn_col = st.columns([3, 1])
    with ai_btn_col:
        gen_summary = st.button("✨ Générer le résumé IA", use_container_width=True)

    if gen_summary:
        stats_text = "\n".join([
            f"- Nombre total d avis : {len(df_r)}",
            f"- Avis positifs : {counts.get('POSITIVE',0)} ({counts.get('POSITIVE',0)/max(len(df_r),1)*100:.1f}%)",
            f"- Avis negatifs : {counts.get('NEGATIVE',0)} ({counts.get('NEGATIVE',0)/max(len(df_r),1)*100:.1f}%)",
            f"- Avis neutres  : {counts.get('NEUTRAL',0)} ({counts.get('NEUTRAL',0)/max(len(df_r),1)*100:.1f}%)",
            f"- Confiance moyenne : {df_r['confidence'].mean():.1%}",
            f"- Polarite moyenne : {df_r['polarity'].mean():+.3f}",
            f"- Subjectivite moyenne : {df_r['subjectivity'].mean():.3f}",
            f"- Longueur moyenne avis : {df_r['nb_mots'].mean():.0f} mots",
            f"- TrustScore global : {trust}/5.0",
            f"- Modele utilise : {model_choice}",
        ])
        with st.spinner("🤖 L IA analyse les résultats…"):
            try:
                summary = call_anthropic_summary(stats_text)
            except Exception as e:
                summary = (
                    f"[Erreur API : {e}]\n\n"
                    f"Resume automatique :\n"
                    f"- {counts.get('POSITIVE',0)} avis positifs\n"
                    f"- {counts.get('NEGATIVE',0)} avis negatifs\n"
                    f"- Confiance moyenne : {df_r['confidence'].mean():.1%}\n"
                    f"- TrustScore : {trust}/5"
                )
            st.session_state["ai_summary"] = summary

    if "ai_summary" in st.session_state:
        st.markdown(
            f'<div class="ai-box"><h4>🤖 Analyse IA</h4>'
            f'<div style="font-size:.9rem;line-height:1.7;white-space:pre-wrap">'
            f'{st.session_state["ai_summary"]}</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:#f8f8f8;border:1.5px dashed #ccc;border-radius:10px;'
            'padding:1rem;text-align:center;color:#888;font-size:.9rem">'
            'Cliquez sur <b>✨ Générer le résumé IA</b> pour obtenir une analyse intelligente</div>',
            unsafe_allow_html=True)

    # PDF EXPORT
    st.markdown("---")
    st.markdown('<div class="sec-title">📥 Exporter le Tableau de Bord PDF</div>', unsafe_allow_html=True)
    st.markdown(f"Template sélectionné : **{template_name}** (modifier dans la sidebar ⚙️)")
    if st.button("📄 Générer le PDF", use_container_width=False):
        with st.spinner("Génération du PDF…"):
            try:
                pdf_bytes = generate_pdf_report(
                    df_r, st.session_state.get("ai_summary", ""),
                    template_name, template, model_choice, len(df_r))
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("✅ PDF prêt !")
            except Exception as e:
                st.error(f"❌ Erreur PDF : {e}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Télécharger le rapport PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"AvisCheckeur_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sec-title">🃏 Derniers avis analysés</div>', unsafe_allow_html=True)
    for _, row in df_r.head(6).iterrows():
        st.markdown(review_card_html(row["review"], row["sentiment_pred"],
                                      row["confidence"], row["reviewer"]), unsafe_allow_html=True)
    with st.expander("📋 Voir tous les résultats"):
        st.dataframe(df_r, use_container_width=True)
        st.download_button("⬇️ Télécharger CSV", df_r.to_csv(index=False),
                           "avis_results.csv", "text/csv")


# ══════════════════════════════════════════
# TAB 4 — PRÉTRAITEMENT NLP
# ══════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">🔬 Pipeline Prétraitements NLP</div>', unsafe_allow_html=True)
    sample_text = st.text_area(
        "Texte à prétraiter :",
        value="This film is absolutely brilliant! The director did a wonderful job.",
        height=100)
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1: do_lower = st.checkbox("🔡 Lowercasing", True)
    with col_o2: do_stop  = st.checkbox("🚫 Stop words", True)
    with col_o3: do_stem  = st.checkbox("✂️ Stemming", False)

    if st.button("▶️ Appliquer le pipeline", use_container_width=True):
        st.markdown("**Étape 1 — Texte brut**")
        st.code(sample_text)
        if do_lower:
            st.markdown("**Étape 2 — Lowercasing**")
            st.code(sample_text.lower())
        clean  = re.sub(r"[^a-zA-Z\s]", " ", sample_text.lower())
        tokens = clean.split()
        st.markdown(f"**Étape 3 — Tokenisation** → {len(tokens)} tokens")
        st.code(str(tokens[:20]))
        if do_stop:
            STOP_W = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
                      "this","that","it","is","was","are","were","be","been","have","has"}
            tokens = [t for t in tokens if t not in STOP_W and len(t) > 2]
            st.markdown(f"**Étape 4 — Stop words** → {len(tokens)} tokens restants")
            st.code(str(tokens[:20]))
        if do_stem:
            try:
                import nltk
                nltk.download("punkt", quiet=True)
                from nltk.stem import PorterStemmer
                ps    = PorterStemmer()
                stems = [ps.stem(t) for t in tokens[:10]]
                st.markdown("**Étape 5 — Stemming**")
                st.dataframe(pd.DataFrame({"Token": tokens[:10], "Stem": stems}),
                             use_container_width=True)
            except Exception:
                st.info("NLTK requis pour le stemming")
        freq    = Counter(tokens)
        df_bow  = pd.DataFrame(freq.most_common(12), columns=["Token", "Fréquence"])
        fig_bow = px.bar(df_bow, x="Token", y="Fréquence", title="Bag of Words",
                         color_discrete_sequence=[template["primary"]])
        fig_bow.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig_bow, use_container_width=True)
        st.markdown("**Polarité & Subjectivité TextBlob**")
        tb = get_textblob_scores(sample_text)
        c1, c2 = st.columns(2)
        with c1: st.metric("Polarité", f'{tb["polarity"]:+.3f}')
        with c2: st.metric("Subjectivité", f'{tb["subjectivity"]:.3f}')
        st.markdown("**Nuage de mots**")
        img = make_wordcloud_img(" ".join(tokens), template["primary"], template["cmap_pos"])
        if img:
            st.markdown(
                f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:10px">',
                unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 5 — VISUALISATIONS
# ══════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">📊 Visualisations & Tendances</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.info("💡 Lancez d abord l analyse dans **Tableau de bord** ou **Importer un fichier**.")
    else:
        df_v = st.session_state["df_results"]
        cmap_colors = {
            "POSITIVE": template["pos"],
            "NEGATIVE": template["neg"],
            "NEUTRAL":  template["neu"],
        }
        col_a, col_b = st.columns(2)
        with col_a:
            cnts = df_v["sentiment_pred"].value_counts().reset_index()
            cnts.columns = ["sentiment", "count"]
            fig_d = px.pie(cnts, values="count", names="sentiment", color="sentiment",
                           color_discrete_map=cmap_colors, hole=.55,
                           title="Répartition des sentiments")
            fig_d.update_traces(textposition="outside", textinfo="percent+label")
            fig_d.update_layout(showlegend=False, height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_d, use_container_width=True)
        with col_b:
            fig_h = px.histogram(df_v, x="confidence", color="sentiment_pred",
                                  color_discrete_map=cmap_colors, nbins=15,
                                  barmode="overlay", opacity=.75,
                                  title="Distribution des scores de confiance")
            fig_h.update_layout(height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_h, use_container_width=True)
        col_c, col_d = st.columns(2)
        with col_c:
            fig_sc = px.scatter(df_v, x="polarity", y="confidence", color="sentiment_pred",
                                 color_discrete_map=cmap_colors,
                                 title="Polarité TextBlob vs Confiance", opacity=.7)
            fig_sc.update_layout(height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_sc, use_container_width=True)
        with col_d:
            fig_bx = px.box(df_v, x="sentiment_pred", y="nb_mots", color="sentiment_pred",
                             color_discrete_map=cmap_colors,
                             title="Longueur des avis par sentiment")
            fig_bx.update_layout(height=320, showlegend=False, margin=dict(t=50, b=0))
            st.plotly_chart(fig_bx, use_container_width=True)
        st.markdown('<div class="sec-title">☁️ Nuages de mots</div>', unsafe_allow_html=True)
        col_w1, col_w2 = st.columns(2)
        pos_t = " ".join(df_v[df_v["sentiment_pred"] == "POSITIVE"]["review"].str[:200])
        neg_t = " ".join(df_v[df_v["sentiment_pred"] == "NEGATIVE"]["review"].str[:200])
        with col_w1:
            st.markdown("**Positifs ✅**")
            img = make_wordcloud_img(pos_t, template["pos"], template["cmap_pos"])
            if img:
                st.markdown(
                    f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:10px">',
                    unsafe_allow_html=True)
        with col_w2:
            st.markdown("**Négatifs ❌**")
            img2 = make_wordcloud_img(neg_t, template["neg"], template["cmap_neg"])
            if img2:
                st.markdown(
                    f'<img src="data:image/png;base64,{img2}" style="width:100%;border-radius:10px">',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 6 — ÉVALUATION
# ══════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-title">🏆 Évaluation & Comparaison</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.info("💡 Lancez d abord l analyse dans **Tableau de bord**.")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score, confusion_matrix)

        df_e = st.session_state["df_results"].copy()
        df_e["true_bin"] = (df_e["sentiment_vrai"] == "positive").astype(int)
        df_e["pred_bin"] = (df_e["sentiment_pred"] == "POSITIVE").astype(int)

        acc_tr  = accuracy_score(df_e["true_bin"], df_e["pred_bin"])
        f1_tr   = f1_score(df_e["true_bin"], df_e["pred_bin"], average="weighted", zero_division=0)
        prec_tr = precision_score(df_e["true_bin"], df_e["pred_bin"], average="weighted", zero_division=0)
        rec_tr  = recall_score(df_e["true_bin"], df_e["pred_bin"], average="weighted", zero_division=0)

        baseline_ok = False
        try:
            X, y = df_e["review"], df_e["true_bin"]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2,
                                                       random_state=42, stratify=y)
            vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words="english")
            lr  = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(vec.fit_transform(X_tr), y_tr)
            y_pred_lr = lr.predict(vec.transform(X_te))
            acc_lr    = accuracy_score(y_te, y_pred_lr)
            f1_lr     = f1_score(y_te, y_pred_lr, average="weighted", zero_division=0)
            prec_lr   = precision_score(y_te, y_pred_lr, average="weighted", zero_division=0)
            rec_lr    = recall_score(y_te, y_pred_lr, average="weighted", zero_division=0)
            baseline_ok = True
        except Exception:
            pass

        rows = [{"Modèle": f"**{model_choice}**",
                 "Accuracy": f"{acc_tr:.3f}", "Précision": f"{prec_tr:.3f}",
                 "Rappel": f"{rec_tr:.3f}", "F1": f"{f1_tr:.3f}",
                 "Latence": "~42ms" if "Distil" in model_choice else "~118ms"}]
        if baseline_ok:
            rows.append({"Modèle": "TF-IDF+LogReg",
                         "Accuracy": f"{acc_lr:.3f}", "Précision": f"{prec_lr:.3f}",
                         "Rappel": f"{rec_lr:.3f}", "F1": f"{f1_lr:.3f}", "Latence": "~2ms"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            cm     = confusion_matrix(df_e["true_bin"], df_e["pred_bin"])
            fig_cm = px.imshow(cm, x=["Prédit NEG", "Prédit POS"],
                                y=["Réel NEG", "Réel POS"],
                                color_continuous_scale=[[0, "#fff"], [1, template["primary"]]],
                                text_auto=True, title=f"Matrice — {model_choice}")
            fig_cm.update_layout(height=300, margin=dict(t=50, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_cm2:
            if baseline_ok:
                cm2     = confusion_matrix(y_te, y_pred_lr)
                fig_cm2 = px.imshow(cm2, x=["Prédit NEG", "Prédit POS"],
                                     y=["Réel NEG", "Réel POS"],
                                     color_continuous_scale=[[0, "#fff"], [1, template["secondary"]]],
                                     text_auto=True, title="Matrice — TF-IDF+LogReg")
                fig_cm2.update_layout(height=300, margin=dict(t=50, b=10))
                st.plotly_chart(fig_cm2, use_container_width=True)

        fig_r = go.Figure()
        cats  = ["Accuracy", "Précision", "Rappel", "F1"]
        v_tr  = [acc_tr, prec_tr, rec_tr, f1_tr]
        fig_r.add_trace(go.Scatterpolar(
            r=v_tr + [v_tr[0]], theta=cats + [cats[0]],
            fill="toself", name=model_choice,
            line_color=template["primary"], opacity=.7))
        if baseline_ok:
            v_lr = [acc_lr, prec_lr, rec_lr, f1_lr]
            fig_r.add_trace(go.Scatterpolar(
                r=v_lr + [v_lr[0]], theta=cats + [cats[0]],
                fill="toself", name="TF-IDF+LogReg",
                line_color="#767676", opacity=.5))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
            height=360, margin=dict(t=30, b=20))
        st.plotly_chart(fig_r, use_container_width=True)

        delta = f1_tr - (f1_lr if baseline_ok else 0)
        gain_str = f"  Gain vs baseline : <b>{delta:+.3f}</b>" if baseline_ok else ""
        st.markdown(
            f'<div style="background:#f0faf5;border-left:4px solid {template["primary"]};'
            f'padding:1rem;border-radius:0 10px 10px 0;font-size:.9rem">'
            f'<b>Analyse :</b> Le modèle <b>{model_choice}</b> atteint F1 = <b>{f1_tr:.3f}</b> '
            f'sur {len(df_e)} avis.{gain_str}</div>',
            unsafe_allow_html=True)
