"""
=============================================================
  AvisCheckeur — Analyse de avis IMDB / Custom
  ISE3 ENEAM · Prof. Gracieux Hounna · 2025
  Auteur: Emmanuella TSIPOAKA
  Stack : Streamlit · HuggingFace Transformers · TextBlob
          Google Gemini AI · FPDF2 · Plotly · NLTK
  Thème : Violet (customisé)
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, re, io, base64, os, tempfile, json
import urllib.request, urllib.error
from collections import Counter
from datetime import datetime

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AvisCheckeur · NLP ISE3 ENEAM",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (thème violet) ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .hero {
    background: linear-gradient(135deg, #7c3aed 0%, #4c1d95 100%);
    padding: 2.2rem 2.5rem 1.8rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,.06);
    border-radius: 50%;
  }
  .hero::after {
    content: "★★★★★";
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 2.2rem;
    letter-spacing: 6px;
    opacity: .15;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: -.3px;
  }
  .hero p {
    font-size: .95rem;
    opacity: .82;
    margin: .5rem 0 0;
    font-weight: 300;
  }

  .review-card {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: .9rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
    transition: box-shadow .2s, transform .2s;
  }
  .review-card:hover {
    box-shadow: 0 6px 20px rgba(124,58,237,.12);
    transform: translateY(-1px);
  }
  .avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #5b21b6);
    color: white;
    display: inline-flex;
    align-items: center; justify-content: center;
    font-weight: 700; font-size: .88rem;
    flex-shrink: 0;
  }
  .stars         { color: #7c3aed; font-size: 1.05rem; margin: .35rem 0 .5rem; letter-spacing: 2px; }
  .stars.neg     { color: #e63946; }
  .stars.neu     { color: #f4a261; }
  .badge {
    display: inline-block;
    padding: .24rem .8rem;
    border-radius: 999px;
    font-size: .72rem;
    font-weight: 700;
    margin-top: .6rem;
    letter-spacing: .3px;
  }
  .badge-pos { background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd; }
  .badge-neg { background: #fdf0f0; color: #b91c1c; border: 1px solid #fccaca; }
  .badge-neu { background: #fff8ec; color: #92400e; border: 1px solid #fed7aa; }

  .stat-box {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,.04);
  }
  .stat-box .num  { font-size: 2.1rem; font-weight: 700; line-height: 1.1; }
  .stat-box .lbl  { font-size: .76rem; color: #888; margin-top: .35rem; font-weight: 500; letter-spacing: .2px; }

  .score-pill {
    background: #7c3aed;
    color: white;
    padding: .32rem .85rem;
    border-radius: 8px;
    font-weight: 800;
    font-size: 1.1rem;
    display: inline-block;
    letter-spacing: -.3px;
  }
  .score-pill.neg { background: #e63946; }
  .score-pill.neu { background: #f4a261; }

  .nlp-tag {
    display: inline-block;
    padding: .22rem .65rem;
    border-radius: 999px;
    font-size: .73rem;
    font-weight: 600;
    margin: .14rem;
    background: #ede9fe;
    color: #5b21b6;
    border: 1px solid #c4b5fd;
  }
  .nlp-tag.neg {
    background: #fdf0f0;
    color: #b91c1c;
    border: 1px solid #fccaca;
  }

  .sec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111;
    margin: 1.2rem 0 .7rem;
    padding-bottom: .45rem;
    border-bottom: 2.5px solid #7c3aed;
    display: flex;
    align-items: center;
    gap: .5rem;
  }

  .ai-box {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1.5px solid #7c3aed;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
  }
  .ai-box h4 {
    color: #5b21b6;
    margin: 0 0 .8rem;
    font-size: 1rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: .4rem;
  }
  .ai-content {
    font-size: .9rem;
    line-height: 1.8;
    color: #1a1a1a;
    white-space: pre-wrap;
  }

  .info-box {
    background: #f5f3ff;
    border: 1.5px solid #7c3aed;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
  }

  .stButton > button {
    background: #7c3aed !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: .2px !important;
    transition: background .2s !important;
  }
  .stButton > button:hover {
    background: #6d28d9 !important;
  }

  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: .92rem;
  }
  .stTabs [aria-selected="true"] {
    color: #7c3aed !important;
    border-bottom-color: #7c3aed !important;
  }

  .trust-banner {
    background: white;
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 1.3rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
  }

  .empty-state {
    background: #f8fffe;
    border: 2px dashed #c4b5fd;
    border-radius: 14px;
    padding: 2.5rem;
    text-align: center;
    color: #666;
  }
  .empty-state .icon  { font-size: 2.5rem; margin-bottom: .6rem; }
  .empty-state .title { font-weight: 700; font-size: 1rem; color: #333; margin-bottom: .3rem; }
  .empty-state .sub   { font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLOR TEMPLATES (version violet par défaut)
# ══════════════════════════════════════════════════════════════════════════════
COLOR_TEMPLATES = {
    "🟣 Violet Pro (défaut)": {
        "pos": "#7c3aed", "neg": "#e63946", "neu": "#f4a261",
        "primary": "#7c3aed", "secondary": "#5b21b6",
        "cmap_pos": "Purples", "cmap_neg": "Reds",
    },
    "🟢 Vert": {
        "pos": "#00b67a", "neg": "#e63946", "neu": "#f4a261",
        "primary": "#00b67a", "secondary": "#006b47",
        "cmap_pos": "Greens", "cmap_neg": "Reds",
    },
    "🔵 Bleu Océan": {
        "pos": "#0077cc", "neg": "#e63946", "neu": "#f4a261",
        "primary": "#0077cc", "secondary": "#004d99",
        "cmap_pos": "Blues", "cmap_neg": "Reds",
    },
    "⚫ Dark Mode": {
        "pos": "#22d3ee", "neg": "#f43f5e", "neu": "#facc15",
        "primary": "#22d3ee", "secondary": "#0891b2",
        "cmap_pos": "Blues", "cmap_neg": "Reds",
    },
}

NAMES_LIST = [
    "John D.", "Sarah M.", "Mike P.", "Emma L.",
    "James W.", "Clara B.", "Tom R.", "Alice K.",
    "David S.", "Nina R.", "Omar K.", "Léa F.",
]

# ══════════════════════════════════════════════════════════════════════════════
# CORE NLP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_transformer(model_key: str):
    from transformers import pipeline
    model_ids = {
        "DistilBERT ⚡": "distilbert-base-uncased-finetuned-sst-2-english",
        "BERT 🎯":        "textattack/bert-base-uncased-SST-2",
    }
    return pipeline(
        "sentiment-analysis",
        model=model_ids[model_key],
        truncation=True,
        max_length=512,
    )


@st.cache_data(show_spinner=False)
def load_imdb(path: str, n: int = 100) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)
    df["review_clean"] = (
        df["review"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.lower()
        .str.strip()
    )
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


def load_custom_file(uploaded_file, text_col=None):
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    elif name.endswith(".tsv"):
        df = pd.read_csv(uploaded_file, sep="\t", encoding="utf-8", errors="replace")
    else:
        df = pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")

    if text_col is None:
        priority = ["review", "text", "comment", "avis", "texte", "content",
                    "body", "description", "message"]
        for keyword in priority:
            matches = [c for c in df.columns if keyword in c.lower()]
            if matches:
                text_col = matches[0]
                break
        if text_col is None:
            obj_cols = df.select_dtypes(include="object").columns
            if len(obj_cols) > 0:
                text_col = max(obj_cols,
                               key=lambda c: df[c].astype(str).str.len().mean())

    df["review"] = df[text_col].astype(str)
    df["review_clean"] = (
        df["review"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"[^a-zA-Z\s]", " ", regex=True)
        .str.lower()
        .str.strip()
    )
    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"
    df["label"] = 0
    return df, text_col


def normalize_label(raw: str) -> str:
    mapping = {
        "LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE",
        "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE",
    }
    return mapping.get(raw.upper(), "NEUTRAL")


def score_to_sentiment(score: float, label: str, threshold: float = 0.70):
    sentiment = normalize_label(label)
    if score < threshold:
        sentiment = "NEUTRAL"
    return sentiment, score


def get_textblob_scores(text: str) -> dict:
    try:
        from textblob import TextBlob
        tb = TextBlob(text)
        return {
            "polarity":     round(tb.sentiment.polarity, 3),
            "subjectivity": round(tb.sentiment.subjectivity, 3),
        }
    except Exception:
        return {"polarity": 0.0, "subjectivity": 0.5}


def extract_keywords(text: str, top_n: int = 10) -> list:
    STOP = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "this","that","it","is","was","are","were","be","been","have","has",
        "i","my","we","our","you","your","they","their","not","no","so","as",
        "by","from","all","just","more","very","too","also","would","could",
        "do","did","does","if","when","then","there","here","what","which",
        "movie","film","one","get","like","really","even","br","its",
        "much","about","some","see","make","well","know","he","she","his","her",
        "had","has","can","but","been","out","time","good","bad","just",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq  = Counter(w for w in words if w not in STOP)
    return [w for w, _ in freq.most_common(top_n)]


def make_wordcloud_img(text: str, color: str = "#7c3aed",
                       cmap: str = "Purples") -> str:
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not text.strip():
            return ""
        wc = WordCloud(
            width=700, height=300,
            background_color="white",
            colormap=cmap,
            max_words=70,
            collocations=False,
            prefer_horizontal=0.85,
        ).generate(text)
        buf = io.BytesIO()
        plt.figure(figsize=(7, 3))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def stars_html(sentiment: str) -> str:
    n   = {"POSITIVE": 5, "NEUTRAL": 3, "NEGATIVE": 1}.get(sentiment, 3)
    cls = {"POSITIVE": "", "NEGATIVE": " neg", "NEUTRAL": " neu"}.get(sentiment, "")
    return f'<div class="stars{cls}">{"★" * n}{"☆" * (5 - n)}</div>'


def review_card_html(text: str, sentiment: str, confidence: float,
                     reviewer: str = "Anonyme", date: str = "") -> str:
    badge_cls = {
        "POSITIVE": "badge-pos",
        "NEGATIVE": "badge-neg",
        "NEUTRAL":  "badge-neu",
    }
    label_fr = {
        "POSITIVE": "Positif ✅",
        "NEGATIVE": "Négatif ❌",
        "NEUTRAL":  "Neutre ➖",
    }
    preview  = text[:230] + "…" if len(text) > 230 else text
    date_str = date or datetime.now().strftime("%d %b %Y")

    return (
        '<div class="review-card">'
          '<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.5rem">'
            f'<div class="avatar">{reviewer[0].upper()}</div>'
            '<div>'
              f'<div style="font-weight:600;font-size:.9rem;color:#111">{reviewer}</div>'
              f'<div style="font-size:.74rem;color:#aaa">Publié le {date_str}</div>'
            '</div>'
          '</div>'
          + stars_html(sentiment) +
          f'<div style="font-size:.875rem;color:#3a3a3a;line-height:1.6">{preview}</div>'
          f'<span class="badge {badge_cls[sentiment]}">'
            f'{label_fr[sentiment]} · {confidence:.0%}'
          '</span>'
        '</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI AI SUMMARY (version corrigée)
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini_summary(stats_text: str, api_key: str) -> str:
    """
    Appel à l'API Google Gemini (gemini-2.0-flash) pour générer
    un résumé analytique structuré des sentiments.
    """
    if not api_key or not api_key.strip():
        raise ValueError("Clé API Google Gemini manquante.")

    prompt = (
        "Tu es un expert senior en analyse de sentiment et en expérience client (CX).\n"
        "Voici les statistiques d'une analyse de sentiments sur des avis clients :\n\n"
        + stats_text +
        "\n\nFournis un rapport analytique structuré en français avec exactement ces sections :\n\n"
        "**1. Vue d'ensemble**\n"
        "Interprétation globale du sentiment dominant et du niveau de satisfaction.\n\n"
        "**2. Points forts**\n"
        "Ce qui ressort positivement des avis (3 points max, concis).\n\n"
        "**3. Points d'attention**\n"
        "Les axes d'amélioration identifiés (3 points max, concis).\n\n"
        "**4. Recommandations opérationnelles**\n"
        "3 actions concrètes, réalistes et priorisées.\n\n"
        "**5. Score de santé client**\n"
        "Note globale sur 10 avec justification en 2 phrases.\n\n"
        "Ton : professionnel, direct et actionnable. Maximum 350 mots."
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 1024,
        }
    }).encode("utf-8")

    url = (
        "https://generativelanguage.googleapis.com/v1/models/"
        f"gemini-2.0-flash:generateContent?key={api_key.strip()}"
    )
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(
                "Réponse vide de Gemini (aucun candidat). "
                "Vérifiez votre clé API ou les quotas."
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(
                "Réponse vide de Gemini (aucune partie). "
                "La génération a peut-être été bloquée par les filtres de sécurité."
            )

        text_result = parts[0].get("text", "").strip()
        if not text_result:
            raise RuntimeError("Le modèle a retourné un texte vide.")

        return text_result

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            err_json = json.loads(body)
            msg = err_json.get("error", {}).get("message", body[:300])
        except Exception:
            msg = body[:300]
        raise RuntimeError(f"Erreur HTTP {e.code} — {msg}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Erreur réseau : {e.reason}. Vérifiez votre connexion internet.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue API Gemini : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF EXPORT — VERSION CORRIGÉE (nettoyage caractères Unicode)
# ══════════════════════════════════════════════════════════════════════════════

def clean_text_for_pdf(text: str) -> str:
    """Remplace les caractères Unicode non supportés par Helvetica."""
    replacements = {
        "—": "-",   # em dash
        "–": "-",   # en dash
        "‘": "'",   # left single quote
        "’": "'",   # right single quote
        "“": "\"",  # left double quote
        "”": "\"",  # right double quote
        "…": "...", # ellipsis
        "•": "-",   # bullet
        " ": " ",   # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Supprimer tout autre caractère non ASCII (remplacé par ?)
    return ''.join(c if ord(c) < 128 else '?' for c in text)


def generate_pdf_report(df_r: pd.DataFrame, summary_text: str,
                        template_name: str, template: dict,
                        model_name: str, n_reviews: int) -> bytes:
    from fpdf import FPDF

    counts   = df_r["sentiment_pred"].value_counts()
    n_pos    = counts.get("POSITIVE", 0)
    n_neg    = counts.get("NEGATIVE", 0)
    n_neu    = counts.get("NEUTRAL",  0)
    avg_conf = df_r["confidence"].mean()
    pct_pos  = n_pos / max(len(df_r), 1)
    trust    = round(1 + pct_pos * 4, 1)

    def hex2rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    primary_rgb = hex2rgb(template["primary"])

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(*primary_rgb)
            self.rect(0, 0, 210, 22, "F")
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(255, 255, 255)
            self.set_xy(10, 5)
            self.cell(0, 12, "AvisCheckeur — Rapport Analyse de Sentiments", align="L")
            self.set_font("Helvetica", "", 8)
            self.set_xy(10, 14)
            self.cell(0, 6,
                      f"Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
                      f"Modèle : {model_name}  |  Template : {template_name}",
                      align="L")
            self.ln(10)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(180, 180, 180)
            self.cell(0, 8, f"AvisCheckeur  ·  ISE3 ENEAM  ·  Page {self.page_no()}", align="C")

        def section_title(self, title):
            self.ln(4)
            self.set_fill_color(*primary_rgb)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 9, f"  {title}", fill=True, ln=True)
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def kpi_row(self, items):
            w = 190 / len(items)
            start_y = self.get_y()
            start_x = 10
            for idx, (label, val, color) in enumerate(items):
                rgb = hex2rgb(color)
                x = start_x + idx * w
                y = start_y
                self.set_fill_color(245, 245, 245)
                self.set_draw_color(220, 220, 220)
                self.rect(x, y, w - 2, 22, "DF")
                self.set_font("Helvetica", "B", 15)
                self.set_text_color(*rgb)
                self.set_xy(x, y + 2)
                self.cell(w - 2, 10, str(val), align="C")
                self.set_font("Helvetica", "", 8)
                self.set_text_color(100, 100, 100)
                self.set_xy(x, y + 13)
                self.cell(w - 2, 6, label, align="C")
            self.set_xy(start_x, start_y + 26)
            self.set_text_color(0, 0, 0)
            self.set_draw_color(0, 0, 0)
            self.ln(4)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # KPIs
    pdf.section_title("Statistiques Clés")
    pdf.kpi_row([
        ("Avis analysés",    n_reviews,          "#333333"),
        ("Positifs",         n_pos,               template["pos"]),
        ("Négatifs",         n_neg,               template["neg"]),
        ("Neutres",          n_neu,               template["neu"]),
        ("Confiance moy.",   f"{avg_conf:.0%}",   template["primary"]),
        ("TrustScore",       f"{trust}/5",         template["primary"]),
    ])

    # Tableau détail
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*primary_rgb)
    pdf.set_text_color(255, 255, 255)
    cols   = ["Catégorie", "Nombre", "Pourcentage", "Confiance moy."]
    widths = [65, 35, 50, 40]
    for col, w in zip(cols, widths):
        pdf.cell(w, 8, col, border=1, fill=True, align="C")
    pdf.ln()

    row_data = [
        ("POSITIF", n_pos, n_pos / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "POSITIVE"]["confidence"].mean() if n_pos > 0 else 0),
        ("NÉGATIF", n_neg, n_neg / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "NEGATIVE"]["confidence"].mean() if n_neg > 0 else 0),
        ("NEUTRE",  n_neu, n_neu / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "NEUTRAL"]["confidence"].mean()  if n_neu > 0 else 0),
    ]
    pdf.set_font("Helvetica", "", 9)
    for i, (cat, cnt, pct, conf) in enumerate(row_data):
        bg = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(65, 7, f"  {cat}", border=1, fill=True)
        pdf.cell(35, 7, str(cnt),       border=1, fill=True, align="C")
        pdf.cell(50, 7, f"{pct:.1%}",   border=1, fill=True, align="C")
        pdf.cell(40, 7, f"{conf:.1%}" if conf > 0 else "—", border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(6)

    # Graphiques
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        pie_colors = [template["pos"], template["neg"], template["neu"]]
        labels_pie = ["Positif", "Négatif", "Neutre"]
        values_pie = [n_pos, n_neg, n_neu]
        non_zero   = [(l, v, c) for l, v, c in zip(labels_pie, values_pie, pie_colors) if v > 0]
        if non_zero:
            axes[0].pie([v for _, v, _ in non_zero], labels=[l for l, _, _ in non_zero],
                        colors=[c for _, _, c in non_zero], autopct="%1.1f%%",
                        startangle=90, pctdistance=0.78,
                        wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[0].set_title("Répartition des sentiments", fontweight="bold", fontsize=10)

        for sent, clr in [("POSITIVE", template["pos"]),
                          ("NEGATIVE", template["neg"]),
                          ("NEUTRAL",  template["neu"])]:
            subset = df_r[df_r["sentiment_pred"] == sent]["confidence"]
            if len(subset) > 0:
                axes[1].hist(subset.values, bins=12, alpha=0.75, color=clr, label=sent, edgecolor="white")
        axes[1].set_title("Distribution des scores de confiance", fontweight="bold", fontsize=10)
        axes[1].set_xlabel("Score de confiance")
        axes[1].set_ylabel("Nombre d'avis")
        axes[1].legend(fontsize=8)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        plt.tight_layout(pad=1.5)

        chart_buf = io.BytesIO()
        plt.savefig(chart_buf, format="png", dpi=130, bbox_inches="tight")
        plt.close()
        chart_buf.seek(0)

        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_img.write(chart_buf.read())
        tmp_img.close()
        pdf.image(tmp_img.name, x=10, w=190)
        os.unlink(tmp_img.name)
        pdf.ln(4)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 8, f"[Graphiques non disponibles : {e}]", ln=True)
        pdf.set_text_color(0, 0, 0)

    # Résumé IA avec nettoyage Unicode
    pdf.section_title("Analyse IA — Résumé Intelligent (Google Gemini)")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    if summary_text:
        clean = summary_text.replace("**", "").replace("##", "").replace("#", "")
        clean = clean_text_for_pdf(clean)   # <-- CORRECTION : nettoyage des caractères
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            if re.match(r"^\d+\.", line):
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5.5, line)
    else:
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 8, "Aucun résumé IA disponible. Cliquez sur 'Générer le résumé IA' avant l'export.", ln=True)
    pdf.set_text_color(0, 0, 0)

    # Échantillon d'avis
    pdf.add_page()
    pdf.section_title("Échantillon d'Avis Analysés (8 premiers)")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*primary_rgb)
    pdf.set_text_color(255, 255, 255)
    for col, w in [("Extrait de l'avis", 115), ("Sentiment", 35), ("Confiance", 25), ("Polarité", 15)]:
        pdf.cell(w, 7, col, border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    for i, row in df_r.head(8).iterrows():
        bg = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        excerpt = str(row["review"])[:90] + ("…" if len(str(row["review"])) > 90 else "")
        pdf.cell(115, 6, f"  {excerpt}", border=1, fill=True)
        pdf.cell(35,  6, str(row["sentiment_pred"]),       border=1, fill=True, align="C")
        pdf.cell(25,  6, f"{row['confidence']:.0%}",       border=1, fill=True, align="C")
        pdf.cell(15,  6, f"{row.get('polarity', 0):+.2f}", border=1, fill=True, align="C")
        pdf.ln()

    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    else:
        return output.encode("latin-1")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR (thème violet)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#7c3aed,#4c1d95);
                padding:.9rem 1.1rem;border-radius:12px;margin-bottom:1.2rem">
      <div style="color:white;font-weight:800;font-size:1.1rem;
                  font-family:'DM Serif Display',serif">✅ AvisCheckeur</div>
      <div style="color:rgba(255,255,255,.7);font-size:.75rem;margin-top:.2rem">
        NLP · ISE3 ENEAM · Prof. Hounna
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration NLP")
    model_choice = st.selectbox(
        "Modèle Transformer",
        ["DistilBERT ⚡", "BERT 🎯"],
        help="DistilBERT : 66M params, ~42ms · BERT : 110M params, ~118ms",
    )
    threshold = st.slider(
        "Seuil de neutralité", 0.50, 0.90, 0.70, 0.05,
        help="Confiance < seuil → classé NEUTRE",
    )
    n_batch = st.slider("Avis à analyser (IMDB)", 20, 200, 50, 10)

    st.markdown("---")
    st.markdown("### 🤖 Google Gemini AI")
    gemini_key = st.text_input(
        "Clé API Google AI Studio",
        value="AIzaSyDeT2D2zalWm3cPC5K7IFpbCSibK43l87s",
        type="password",
        help="Obtenez votre clé sur https://aistudio.google.com",
    )
    if gemini_key and gemini_key.startswith("AIza"):
        st.success("✅ Clé API Google Gemini configurée")
    elif gemini_key:
        st.warning("⚠️ Format de clé inhabituel (devrait commencer par AIza…)")
    else:
        st.markdown("""
        <div style="background:#fff8ec;border:1px solid #fed7aa;border-radius:8px;
                    padding:.7rem .9rem;font-size:.8rem;color:#92400e">
          📌 Obtenez votre clé gratuite :<br>
          <a href="https://aistudio.google.com" target="_blank"
             style="color:#7c3aed;font-weight:600">aistudio.google.com</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎨 Template PDF")
    # Par défaut, on sélectionne le template violet
    default_index = list(COLOR_TEMPLATES.keys()).index("🟣 Violet Pro (défaut)")
    template_name = st.selectbox("Palette de couleurs", list(COLOR_TEMPLATES.keys()), index=default_index)
    template      = COLOR_TEMPLATES[template_name]
    st.markdown(
        f'<div style="display:flex;gap:6px;margin-top:8px;align-items:center">'
        f'<div style="width:26px;height:26px;border-radius:6px;background:{template["pos"]}"></div>'
        f'<div style="width:26px;height:26px;border-radius:6px;background:{template["neg"]}"></div>'
        f'<div style="width:26px;height:26px;border-radius:6px;background:{template["neu"]}"></div>'
        f'<span style="font-size:.72rem;color:#888;margin-left:4px">Positif · Négatif · Neutre</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📚 Notions NLP intégrées")
    st.markdown("""
    <div style="font-size:.81rem;color:#555;line-height:2">
      ✅ Tokenisation<br>
      ✅ Stop words<br>
      ✅ Stemming<br>
      ✅ Nuage de mots<br>
      ✅ Polarité TextBlob<br>
      ✅ BOW / TF-IDF<br>
      ✅ Régression logistique<br>
      ✅ Matrice de confusion<br>
      ✅ DistilBERT / BERT<br>
      ✅ Train / Test split
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ISE3 ENEAM · Emmanuella TSIPOAKA · 2025-2026")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>✅ AvisCheckeur</h1>
  <p>Analyse de sentiments intelligente · ISE3 ENEAM · Prof. Gracieux Hounna</p>
</div>
""", unsafe_allow_html=True)

with st.spinner(f"⏳ Chargement de {model_choice}…"):
    classifier = load_transformer(model_choice)
st.success(f"✅ **{model_choice}** chargé et prêt !")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Analyser un avis",
    "📂 Importer un fichier",
    "📋 Tableau de bord",
    "🔬 Prétraitement NLP",
    "📊 Visualisations",
    "🏆 Évaluation",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSER UN AVIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        st.markdown('<div class="sec-title">✍️ Saisir un avis</div>', unsafe_allow_html=True)
        reviewer_name = st.text_input("Nom du reviewer", value="Anonyme")
        user_text = st.text_area(
            "Texte de l'avis (anglais)",
            height=160,
            placeholder="Ex : This movie is absolutely brilliant! The acting was superb…",
        )
        examples = [
            "This film is an absolute masterpiece. The acting, direction and story are all top-notch.",
            "Terrible waste of time. Boring plot, wooden acting, and no coherent story at all.",
            "The movie was okay. Some good scenes but overall a bit disappointing for what it promised.",
            "One of the best films I have ever seen! Moved me to tears. Highly recommended.",
        ]
        ex = st.selectbox("Ou choisir un exemple :", ["— exemples —"] + examples)
        if ex != "— exemples —":
            user_text = ex
        analyze_btn = st.button("🔍 Analyser cet avis", use_container_width=True)

    with col_out:
        st.markdown('<div class="sec-title">📊 Résultat de l\'analyse</div>', unsafe_allow_html=True)
        if analyze_btn and user_text.strip():
            with st.spinner("Analyse en cours…"):
                t0  = time.time()
                raw = classifier(user_text)[0]
                ms  = (time.time() - t0) * 1000
            sent, conf = score_to_sentiment(raw["score"], raw["label"], threshold)
            tb = get_textblob_scores(user_text)
            st.markdown(review_card_html(user_text, sent, conf, reviewer_name), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                cls = "" if sent == "POSITIVE" else (" neg" if sent == "NEGATIVE" else " neu")
                st.markdown(
                    f'<div class="stat-box"><div class="num">'
                    f'<span class="score-pill{cls}">{conf:.0%}</span></div>'
                    f'<div class="lbl">Confiance Transformer</div></div>',
                    unsafe_allow_html=True)
            with c2:
                col_c = "#7c3aed" if tb["polarity"] > 0 else "#e63946"
                st.markdown(
                    f'<div class="stat-box"><div class="num" style="color:{col_c}">'
                    f'{tb["polarity"]:+.2f}</div>'
                    f'<div class="lbl">Polarité TextBlob</div></div>',
                    unsafe_allow_html=True)
            with c3:
                st.markdown(
                    f'<div class="stat-box"><div class="num" style="color:#888">'
                    f'{tb["subjectivity"]:.2f}</div>'
                    f'<div class="lbl">Subjectivité</div></div>',
                    unsafe_allow_html=True)
            keywords = extract_keywords(user_text)
            if keywords:
                kw_cls = "nlp-tag" if sent == "POSITIVE" else "nlp-tag neg"
                tags   = " ".join(f'<span class="{kw_cls}">{w}</span>' for w in keywords[:8])
                st.markdown(f"**🔑 Mots-clés extraits :** {tags}", unsafe_allow_html=True)
            st.caption(f"⏱ Temps d'inférence : {ms:.0f} ms · Modèle : {model_choice}")
        else:
            st.markdown("""
            <div class="empty-state">
              <div class="icon">🎬</div>
              <div class="title">Entrez un avis pour le voir apparaître ici</div>
              <div class="sub">sous forme de carte d'avis au style Trustpilot</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IMPORTER UN FICHIER (identique à l'original, juste la couleur change)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">📂 Importer vos propres données</div>', unsafe_allow_html=True)
    col_up, col_info = st.columns([1, 1], gap="large")
    with col_up:
        st.markdown("**Formats supportés :** CSV · Excel (.xlsx/.xls) · TSV")
        uploaded_file = st.file_uploader(
            "Déposer votre fichier ici",
            type=["csv", "xlsx", "xls", "tsv"],
            help="Doit contenir au moins une colonne de texte d'avis.",
        )
        if uploaded_file is not None:
            try:
                preview_df, detected_col = load_custom_file(uploaded_file)
                st.success(f"✅ **{len(preview_df)} lignes** chargées · Colonne détectée : **{detected_col}**")
                all_text_cols = preview_df.select_dtypes(include="object").columns.tolist()
                idx = all_text_cols.index(detected_col) if detected_col in all_text_cols else 0
                selected_col = st.selectbox("Colonne de texte à analyser :", all_text_cols, index=idx)
                st.dataframe(preview_df[[selected_col]].head(4), use_container_width=True)
                max_rows = min(len(preview_df), 500)
                n_file   = st.slider("Nombre d'avis à analyser", 10, max_rows, min(50, max_rows), 10)
                if st.button("🚀 Lancer l'analyse du fichier", use_container_width=True):
                    uploaded_file.seek(0)
                    df_custom, _ = load_custom_file(uploaded_file, selected_col)
                    df_custom    = df_custom.head(n_file).reset_index(drop=True)
                    results      = []
                    bar          = st.progress(0)
                    with st.spinner(f"Classification de {len(df_custom)} avis…"):
                        for i, row in enumerate(df_custom.itertuples()):
                            raw_pred = classifier(str(row.review_clean)[:512])[0]
                            s, c = score_to_sentiment(raw_pred["score"], raw_pred["label"], threshold)
                            tb = get_textblob_scores(str(row.review_clean)[:300])
                            results.append({
                                "review":         str(row.review)[:200],
                                "sentiment_vrai":  str(row.sentiment),
                                "sentiment_pred":  s,
                                "confidence":      round(c, 3),
                                "polarity":        tb["polarity"],
                                "subjectivity":    tb["subjectivity"],
                                "nb_mots":         len(str(row.review_clean).split()),
                                "reviewer":        NAMES_LIST[i % len(NAMES_LIST)],
                            })
                            bar.progress((i + 1) / len(df_custom))
                        bar.empty()
                    df_r = pd.DataFrame(results)
                    st.session_state["df_results"]  = df_r
                    st.session_state["last_n"]      = len(df_r)
                    st.session_state["data_source"] = f"Fichier importé : {uploaded_file.name}"
                    st.session_state.pop("ai_summary", None)
                    st.session_state.pop("pdf_bytes", None)
                    st.success(f"✅ **{len(df_r)} avis analysés** avec succès ! → Consultez le **Tableau de bord**")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")
    with col_info:
        st.markdown("""
        <div class="info-box">
          <h4 style="color:#5b21b6;margin:0 0 .7rem">📋 Format attendu</h4>
          <b>Colonnes auto-détectées :</b><br>
          <code>review</code> · <code>text</code> · <code>comment</code> ·
          <code>avis</code> · <code>texte</code> · <code>content</code>
          <br><br>
          <b>Colonne optionnelle :</b><br>
          <code>sentiment</code> = <code>positive</code> / <code>negative</code>
        </div>
        """, unsafe_allow_html=True)
        sample_csv = pd.DataFrame({
            "review": [
                "This product is amazing! Exceeded all expectations.",
                "Very disappointing. Poor quality and bad service.",
                "It was okay. Nothing special but does the job.",
                "Incredible quality! Would definitely recommend.",
                "Not what I expected. Will not buy again.",
            ],
            "sentiment": ["positive", "negative", "positive", "positive", "negative"],
        }).to_csv(index=False)
        st.download_button("⬇️ Télécharger un modèle CSV exemple", sample_csv, "modele_avis.csv", "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TABLEAU DE BORD (identique, la couleur est déjà gérée via CSS)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">📋 Tableau de bord</div>', unsafe_allow_html=True)
    possible_paths = ["IMDB_Dataset.csv", "/content/sentimentscope/IMDB_Dataset.csv", "./IMDB_Dataset.csv"]
    csv_path    = next((p for p in possible_paths if os.path.exists(p)), None)
    data_source = st.session_state.get("data_source", "IMDB Dataset")
    if "df_results" not in st.session_state:
        if csv_path is None:
            st.info("💡 Importez un fichier via **📂 Importer un fichier** ou uploadez le dataset IMDB ci-dessous.")
            uploaded_csv = st.file_uploader("Uploader IMDB_Dataset.csv", type="csv", key="imdb_upload")
            if uploaded_csv:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded_csv.read())
                tmp.flush()
                csv_path = tmp.name
            else:
                st.stop()
        with st.spinner("Chargement du dataset IMDB…"):
            df = load_imdb(csv_path, n=n_batch)
        results = []
        bar = st.progress(0)
        with st.spinner(f"Classification de {len(df)} avis avec {model_choice}…"):
            for i, row in enumerate(df.itertuples()):
                raw = classifier(str(row.review_clean)[:512])[0]
                s, c = score_to_sentiment(raw["score"], raw["label"], threshold)
                tb   = get_textblob_scores(str(row.review_clean)[:300])
                results.append({
                    "review":         str(row.review)[:200],
                    "sentiment_vrai":  row.sentiment,
                    "sentiment_pred":  s,
                    "confidence":      round(c, 3),
                    "polarity":        tb["polarity"],
                    "subjectivity":    tb["subjectivity"],
                    "nb_mots":         len(str(row.review_clean).split()),
                    "reviewer":        NAMES_LIST[i % len(NAMES_LIST)],
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
    st.markdown(f'<div style="font-size:.8rem;color:#aaa;margin-bottom:.6rem">📁 Source : <b>{data_source}</b></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="trust-banner">'
          f'<span class="score-pill" style="font-size:1.7rem;padding:.45rem 1.1rem">{trust}</span>'
          f'<div>'
            f'<div style="font-size:1.05rem;font-weight:700;color:#111">TrustScore global</div>'
            f'<div style="color:#7c3aed;font-size:1.3rem;letter-spacing:4px">'
              f'{"★" * int(trust)}{"☆" * (5 - int(trust))}'
            f'</div>'
            f'<div style="font-size:.8rem;color:#aaa">Basé sur {len(df_r)} avis analysés</div>'
          f'</div>'
          f'<div style="margin-left:auto;text-align:right;font-size:.87rem;color:#555;line-height:2">'
            f'✅ <b>{counts.get("POSITIVE", 0)}</b> positifs<br>'
            f'❌ <b>{counts.get("NEGATIVE", 0)}</b> négatifs<br>'
            f'➖ <b>{counts.get("NEUTRAL",  0)}</b> neutres'
          f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    for col_w, (val, color, lbl) in zip([c1, c2, c3, c4], [
        (len(df_r),                           "#111",    "Avis analysés"),
        (counts.get("POSITIVE", 0),           "#7c3aed", "Positifs ★★★★★"),
        (counts.get("NEGATIVE", 0),           "#e63946", "Négatifs ★☆☆☆☆"),
        (f'{df_r["confidence"].mean():.1%}',  "#5b21b6", "Confiance moy."),
    ]):
        with col_w:
            st.markdown(
                f'<div class="stat-box">'
                  f'<div class="num" style="color:{color}">{val}</div>'
                  f'<div class="lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.markdown('<div class="sec-title">🤖 Résumé IA — Google Gemini</div>', unsafe_allow_html=True)
    if not gemini_key or not gemini_key.startswith("AIza"):
        st.markdown("""
        <div style="background:#fff8ec;border:1.5px solid #fed7aa;border-radius:12px;
                    padding:1rem 1.2rem;font-size:.9rem;color:#92400e">
          ⚠️ <b>Clé API Google Gemini invalide ou manquante.</b><br>
          Vérifiez la clé dans la <b>sidebar ⚙️</b> (doit commencer par <code>AIza…</code>).<br>
          Obtenez une clé gratuite sur <a href="https://aistudio.google.com" target="_blank"
             style="color:#7c3aed;font-weight:600">aistudio.google.com</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        _, btn_col = st.columns([3, 1])
        with btn_col:
            gen_btn = st.button("✨ Générer le résumé IA", use_container_width=True)
        if gen_btn:
            stats_text = "\n".join([
                f"- Total avis analysés : {len(df_r)}",
                f"- Avis positifs  : {counts.get('POSITIVE',0)} ({counts.get('POSITIVE',0)/max(len(df_r),1)*100:.1f}%)",
                f"- Avis négatifs  : {counts.get('NEGATIVE',0)} ({counts.get('NEGATIVE',0)/max(len(df_r),1)*100:.1f}%)",
                f"- Avis neutres   : {counts.get('NEUTRAL',0)} ({counts.get('NEUTRAL',0)/max(len(df_r),1)*100:.1f}%)",
                f"- Confiance moyenne : {df_r['confidence'].mean():.1%}",
                f"- Polarité moyenne  : {df_r['polarity'].mean():+.3f}",
                f"- Subjectivité moy. : {df_r['subjectivity'].mean():.3f}",
                f"- Longueur moy. avis : {df_r['nb_mots'].mean():.0f} mots",
                f"- TrustScore global : {trust}/5.0",
                f"- Modèle utilisé    : {model_choice}",
            ])
            with st.spinner("🤖 Google Gemini analyse vos résultats… (peut prendre 5–10 s)"):
                try:
                    summary = call_gemini_summary(stats_text, gemini_key)
                    st.session_state["ai_summary"] = summary
                    st.success("✅ Résumé IA généré avec succès !")
                except RuntimeError as e:
                    st.error(f"❌ Erreur API Gemini : {e}")
                    st.session_state.pop("ai_summary", None)
        if "ai_summary" in st.session_state:
            st.markdown(
                f'<div class="ai-box">'
                  f'<h4>🤖 Analyse Gemini AI</h4>'
                  f'<div class="ai-content">{st.session_state["ai_summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div style="background:#f8f8f8;border:1.5px dashed #ddd;border-radius:12px;
                        padding:1.2rem;text-align:center;color:#999;font-size:.9rem">
              Cliquez sur <b>✨ Générer le résumé IA</b> pour obtenir une analyse intelligente
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div class="sec-title">📥 Export PDF</div>', unsafe_allow_html=True)
    col_pdf1, col_pdf2 = st.columns([2, 1])
    with col_pdf1:
        st.markdown(f"Template sélectionné : **{template_name}** _(modifier dans la sidebar ⚙️)_")
        if "ai_summary" not in st.session_state:
            st.info("💡 Astuce : générez d'abord le résumé IA pour l'inclure dans le PDF.")
    with col_pdf2:
        gen_pdf_btn = st.button("📄 Générer le rapport PDF", use_container_width=True)
    if gen_pdf_btn:
        with st.spinner("Génération du PDF en cours… (quelques secondes)"):
            try:
                pdf_bytes = generate_pdf_report(df_r, st.session_state.get("ai_summary", ""),
                                                template_name, template, model_choice, len(df_r))
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success(f"✅ Rapport PDF généré ! ({len(pdf_bytes)//1024} Ko · {len(df_r)} avis · template {template_name})")
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération PDF : {e}")
                import traceback
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())
    if "pdf_bytes" in st.session_state:
        st.download_button(
            label="⬇️ Télécharger le rapport PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"AvisCheckeur_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    st.markdown("---")
    st.markdown('<div class="sec-title">🃏 Derniers avis analysés</div>', unsafe_allow_html=True)
    for _, row in df_r.head(6).iterrows():
        st.markdown(review_card_html(row["review"], row["sentiment_pred"], row["confidence"], row["reviewer"]), unsafe_allow_html=True)
    with st.expander("📋 Voir tous les résultats (tableau complet)"):
        st.dataframe(df_r, use_container_width=True)
        st.download_button("⬇️ Télécharger CSV", df_r.to_csv(index=False), "avis_results.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRÉTRAITEMENT NLP (identique)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-title">🔬 Pipeline Prétraitement NLP (cours ISE3)</div>', unsafe_allow_html=True)
    sample_text = st.text_area(
        "Texte à prétraiter :",
        value="This film is absolutely brilliant! The director did a wonderful job creating magic.",
        height=100,
    )
    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1: do_lower = st.checkbox("🔡 Lowercasing",  True)
    with col_o2: do_stop  = st.checkbox("🚫 Stop words",   True)
    with col_o3: do_stem  = st.checkbox("✂️ Stemming",     False)
    if st.button("▶️ Appliquer le pipeline NLP", use_container_width=True):
        st.markdown("**Étape 1 — Texte brut**")
        st.code(sample_text)
        lowered = sample_text.lower() if do_lower else sample_text
        if do_lower:
            st.markdown("**Étape 2 — Lowercasing**")
            st.code(lowered)
        clean  = re.sub(r"[^a-zA-Z\s]", " ", lowered)
        tokens = clean.split()
        st.markdown(f"**Étape 3 — Tokenisation** → **{len(tokens)} tokens**")
        st.code(str(tokens[:20]))
        if do_stop:
            STOP_W = {"the","a","an","and","or","but","in","on","at","to","for","of","with","this","that","it","is","was","are","were","be","been","have","has","i","my","we","our","you","your","they","do","did","does",}
            tokens = [t for t in tokens if t not in STOP_W and len(t) > 2]
            st.markdown(f"**Étape 4 — Suppression stop words** → **{len(tokens)} tokens restants**")
            st.code(str(tokens[:20]))
        if do_stem:
            try:
                import nltk
                nltk.download("punkt", quiet=True)
                from nltk.stem import PorterStemmer
                ps    = PorterStemmer()
                stems = [ps.stem(t) for t in tokens[:12]]
                st.markdown("**Étape 5 — Stemming**")
                st.dataframe(pd.DataFrame({"Token original": tokens[:12], "Forme réduite (stem)": stems}), use_container_width=True)
            except Exception:
                st.info("ℹ️ NLTK non disponible pour le stemming dans cet environnement.")
        st.markdown("---")
        freq   = Counter(tokens)
        df_bow = pd.DataFrame(freq.most_common(14), columns=["Token", "Fréquence"])
        fig_bow = px.bar(df_bow, x="Token", y="Fréquence", title="Bag of Words — Top 14 tokens", color_discrete_sequence=[template["primary"]])
        fig_bow.update_layout(height=290, margin=dict(t=45, b=0))
        st.plotly_chart(fig_bow, use_container_width=True)
        st.markdown("---")
        st.markdown("**Polarité & Subjectivité — TextBlob**")
        tb = get_textblob_scores(sample_text)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Polarité", f'{tb["polarity"]:+.3f}', help="−1 = très négatif · 0 = neutre · +1 = très positif")
        with c2:
            st.metric("Subjectivité", f'{tb["subjectivity"]:.3f}', help="0 = objectif (faits) · 1 = subjectif (opinion)")
        st.markdown("---")
        st.markdown("**Nuage de mots**")
        img = make_wordcloud_img(" ".join(tokens), template["primary"], template["cmap_pos"])
        if img:
            st.markdown(f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:12px;border:1px solid #ececec">', unsafe_allow_html=True)
        else:
            st.info("ℹ️ Nuage de mots non disponible (wordcloud non installé).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">📊 Visualisations & Tendances</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">📊</div>
          <div class="title">Aucune donnée disponible</div>
          <div class="sub">Lancez d'abord l'analyse dans <b>Tableau de bord</b> ou <b>Importer un fichier</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_v = st.session_state["df_results"]
        cmap_c = {"POSITIVE": template["pos"], "NEGATIVE": template["neg"], "NEUTRAL": template["neu"]}
        col_a, col_b = st.columns(2)
        with col_a:
            cnts = df_v["sentiment_pred"].value_counts().reset_index()
            cnts.columns = ["sentiment", "count"]
            fig_d = px.pie(cnts, values="count", names="sentiment", color="sentiment", color_discrete_map=cmap_c, hole=.58, title="Répartition des sentiments")
            fig_d.update_traces(textposition="outside", textinfo="percent+label")
            fig_d.update_layout(showlegend=False, height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_d, use_container_width=True)
        with col_b:
            fig_h = px.histogram(df_v, x="confidence", color="sentiment_pred", color_discrete_map=cmap_c, nbins=15, barmode="overlay", opacity=.75, title="Distribution des scores de confiance")
            fig_h.update_layout(height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_h, use_container_width=True)
        col_c, col_d = st.columns(2)
        with col_c:
            fig_sc = px.scatter(df_v, x="polarity", y="confidence", color="sentiment_pred", color_discrete_map=cmap_c, title="Polarité TextBlob vs Confiance Transformer", labels={"polarity": "Polarité (TextBlob)", "confidence": "Confiance (Transformer)"}, opacity=.7)
            fig_sc.update_layout(height=320, margin=dict(t=50, b=0))
            st.plotly_chart(fig_sc, use_container_width=True)
        with col_d:
            fig_bx = px.box(df_v, x="sentiment_pred", y="nb_mots", color="sentiment_pred", color_discrete_map=cmap_c, title="Longueur des avis par sentiment", labels={"nb_mots": "Nombre de mots", "sentiment_pred": "Sentiment"})
            fig_bx.update_layout(height=320, showlegend=False, margin=dict(t=50, b=0))
            st.plotly_chart(fig_bx, use_container_width=True)
        st.markdown('<div class="sec-title">☁️ Nuages de mots comparatifs</div>', unsafe_allow_html=True)
        col_w1, col_w2 = st.columns(2)
        pos_t = " ".join(df_v[df_v["sentiment_pred"] == "POSITIVE"]["review"].str[:200])
        neg_t = " ".join(df_v[df_v["sentiment_pred"] == "NEGATIVE"]["review"].str[:200])
        with col_w1:
            st.markdown("**Avis Positifs ✅**")
            img = make_wordcloud_img(pos_t, template["pos"], template["cmap_pos"])
            if img:
                st.markdown(f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:12px;border:1px solid #ececec">', unsafe_allow_html=True)
        with col_w2:
            st.markdown("**Avis Négatifs ❌**")
            img2 = make_wordcloud_img(neg_t, template["neg"], template["cmap_neg"])
            if img2:
                st.markdown(f'<img src="data:image/png;base64,{img2}" style="width:100%;border-radius:12px;border:1px solid #ececec">', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ÉVALUATION (identique)
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-title">🏆 Évaluation & Comparaison des modèles</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🏆</div>
          <div class="title">Aucune donnée disponible</div>
          <div class="sub">Lancez d'abord l'analyse dans <b>Tableau de bord</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
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
        rows = [{"Modèle": f"✅ {model_choice}", "Accuracy": f"{acc_tr:.3f}", "Précision": f"{prec_tr:.3f}", "Rappel": f"{rec_tr:.3f}", "F1-Score": f"{f1_tr:.3f}", "Latence": "~42ms" if "Distil" in model_choice else "~118ms", "Paramètres": "66M" if "Distil" in model_choice else "110M"}]
        if baseline_ok:
            rows.append({"Modèle": "📊 TF-IDF + LogReg (baseline)", "Accuracy": f"{acc_lr:.3f}", "Précision": f"{prec_lr:.3f}", "Rappel": f"{rec_lr:.3f}", "F1-Score": f"{f1_lr:.3f}", "Latence": "~2ms", "Paramètres": "~500K"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown('<div class="sec-title">🔢 Matrices de confusion</div>', unsafe_allow_html=True)
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            cm = confusion_matrix(df_e["true_bin"], df_e["pred_bin"])
            fig_cm = px.imshow(cm, x=["Prédit NEG", "Prédit POS"], y=["Réel NEG", "Réel POS"], color_continuous_scale=[[0, "#fff"], [1, template["primary"]]], text_auto=True, title=f"Matrice — {model_choice}")
            fig_cm.update_layout(height=310, margin=dict(t=55, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_cm2:
            if baseline_ok:
                cm2 = confusion_matrix(y_te, y_pred_lr)
                fig_cm2 = px.imshow(cm2, x=["Prédit NEG", "Prédit POS"], y=["Réel NEG", "Réel POS"], color_continuous_scale=[[0, "#fff"], [1, template["secondary"]]], text_auto=True, title="Matrice — TF-IDF + LogReg")
                fig_cm2.update_layout(height=310, margin=dict(t=55, b=10))
                st.plotly_chart(fig_cm2, use_container_width=True)
        st.markdown('<div class="sec-title">📡 Radar comparatif des métriques</div>', unsafe_allow_html=True)
        fig_r = go.Figure()
        cats = ["Accuracy", "Précision", "Rappel", "F1-Score"]
        v_tr = [acc_tr, prec_tr, rec_tr, f1_tr]
        fig_r.add_trace(go.Scatterpolar(r=v_tr + [v_tr[0]], theta=cats + [cats[0]], fill="toself", name=model_choice, line_color=template["primary"], opacity=.75))
        if baseline_ok:
            v_lr = [acc_lr, prec_lr, rec_lr, f1_lr]
            fig_r.add_trace(go.Scatterpolar(r=v_lr + [v_lr[0]], theta=cats + [cats[0]], fill="toself", name="TF-IDF + LogReg", line_color="#aaa", opacity=.5))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])), height=380, margin=dict(t=30, b=20))
        st.plotly_chart(fig_r, use_container_width=True)
        delta = f1_tr - (f1_lr if baseline_ok else 0)
        gain_str = f"  Gain vs baseline : <b>{delta:+.3f}</b>" if baseline_ok else ""
        st.markdown(
            f'<div style="background:#f5f3ff;border-left:4px solid {template["primary"]};'
            f'padding:1rem 1.2rem;border-radius:0 12px 12px 0;font-size:.9rem;line-height:1.7">'
            f'<b>📋 Interprétation :</b> Le modèle <b>{model_choice}</b> atteint '
            f'un F1-Score de <b>{f1_tr:.3f}</b> sur {len(df_e)} avis.{gain_str}<br>'
            f'La classe <b>NEUTRE</b> (seuillage à {threshold:.0%}) est la plus difficile '
            f'à distinguer des classes binaires natives du dataset.</div>',
            unsafe_allow_html=True,
        )
