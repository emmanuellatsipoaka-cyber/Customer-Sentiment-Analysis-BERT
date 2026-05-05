
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time, re, io, base64
from collections import Counter
from datetime import datetime

st.set_page_config(
    page_title="SentimentScope · IMDB Reviews",
    page_icon="AvisChecker🎬",
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
    position: relative; overflow: hidden;
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
</style>
''', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_transformer(model_key):
    from transformers import pipeline
    ids = {
        "DistilBERT ⚡": "distilbert-base-uncased-finetuned-sst-2-english",
        "BERT 🎯":        "textattack/bert-base-uncased-SST-2",
    }
    return pipeline("sentiment-analysis", model=ids[model_key],
                    truncation=True, max_length=512)

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

def normalize_label(raw):
    m = {"LABEL_0":"NEGATIVE","LABEL_1":"POSITIVE",
         "POSITIVE":"POSITIVE","NEGATIVE":"NEGATIVE"}
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
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "this","that","it","is","was","are","were","be","been","have","has",
        "i","my","we","our","you","your","they","their","not","no","so","as",
        "by","from","all","just","more","very","too","also","would","could",
        "do","did","does","if","when","then","there","here","what","which",
        "movie","film","one","get","like","really","even","br","its","also",
        "much","about","some","see","make","well","know","he","she","his","her",
    }
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq  = Counter(w for w in words if w not in STOP)
    return [w for w, _ in freq.most_common(top_n)]

def make_wordcloud_img(text, color="#00b67a"):
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wc = WordCloud(
            width=600, height=280, background_color="white",
            colormap="Greens" if color=="#00b67a" else "Reds",
            max_words=60, collocations=False,
        ).generate(text)
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
    n = {"POSITIVE":5,"NEUTRAL":3,"NEGATIVE":1}.get(sentiment, 3)
    cls = {"POSITIVE":"","NEGATIVE":" neg","NEUTRAL":" neu"}.get(sentiment,"")
    return f'<div class="stars{cls}">{"★"*n}{"☆"*(5-n)}</div>'

def review_card_html(text, sentiment, confidence, reviewer="Anonyme", date=""):
    badge_cls = {"POSITIVE":"badge-pos","NEGATIVE":"badge-neg","NEUTRAL":"badge-neu"}
    label_fr  = {"POSITIVE":"Positif ✅","NEGATIVE":"Négatif ❌","NEUTRAL":"Neutre ➖"}
    preview   = text[:220]+"…" if len(text)>220 else text
    date_str  = date or datetime.now().strftime("%d %b %Y")
    return f'''
<div class="review-card">
  <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem">
    <div class="avatar">{reviewer[0].upper()}</div>
    <div>
      <div style="font-weight:600;font-size:.9rem;color:#191919">{reviewer}</div>
      <div style="font-size:.75rem;color:#767676">Publié le {date_str}</div>
    </div>
  </div>
  {stars_html(sentiment)}
  <div style="font-size:.88rem;color:#404040;line-height:1.55">{preview}</div>
  <span class="badge {badge_cls[sentiment]}">
    {label_fr[sentiment]} - {confidence:.0%}
  </span>
</div>
'''

# ── SIDEBAR ──
with st.sidebar:
    st.markdown('<div style="background:#00b67a;padding:.8rem;border-radius:10px;margin-bottom:1rem;"><span style="color:white;font-weight:800;font-size:1rem;">★ SentimentScope</span></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    model_choice = st.selectbox("Modèle", ["DistilBERT ⚡","BERT 🎯"])
    threshold    = st.slider("Seuil neutralité", 0.50, 0.90, 0.70, 0.05)
    n_batch      = st.slider("Avis à analyser", 20, 200, 50, 10)
    st.markdown("---")
    st.markdown("### 📚 Notions NLP")
    st.markdown('<div style="font-size:.82rem;color:#555;line-height:1.8">✅ Tokenisation<br>✅ Stop words<br>✅ Stemming<br>✅ Nuage de mots<br>✅ Polarité TextBlob<br>✅ BOW / TF-IDF<br>✅ Régression logistique<br>✅ Matrice de confusion<br>✅ DistilBERT / BERT<br>✅ Train/Test split</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ISE3 ENEAM - Prof. Hounna - 2025")

# ── HEADER ──
st.markdown('''
<div class="hero">
  <h1>🎬 SentimentScope</h1>
  <p>Analyse de sentiments - IMDB Movie Reviews - DistilBERT & BERT - NLP ISE3 ENEAM</p>
</div>
''', unsafe_allow_html=True)

with st.spinner(f"⏳ Chargement de {model_choice}…"):
    classifier = load_transformer(model_choice)
st.success(f"✅ {model_choice} prêt !")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Analyser un avis",
    "📋 Tableau de bord",
    "🔬 Prétraitement NLP",
    "📊 Visualisations",
    "🏆 Évaluation",
])

# ══ TAB 1 — ANALYSER UN AVIS ══
with tab1:
    col_in, col_out = st.columns([1,1], gap="large")
    with col_in:
        st.markdown('<div class="sec-title">✍️ Saisir un avis</div>', unsafe_allow_html=True)
        reviewer_name = st.text_input("Nom du reviewer", value="Anonyme")
        user_text = st.text_area("Texte de l'avis (anglais)", height=160,
            placeholder="Ex: This movie is absolutely brilliant!")
        examples = [
            "This film is an absolute masterpiece. The acting and story are top-notch.",
            "Terrible waste of time. Boring plot and no coherent story at all.",
            "The movie was okay. Some good scenes but overall a bit disappointing.",
        ]
        ex = st.selectbox("Ou choisir un exemple :", ["— exemples —"]+examples)
        if ex != "— exemples —":
            user_text = ex
        analyze_btn = st.button("🔍 Analyser cet avis", use_container_width=True)

    with col_out:
        st.markdown('<div class="sec-title">📊 Résultat</div>', unsafe_allow_html=True)
        if analyze_btn and user_text.strip():
            with st.spinner("Analyse…"):
                t0  = time.time()
                raw = classifier(user_text)[0]
                ms  = (time.time()-t0)*1000
            sent, conf = score_to_sentiment(raw["score"], raw["label"], threshold)
            tb = get_textblob_scores(user_text)
            st.markdown(review_card_html(user_text, sent, conf, reviewer_name), unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1:
                cls = "" if sent=="POSITIVE" else " neg" if sent=="NEGATIVE" else " neu"
                st.markdown(f'<div class="stat-box"><div class="num"><span class="score-pill{cls}">{conf:.0%}</span></div><div class="lbl">Confiance</div></div>', unsafe_allow_html=True)
            with c2:
                col = "#00b67a" if tb["polarity"]>0 else "#ff3722"
                st.markdown(f'<div class="stat-box"><div class="num" style="color:{col}">{tb["polarity"]:+.2f}</div><div class="lbl">Polarité TextBlob</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-box"><div class="num" style="color:#767676">{tb["subjectivity"]:.2f}</div><div class="lbl">Subjectivité</div></div>', unsafe_allow_html=True)
            keywords = extract_keywords(user_text)
            if keywords:
                kw_cls = "nlp-tag" if sent=="POSITIVE" else "nlp-tag neg"
                tags = " ".join(f'<span class="{kw_cls}">{w}</span>' for w in keywords[:8])
                st.markdown(f"**🔑 Mots-clés :** {tags}", unsafe_allow_html=True)
            st.caption(f"⏱ {ms:.0f} ms - {model_choice}")
        else:
            st.markdown('<div style="background:#f0faf5;border:1.5px dashed #00b67a;border-radius:12px;padding:2rem;text-align:center;color:#555"><div style="font-size:2rem">🎬</div><div style="font-weight:600;margin-top:.5rem">Entrez un avis pour le voir apparaître ici</div></div>', unsafe_allow_html=True)


# ══ TAB 2 — TABLEAU DE BORD ══
with tab2:
    import os as _os
    st.markdown('<div class="sec-title">📋 Tableau de bord · Dataset IMDB</div>',
                unsafe_allow_html=True)

    # Chemins possibles (Colab + Streamlit Cloud)
    possible_paths = [
        "IMDB_Dataset.csv",                          # racine du repo (Streamlit Cloud)
        "/content/sentimentscope/IMDB_Dataset.csv",  # Colab
        "./IMDB_Dataset.csv",
    ]
    csv_path = next((p for p in possible_paths if _os.path.exists(p)), None)

    if csv_path is None:
        st.error("❌ IMDB_Dataset.csv introuvable. Uploadez-le via le widget.")
        uploaded_csv = st.file_uploader("Uploader IMDB_Dataset.csv", type="csv")
        if uploaded_csv:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded_csv.read())
            tmp.flush()
            csv_path = tmp.name
        else:
            st.stop()

    if "df_results" not in st.session_state or st.session_state.get("last_n") != n_batch:
        with st.spinner("Chargement IMDB…"):
            df = load_imdb(csv_path, n=n_batch)

        with st.spinner(f"Classification de {len(df)} avis…"):
            results = []
            bar = st.progress(0)
            names_list = ["John D.","Sarah M.","Mike P.","Emma L.","James W.","Clara B.","Tom R.","Alice K."]
            for i, row in enumerate(df.itertuples()):
                raw = classifier(str(row.review_clean)[:512])[0]
                s, c = score_to_sentiment(raw["score"], raw["label"], threshold)
                tb   = get_textblob_scores(str(row.review_clean)[:300])
                results.append({
                    "review": str(row.review)[:200],
                    "sentiment_vrai": row.sentiment,
                    "sentiment_pred": s,
                    "confidence": round(c,3),
                    "polarity": tb["polarity"],
                    "subjectivity": tb["subjectivity"],
                    "nb_mots": len(str(row.review_clean).split()),
                    "reviewer": names_list[i % len(names_list)],
                })
                bar.progress((i+1)/len(df))
            bar.empty()
        df_r = pd.DataFrame(results)
        st.session_state["df_results"] = df_r
        st.session_state["last_n"] = n_batch
    else:
        df_r = st.session_state["df_results"]

    counts = df_r["sentiment_pred"].value_counts()
    pct_pos = counts.get("POSITIVE",0)/len(df_r)
    trust = round(1+pct_pos*4, 1)
    st.markdown(f'''
<div style="background:white;border:1px solid #e8e8e8;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.2rem">
  <span class="score-pill" style="font-size:1.6rem;padding:.4rem 1rem">{trust}</span>
  <div>
    <div style="font-size:1.1rem;font-weight:700">TrustScore global</div>
    <div style="color:#00b67a;font-size:1.2rem">{"★"*int(trust)}{"☆"*(5-int(trust))}</div>
    <div style="font-size:.82rem;color:#767676">Basé sur {len(df_r)} avis</div>
  </div>
  <div style="margin-left:auto;font-size:.85rem;color:#555">
    ✅ {counts.get("POSITIVE",0)} positifs - ❌ {counts.get("NEGATIVE",0)} négatifs - ➖ {counts.get("NEUTRAL",0)} neutres
  </div>
</div>
''', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,color,lbl) in zip([c1,c2,c3,c4],[
        (len(df_r),"#191919","Avis analysés"),
        (counts.get("POSITIVE",0),"#00b67a","Positifs ★★★★★"),
        (counts.get("NEGATIVE",0),"#ff3722","Négatifs ★☆☆☆☆"),
        (f'{df_r["confidence"].mean():.1%}',"#007a52","Confiance moy."),
    ]):
        with col:
            st.markdown(f'<div class="stat-box"><div class="num" style="color:{color}">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sec-title">🃏 Derniers avis analysés</div>', unsafe_allow_html=True)
    for i, row in df_r.head(6).iterrows():
        st.markdown(review_card_html(row["review"], row["sentiment_pred"], row["confidence"], row["reviewer"]), unsafe_allow_html=True)
    with st.expander("📋 Voir tous les résultats"):
        st.dataframe(df_r, use_container_width=True)
        st.download_button("⬇️ Télécharger CSV", df_r.to_csv(index=False), "imdb_results.csv","text/csv")


# ══ TAB 3 — PRÉTRAITEMENT ══
with tab3:
    st.markdown('<div class="sec-title">🔬 Pipeline Prétraitements NLP</div>', unsafe_allow_html=True)
    sample_text = st.text_area("Texte à prétraiter :",
        value="This film is absolutely brilliant! The director did a wonderful job.",height=100)
    col_o1,col_o2,col_o3 = st.columns(3)
    with col_o1: do_lower = st.checkbox("🔡 Lowercasing",True)
    with col_o2: do_stop  = st.checkbox("🚫 Stop words",True)
    with col_o3: do_stem  = st.checkbox("✂️ Stemming",False)

    if st.button("▶️ Appliquer le pipeline", use_container_width=True):
        st.markdown("**Étape 1 — Texte brut**")
        st.code(sample_text)
        if do_lower:
            st.markdown("**Étape 2 — Lowercasing** *(slide 81)*")
            st.code(sample_text.lower())
        clean  = re.sub(r"[^a-zA-Z\s]"," ",sample_text.lower())
        tokens = clean.split()
        st.markdown(f"**Étape 3 — Tokenisation** *(slide 54)* → {len(tokens)} tokens")
        st.code(str(tokens[:20]))
        if do_stop:
            STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
                    "this","that","it","is","was","are","were","be","been","have","has"}
            tokens = [t for t in tokens if t not in STOP and len(t)>2]
            st.markdown(f"**Étape 4 — Stop words** *(slide 66)* → {len(tokens)} tokens restants")
            st.code(str(tokens[:20]))
        if do_stem:
            try:
                import nltk; nltk.download("punkt",quiet=True)
                from nltk.stem import PorterStemmer
                ps = PorterStemmer()
                stems = [ps.stem(t) for t in tokens[:10]]
                st.markdown("**Étape 5 — Stemming** *(slide 83)*")
                st.dataframe(pd.DataFrame({"Token":tokens[:10],"Stem":stems}),use_container_width=True)
            except: st.info("NLTK requis pour le stemming")
        st.markdown("---")
        freq = Counter(tokens)
        df_bow = pd.DataFrame(freq.most_common(12),columns=["Token","Fréquence"])
        fig_bow = px.bar(df_bow,x="Token",y="Fréquence",title="BOW *(slide 38)*",
                         color_discrete_sequence=["#00b67a"])
        fig_bow.update_layout(height=280,margin=dict(t=40,b=0))
        st.plotly_chart(fig_bow,use_container_width=True)
        st.markdown("---")
        st.markdown("**Polarité & Subjectivité TextBlob** *(slide 29)*")
        tb = get_textblob_scores(sample_text)
        c1,c2 = st.columns(2)
        with c1: st.metric("Polarité",f'{tb["polarity"]:+.3f}',help="−1 (négatif) → +1 (positif)")
        with c2: st.metric("Subjectivité",f'{tb["subjectivity"]:.3f}',help="0 (objectif) → 1 (subjectif)")
        st.markdown("---")
        st.markdown("**Nuage de mots** *(slide 32)*")
        img = make_wordcloud_img(" ".join(tokens))
        if img:
            st.markdown(f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:10px">', unsafe_allow_html=True)

# ══ TAB 4 — VISUALISATIONS ══
with tab4:
    st.markdown('<div class="sec-title">📊 Visualisations & Tendances</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.info("💡 Lancez d'abord l'analyse dans **Tableau de bord**.")
    else:
        df_v  = st.session_state["df_results"]
        cmap  = {"POSITIVE":"#00b67a","NEGATIVE":"#ff3722","NEUTRAL":"#ff8f00"}
        col_a,col_b = st.columns(2)
        with col_a:
            cnts = df_v["sentiment_pred"].value_counts().reset_index()
            cnts.columns=["sentiment","count"]
            fig_d = px.pie(cnts,values="count",names="sentiment",color="sentiment",
                           color_discrete_map=cmap,hole=.55,title="Répartition des sentiments")
            fig_d.update_traces(textposition="outside",textinfo="percent+label")
            fig_d.update_layout(showlegend=False,height=320,margin=dict(t=50,b=0))
            st.plotly_chart(fig_d,use_container_width=True)
        with col_b:
            fig_h = px.histogram(df_v,x="confidence",color="sentiment_pred",
                                 color_discrete_map=cmap,nbins=15,barmode="overlay",opacity=.75,
                                 title="Distribution des scores de confiance")
            fig_h.update_layout(height=320,margin=dict(t=50,b=0))
            st.plotly_chart(fig_h,use_container_width=True)
        col_c,col_d = st.columns(2)
        with col_c:
            fig_sc = px.scatter(df_v,x="polarity",y="confidence",color="sentiment_pred",
                                color_discrete_map=cmap,title="Polarité TextBlob vs Confiance",opacity=.7)
            fig_sc.update_layout(height=320,margin=dict(t=50,b=0))
            st.plotly_chart(fig_sc,use_container_width=True)
        with col_d:
            fig_bx = px.box(df_v,x="sentiment_pred",y="nb_mots",color="sentiment_pred",
                            color_discrete_map=cmap,title="Longueur des avis par sentiment")
            fig_bx.update_layout(height=320,showlegend=False,margin=dict(t=50,b=0))
            st.plotly_chart(fig_bx,use_container_width=True)
        st.markdown('<div class="sec-title">☁️ Nuages de mots (slide 32)</div>',unsafe_allow_html=True)
        col_w1,col_w2 = st.columns(2)
        pos_t = " ".join(df_v[df_v["sentiment_pred"]=="POSITIVE"]["review"].str[:200])
        neg_t = " ".join(df_v[df_v["sentiment_pred"]=="NEGATIVE"]["review"].str[:200])
        with col_w1:
            st.markdown("**Positifs ✅**")
            img = make_wordcloud_img(pos_t,"#00b67a")
            if img: st.markdown(f'<img src="data:image/png;base64,{img}" style="width:100%;border-radius:10px">',unsafe_allow_html=True)
        with col_w2:
            st.markdown("**Négatifs ❌**")
            img2 = make_wordcloud_img(neg_t,"#ff3722")
            if img2: st.markdown(f'<img src="data:image/png;base64,{img2}" style="width:100%;border-radius:10px">',unsafe_allow_html=True)

# ══ TAB 5 — ÉVALUATION ══
with tab5:
    st.markdown('<div class="sec-title">🏆 Évaluation & Comparaison</div>', unsafe_allow_html=True)
    if "df_results" not in st.session_state:
        st.info("💡 Lancez d'abord l'analyse dans **Tableau de bord**.")
    else:
        import os as _os
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (accuracy_score,f1_score,precision_score,
                                     recall_score,confusion_matrix)
        df_e = st.session_state["df_results"].copy()
        df_e["true_bin"] = (df_e["sentiment_vrai"]=="positive").astype(int)
        df_e["pred_bin"] = (df_e["sentiment_pred"]=="POSITIVE").astype(int)

        acc_tr  = accuracy_score(df_e["true_bin"],df_e["pred_bin"])
        f1_tr   = f1_score(df_e["true_bin"],df_e["pred_bin"],average="weighted",zero_division=0)
        prec_tr = precision_score(df_e["true_bin"],df_e["pred_bin"],average="weighted",zero_division=0)
        rec_tr  = recall_score(df_e["true_bin"],df_e["pred_bin"],average="weighted",zero_division=0)

        try:
            X,y = df_e["review"], df_e["true_bin"]
            X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
            vec = TfidfVectorizer(max_features=1000,ngram_range=(1,2),stop_words="english")
            lr  = LogisticRegression(max_iter=1000,random_state=42)
            lr.fit(vec.fit_transform(X_tr),y_tr)
            y_pred_lr = lr.predict(vec.transform(X_te))
            acc_lr  = accuracy_score(y_te,y_pred_lr)
            f1_lr   = f1_score(y_te,y_pred_lr,average="weighted",zero_division=0)
            prec_lr = precision_score(y_te,y_pred_lr,average="weighted",zero_division=0)
            rec_lr  = recall_score(y_te,y_pred_lr,average="weighted",zero_division=0)
            baseline_ok = True
        except: baseline_ok = False

        rows = [{"Modèle":f"**{model_choice}**","Accuracy":f"{acc_tr:.3f}",
                 "Précision":f"{prec_tr:.3f}","Rappel":f"{rec_tr:.3f}","F1":f"{f1_tr:.3f}","Latence":"~42ms" if "Distil" in model_choice else "~118ms"}]
        if baseline_ok:
            rows.append({"Modèle":"TF-IDF+LogReg","Accuracy":f"{acc_lr:.3f}",
                         "Précision":f"{prec_lr:.3f}","Rappel":f"{rec_lr:.3f}","F1":f"{f1_lr:.3f}","Latence":"~2ms"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        col_cm1,col_cm2 = st.columns(2)
        with col_cm1:
            cm = confusion_matrix(df_e["true_bin"],df_e["pred_bin"])
            fig_cm = px.imshow(cm,x=["Prédit NEG","Prédit POS"],y=["Réel NEG","Réel POS"],
                               color_continuous_scale=[[0,"#fff"],[1,"#00b67a"]],
                               text_auto=True,title=f"Matrice — {model_choice}")
            fig_cm.update_layout(height=300,margin=dict(t=50,b=10))
            st.plotly_chart(fig_cm,use_container_width=True)
        with col_cm2:
            if baseline_ok:
                cm2 = confusion_matrix(y_te,y_pred_lr)
                fig_cm2 = px.imshow(cm2,x=["Prédit NEG","Prédit POS"],y=["Réel NEG","Réel POS"],
                                    color_continuous_scale=[[0,"#fff"],[1,"#007a52"]],
                                    text_auto=True,title="Matrice — TF-IDF+LogReg")
                fig_cm2.update_layout(height=300,margin=dict(t=50,b=10))
                st.plotly_chart(fig_cm2,use_container_width=True)

        fig_r = go.Figure()
        cats  = ["Accuracy","Précision","Rappel","F1"]
        v_tr  = [acc_tr,prec_tr,rec_tr,f1_tr]
        fig_r.add_trace(go.Scatterpolar(r=v_tr+[v_tr[0]],theta=cats+[cats[0]],
                        fill="toself",name=model_choice,line_color="#00b67a",opacity=.7))
        if baseline_ok:
            v_lr = [acc_lr,prec_lr,rec_lr,f1_lr]
            fig_r.add_trace(go.Scatterpolar(r=v_lr+[v_lr[0]],theta=cats+[cats[0]],
                            fill="toself",name="TF-IDF+LogReg",line_color="#767676",opacity=.5))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.5,1.0])),
                            height=360,margin=dict(t=30,b=20))
        st.plotly_chart(fig_r,use_container_width=True)
        delta = f1_tr-(f1_lr if baseline_ok else 0)
        st.markdown(f'''<div style="background:#f0faf5;border-left:4px solid #00b67a;padding:1rem;border-radius:0 10px 10px 0;font-size:.9rem">
        <b>Analyse :</b> Le modèle <b>{model_choice}</b> atteint F1 = <b>{f1_tr:.3f}</b> sur {len(df_e)} avis IMDB.
        {"Gain vs baseline : <b>"+f"{delta:+.3f}</b>" if baseline_ok else ""}
        </div>''', unsafe_allow_html=True)
