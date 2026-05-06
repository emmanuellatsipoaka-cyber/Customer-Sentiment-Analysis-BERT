"""
=============================================================
  CORRECTIONS AvisCheckeur
  Remplacez les deux fonctions suivantes dans votre app.py
=============================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION 1 — call_gemini_summary
# Changement : gemini-1.5-flash → gemini-2.0-flash (modèle actif)
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini_summary(stats_text: str, api_key: str) -> str:
    import urllib.request, urllib.error, json

    if not api_key or not api_key.strip():
        raise ValueError("Clé API Google Gemini manquante.")

    prompt = (
        "Tu es un expert senior en analyse de sentiment et en experience client (CX).\n"
        "Voici les statistiques d'une analyse de sentiments sur des avis clients :\n\n"
        + stats_text +
        "\n\nFournis un rapport analytique structure en francais avec exactement ces sections :\n\n"
        "**1. Vue d'ensemble**\n"
        "Interpretation globale du sentiment dominant et du niveau de satisfaction.\n\n"
        "**2. Points forts**\n"
        "Ce qui ressort positivement des avis (3 points max, concis).\n\n"
        "**3. Points d'attention**\n"
        "Les axes d'amelioration identifies (3 points max, concis).\n\n"
        "**4. Recommandations operationnelles**\n"
        "3 actions concretes, realistes et priorisees.\n\n"
        "**5. Score de sante client**\n"
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

    # ✅ CORRECTION : gemini-2.0-flash (modèle actif et gratuit)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
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
                "Reponse vide de Gemini (aucun candidat). "
                "Verifiez votre cle API ou les quotas."
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            finish = candidates[0].get("finishReason", "UNKNOWN")
            raise RuntimeError(
                f"Reponse vide de Gemini (finishReason={finish}). "
                "La generation a peut-etre ete bloquee par les filtres de securite."
            )

        text_result = parts[0].get("text", "").strip()
        if not text_result:
            raise RuntimeError("Le modele a retourne un texte vide.")

        return text_result

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            err_json = json.loads(body)
            msg = err_json.get("error", {}).get("message", body[:300])
        except Exception:
            msg = body[:300]
        raise RuntimeError(f"Erreur HTTP {e.code} : {msg}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Erreur reseau : {e.reason}. "
            "Verifiez votre connexion internet."
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Erreur inattendue API Gemini : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTION 2 — generate_pdf_report
# Problème : Helvetica dans fpdf2 ne supporte PAS les caractères
#            hors Latin-1 : —  ·  é  è  à  ç  «  »  etc.
# Solution : fonction sanitize() qui remplace/encode tous ces caractères
#            avant tout appel à cell() / multi_cell()
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(df_r, summary_text: str,
                        template_name: str, template: dict,
                        model_name: str, n_reviews: int) -> bytes:
    import io, os, re, tempfile
    from fpdf import FPDF
    from datetime import datetime
    import pandas as pd

    # ── Helper : nettoie le texte pour Latin-1 / Helvetica ──────────────────
    def sanitize(text: str) -> str:
        """
        Remplace les caractères Unicode courants incompatibles avec
        Helvetica (Latin-1) par leurs équivalents ASCII.
        """
        _map = {
            "\u2014": "-",   # em dash  —
            "\u2013": "-",   # en dash  –
            "\u2019": "'",   # '
            "\u2018": "'",   # '
            "\u201c": '"',   # "
            "\u201d": '"',   # "
            "\u00b7": ".",   # point median  ·
            "\u2022": "-",   # puce  •
            "\u00e9": "e",   # é
            "\u00e8": "e",   # è
            "\u00ea": "e",   # ê
            "\u00eb": "e",   # ë
            "\u00e0": "a",   # à
            "\u00e2": "a",   # â
            "\u00f4": "o",   # ô
            "\u00fb": "u",   # û
            "\u00fc": "u",   # ü
            "\u00ee": "i",   # î
            "\u00ef": "i",   # ï
            "\u00e7": "c",   # ç
            "\u00f9": "u",   # ù
            "\u00e6": "ae",  # æ
            "\u0153": "oe",  # œ
            "\u00ab": '"',   # «
            "\u00bb": '"',   # »
            "\u00a0": " ",   # espace insécable
            "\u2026": "...", # …
            "\u00b0": "deg", # °
        }
        for char, repl in _map.items():
            text = text.replace(char, repl)
        # Encodage de sécurité : remplace tout ce qui reste hors Latin-1
        return text.encode("latin-1", errors="replace").decode("latin-1")

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
            self.cell(0, 12,
                      sanitize("AvisCheckeur - Rapport Analyse de Sentiments"),
                      align="L")
            self.set_font("Helvetica", "", 8)
            self.set_xy(10, 14)
            self.cell(0, 6,
                      sanitize(
                          f"Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')}  |  "
                          f"Modele : {model_name}  |  Template : {template_name}"
                      ),
                      align="L")
            self.ln(10)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(180, 180, 180)
            self.cell(0, 8,
                      sanitize(f"AvisCheckeur  -  ISE3 ENEAM  -  Page {self.page_no()}"),
                      align="C")

        def section_title(self, title):
            self.ln(4)
            self.set_fill_color(*primary_rgb)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 9, sanitize(f"  {title}"), fill=True, ln=True)
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
                self.cell(w - 2, 10, sanitize(str(val)), align="C")

                self.set_font("Helvetica", "", 8)
                self.set_text_color(100, 100, 100)
                self.set_xy(x, y + 13)
                self.cell(w - 2, 6, sanitize(str(label)), align="C")

            self.set_xy(start_x, start_y + 26)
            self.set_text_color(0, 0, 0)
            self.set_draw_color(0, 0, 0)
            self.ln(4)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── KPIs principaux ──────────────────────────────────────────────────────
    pdf.section_title("Statistiques Cles")
    pdf.kpi_row([
        ("Avis analyses",    n_reviews,          "#333333"),
        ("Positifs",         n_pos,               template["pos"]),
        ("Negatifs",         n_neg,               template["neg"]),
        ("Neutres",          n_neu,               template["neu"]),
        ("Confiance moy.",   f"{avg_conf:.0%}",   template["primary"]),
        ("TrustScore",       f"{trust}/5",         template["primary"]),
    ])

    # ── Tableau détail ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*primary_rgb)
    pdf.set_text_color(255, 255, 255)
    cols   = ["Categorie", "Nombre", "Pourcentage", "Confiance moy."]
    widths = [65, 35, 50, 40]
    for col, w in zip(cols, widths):
        pdf.cell(w, 8, sanitize(col), border=1, fill=True, align="C")
    pdf.ln()

    row_data = [
        ("POSITIF", n_pos, n_pos / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "POSITIVE"]["confidence"].mean() if n_pos > 0 else 0),
        ("NEGATIF", n_neg, n_neg / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "NEGATIVE"]["confidence"].mean() if n_neg > 0 else 0),
        ("NEUTRE",  n_neu, n_neu / max(len(df_r), 1),
         df_r[df_r["sentiment_pred"] == "NEUTRAL"]["confidence"].mean()  if n_neu > 0 else 0),
    ]
    pdf.set_font("Helvetica", "", 9)
    for i, (cat, cnt, pct, conf) in enumerate(row_data):
        bg = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(65, 7, sanitize(f"  {cat}"), border=1, fill=True)
        pdf.cell(35, 7, str(cnt),       border=1, fill=True, align="C")
        pdf.cell(50, 7, f"{pct:.1%}",   border=1, fill=True, align="C")
        pdf.cell(40, 7, f"{conf:.1%}" if conf > 0 else "-",
                 border=1, fill=True, align="C")
        pdf.ln()
    pdf.ln(6)

    # ── Graphiques matplotlib ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        pie_colors = [template["pos"], template["neg"], template["neu"]]
        labels_pie = ["Positif", "Negatif", "Neutre"]
        values_pie = [n_pos, n_neg, n_neu]
        non_zero   = [(l, v, c) for l, v, c in
                      zip(labels_pie, values_pie, pie_colors) if v > 0]
        if non_zero:
            axes[0].pie(
                [v for _, v, _ in non_zero],
                labels=[l for l, _, _ in non_zero],
                colors=[c for _, _, c in non_zero],
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.78,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
        axes[0].set_title("Repartition des sentiments", fontweight="bold", fontsize=10)

        for sent, clr in [("POSITIVE", template["pos"]),
                          ("NEGATIVE", template["neg"]),
                          ("NEUTRAL",  template["neu"])]:
            subset = df_r[df_r["sentiment_pred"] == sent]["confidence"]
            if len(subset) > 0:
                axes[1].hist(subset.values, bins=12, alpha=0.75,
                             color=clr, label=sent, edgecolor="white")
        axes[1].set_title("Distribution des scores de confiance",
                          fontweight="bold", fontsize=10)
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
        pdf.cell(0, 8, sanitize(f"[Graphiques non disponibles : {e}]"), ln=True)
        pdf.set_text_color(0, 0, 0)

    # ── Résumé IA ─────────────────────────────────────────────────────────────
    pdf.section_title("Analyse IA - Resume Intelligent (Google Gemini)")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)
    if summary_text:
        clean = summary_text.replace("**", "").replace("##", "").replace("#", "")
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            if re.match(r"^\d+\.", line):
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_font("Helvetica", "", 9)
            # ✅ sanitize() appliqué ici pour éviter l'erreur de caractère
            pdf.multi_cell(0, 5.5, sanitize(line))
    else:
        pdf.set_text_color(160, 160, 160)
        pdf.cell(0, 8,
                 sanitize("Aucun resume IA disponible. "
                           "Cliquez sur 'Generer le resume IA' avant l'export."),
                 ln=True)
    pdf.set_text_color(0, 0, 0)

    # ── Échantillon d'avis ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("Echantillon d'Avis Analyses (8 premiers)")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(*primary_rgb)
    pdf.set_text_color(255, 255, 255)
    for col, w in [("Extrait de l'avis", 115), ("Sentiment", 35),
                   ("Confiance", 25), ("Polarite", 15)]:
        pdf.cell(w, 7, sanitize(col), border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(0, 0, 0)
    for i, row in df_r.head(8).iterrows():
        bg = (248, 248, 248) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        excerpt = str(row["review"])[:90] + (
            "..." if len(str(row["review"])) > 90 else "")
        pdf.cell(115, 6, sanitize(f"  {excerpt}"), border=1, fill=True)
        pdf.cell(35,  6, sanitize(str(row["sentiment_pred"])), border=1, fill=True, align="C")
        pdf.cell(25,  6, f"{row['confidence']:.0%}",  border=1, fill=True, align="C")
        pdf.cell(15,  6, f"{row.get('polarity', 0):+.2f}", border=1, fill=True, align="C")
        pdf.ln()

    # ── Pied de rapport ───────────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 6,
             sanitize(f"Rapport genere par AvisCheckeur - ISE3 ENEAM - Emmanuella TSIPOAKA - 2025-2026"),
             align="C", ln=True)

    # ── Output bytes ──────────────────────────────────────────────────────────
    output = pdf.output()
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    else:
        return output.encode("latin-1")
