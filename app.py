"""
Tenzing
From a French startup name to a one-page investment memo, using public sources only.
Sources: API Recherche d'entreprises (SIRENE, data.gouv.fr), company website, Google News RSS.
LLM: Groq 
"""

import base64
import io
import re
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import date

import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

SIRENE_API = "https://recherche-entreprises.api.gouv.fr/search"
NEWS_RSS = "https://news.google.com/rss/search"
# OpenAI-compatible providers. Free plans limit each model separately,
# so each provider lists models to try in order.
PROVIDERS = {
    "Groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "secret": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    },
    "Mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "secret": "MISTRAL_API_KEY",
        "models": ["mistral-small-latest", "mistral-medium-latest"],
    },
}
CLAUDE_MODEL = "claude-sonnet-5-5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Tenzing - educational project)"}

# INSEE codes for headcount brackets (tranche d'effectif salarié)
EFFECTIFS = {
    "NN": "Non renseigné", "00": "0 salarié", "01": "1-2", "02": "3-5", "03": "6-9",
    "11": "10-19", "12": "20-49", "21": "50-99", "22": "100-199", "31": "200-249",
    "32": "250-499", "41": "500-999", "42": "1 000-1 999", "51": "2 000-4 999",
    "52": "5 000-9 999", "53": "10 000+",
}

st.set_page_config(page_title="Tenzing · Due Diligence", page_icon="🏔️", layout="wide",
                   initial_sidebar_state="collapsed")


# ----------------------------------------------------------------------------
# 1. Data collection
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def search_companies(name: str) -> list[dict]:
    r = requests.get(SIRENE_API, params={"q": name, "per_page": 10}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])


def parse_finances(raw) -> list[dict]:
    """The API returns finances as {year: {ca, resultat_net}}; handle a list format too."""
    rows = []
    if isinstance(raw, dict):
        for year, v in raw.items():
            rows.append({"année": year, "ca": (v or {}).get("ca"), "resultat_net": (v or {}).get("resultat_net")})
    elif isinstance(raw, list):
        for v in raw:
            rows.append({"année": v.get("year") or v.get("annee"), "ca": v.get("ca"), "resultat_net": v.get("resultat_net")})
    return sorted(rows, key=lambda x: str(x["année"]), reverse=True)


def company_profile(c: dict) -> dict:
    siege = c.get("siege") or {}
    dirigeants = []
    for d in c.get("dirigeants") or []:
        if d.get("denomination"):
            dirigeants.append(f"{d.get('denomination')} ({d.get('qualite', 'n.c.')})")
        else:
            nom = f"{d.get('prenoms', '')} {d.get('nom', '')}".strip()
            dirigeants.append(f"{nom} ({d.get('qualite', 'n.c.')})")
    return {
        "Dénomination": c.get("nom_complet"),
        "SIREN": c.get("siren"),
        "Date de création": c.get("date_creation"),
        "Statut": "Active" if c.get("etat_administratif") == "A" else "Cessée / inconnue",
        "Code NAF": c.get("activite_principale"),
        "Effectif (tranche INSEE)": EFFECTIFS.get(str(c.get("tranche_effectif_salarie")), "Non renseigné"),
        "Siège": siege.get("adresse"),
        "Établissements ouverts": c.get("nombre_etablissements_ouverts"),
        "Dirigeants": dirigeants[:8],
        "Finances publiées": parse_finances(c.get("finances")),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_website(url: str, max_chars: int = 3500) -> str:
    if not url:
        return ""
    if not url.startswith("http"):
        url = "https://" + url
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        return f"[Site inaccessible : {e}]"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    meta = soup.find("meta", attrs={"name": "description"})
    desc = meta.get("content", "") if meta else ""
    text = " ".join(soup.get_text(" ").split())
    return f"Titre : {title}\nDescription : {desc}\nContenu : {text}"[:max_chars]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news(name: str, limit: int = 10) -> list[dict]:
    params = {"q": f'"{name}"', "hl": "fr", "gl": "FR", "ceid": "FR:fr"}
    try:
        r = requests.get(NEWS_RSS, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError):
        return []
    items = []
    for item in root.iter("item"):
        items.append({
            "titre": item.findtext("title", ""),
            "date": item.findtext("pubDate", ""),
            "source": item.findtext("source", ""),
            "lien": item.findtext("link", ""),
        })
        if len(items) >= limit:
            break
    return items


# ----------------------------------------------------------------------------
# 2. Memo generation
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """Tu es analyste dans un fonds de venture capital. Tu rédiges une note
d'investissement préliminaire d'UNE page, à partir des seules sources fournies.

Règles :
- N'invente aucun chiffre, aucun nom, aucun fait. Si une information manque, écris-le.
- Distingue clairement les FAITS (sourcés [S1], [S2]...) des HYPOTHÈSES (préfixées "Hypothèse :").
- Style : sobre, précis, orienté décision. Pas de superlatifs marketing.

Structure attendue (Markdown) :
## Synthèse (3 lignes max)
## Fiche d'identité
## Modèle d'affaires (tel que compris)
## Signaux de traction et actualité
## Forces (3 à 5)
## Risques (3 à 5, du plus au moins critique)
## Questions à poser au management (8 à 10, priorisées)
## Angles morts de cette note (données manquantes)
"""


def build_context(profile: dict, website: str, news: list[dict]) -> str:
    lines = ["[S1] Registre SIRENE (API Recherche d'entreprises, data.gouv.fr)"]
    for k, v in profile.items():
        lines.append(f"- {k} : {v}")
    lines.append("\n[S2] Site web de l'entreprise")
    lines.append(website or "Non fourni.")
    lines.append("\n[S3] Revue de presse (Google News, titres uniquement)")
    if news:
        for n in news:
            lines.append(f"- {n['date']} | {n['source']} | {n['titre']}")
    else:
        lines.append("Aucun article trouvé.")
    return "\n".join(lines)


def generate_memo(provider: str, api_key: str, context: str, language: str) -> str:
    system = SYSTEM_PROMPT + f"\nLangue de rédaction : {language}."
    user = f"Sources disponibles :\n\n{context}\n\nRédige la note."
    if provider in PROVIDERS:
        conf = PROVIDERS[provider]
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last = None
        for model in conf["models"]:
            payload = {"model": model, "max_tokens": 1800, "temperature": 0.2, "messages": messages}
            for wait in (8, None):  # one automatic retry per model on 429
                r = requests.post(conf["url"], headers=headers, json=payload, timeout=120)
                last = r
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                if r.status_code != 429 or wait is None:
                    break
                time.sleep(wait)
            if r.status_code not in (429, 400, 404):  # auth or server error: no point trying another model
                break
        if last is not None and last.status_code == 429:
            raise RuntimeError(f"le quota gratuit de {provider} est atteint pour tous les modèles testés "
                               f"({', '.join(conf['models'])}). Réessaie plus tard ou change de moteur.")
        raise RuntimeError(f"{provider} {last.status_code} : {last.text[:300]}")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(model=CLAUDE_MODEL, max_tokens=2500, system=system,
                                 messages=[{"role": "user", "content": user}])
    return "".join(block.text for block in msg.content if block.type == "text")


# ----------------------------------------------------------------------------
# 3. Look & feel
# ----------------------------------------------------------------------------
ASSETS = Path(__file__).parent / "assets"


ROOT = Path(__file__).parent


def find_image(name: str) -> Path | None:
    """Look in assets/ first, then the repo root, ignoring upper/lower case."""
    for folder in (ASSETS, ROOT):
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name.lower() == name.lower():
                    return f
    return None


@st.cache_data(show_spinner=False)
def img_b64(name: str) -> str:
    path = find_image(name)
    return base64.b64encode(path.read_bytes()).decode() if path else ""


T = {
    "fr": {
        "eyebrow": "Due diligence · Startups françaises",
        "lead": "Du nom d'une startup à une note d'investissement d'une page. "
                "Sources publiques, méthode d'analyste, en moins d'une minute.",
        "cta": "Commencer l'analyse ↓",
        "method": "La méthode",
        "steps": [("01", "Identifier", "L'entité juridique exacte dans le registre officiel SIRENE, "
                   "validée par l'utilisateur pour écarter les homonymes."),
                  ("02", "Collecter", "Données légales, comptes publiés, dirigeants, site web et "
                   "actualité récente. Uniquement des sources publiques."),
                  ("03", "Synthétiser", "Une note d'une page : forces, risques et questions à poser "
                   "au management. Chaque fait est sourcé.")],
        "quote": "Avant chaque ascension, quelqu'un doit reconnaître la voie.",
        "quote_by": "En hommage à Tenzing Norgay, Everest, 1953",
        "tool": "L'analyse",
        "name": "Nom de l'entreprise", "url": "Site web (optionnel)",
        "search": "1 · Rechercher dans le registre",
        "pick": "Sélectionnez la bonne entité juridique",
        "registry": "Fiche registre · SIRENE",
        "execs": "Dirigeants", "none": "non publiés",
        "fin": "Comptes publiés",
        "nofin": "Aucun compte publié (fréquent pour les startups qui demandent la confidentialité).",
        "generate": "2 · Générer la note d'investissement",
        "settings": "Paramètres avancés", "engine": "Moteur d'analyse", "key": "Clé API",
        "needkey": "Renseignez une clé API dans les paramètres avancés pour générer la note.",
        "climb": ["Camp de base", "Camp I", "Camp II", "Sommet"],
        "climb_sub": ["Registre", "Site web", "Presse", "Note"],
        "memo": "Note d'investissement", "download": "Markdown (.md)",
        "download_pdf": "Télécharger le PDF", "pdf_label": "Note d'investissement préliminaire",
        "pdf_foot": "Tenzing · Sources publiques (SIRENE, site web, presse) · Ne constitue pas un conseil en investissement",
        "pdf_created": "Création", "pdf_staff": "Effectif", "pdf_hq": "Siège", "pdf_date": "Note du",
        "press": "Sources presse [S3]",
        "err_api": "Erreur API SIRENE", "err_engine": "Erreur du moteur",
        "noresult": "Aucune entreprise trouvée pour ce nom.",
        "disclaimer": "Outil pédagogique fondé sur des données publiques. "
                      "Ne constitue pas un conseil en investissement.",
        "memo_lang": "français",
        "sources": "Sources : SIRENE (data.gouv.fr), site web, Google News.",
    },
    "en": {
        "eyebrow": "Due diligence · French startups",
        "lead": "From a startup's name to a one-page investment memo. "
                "Public sources, an analyst's method, in under a minute.",
        "cta": "Start the analysis ↓",
        "method": "The method",
        "steps": [("01", "Identify", "The exact legal entity in the official SIRENE register, "
                   "validated by the user to rule out namesakes."),
                  ("02", "Collect", "Legal data, published accounts, executives, website and "
                   "recent news. Public sources only."),
                  ("03", "Synthesise", "A one-page memo: strengths, risks and questions for "
                   "management. Every fact is sourced.")],
        "quote": "Before any ascent, someone has to scout the route.",
        "quote_by": "A tribute to Tenzing Norgay, Everest, 1953",
        "tool": "The analysis",
        "name": "Company name", "url": "Website (optional)",
        "search": "1 · Search the register",
        "pick": "Select the right legal entity",
        "registry": "Registry record · SIRENE",
        "execs": "Executives", "none": "not published",
        "fin": "Published accounts",
        "nofin": "No published accounts (common for startups that file confidentially).",
        "generate": "2 · Generate the investment memo",
        "settings": "Advanced settings", "engine": "Analysis engine", "key": "API key",
        "needkey": "Enter an API key in the advanced settings to generate the memo.",
        "climb": ["Base camp", "Camp I", "Camp II", "Summit"],
        "climb_sub": ["Register", "Website", "Press", "Memo"],
        "memo": "Investment memo", "download": "Markdown (.md)",
        "download_pdf": "Download PDF", "pdf_label": "Preliminary investment memo",
        "pdf_foot": "Tenzing · Public sources (SIRENE, website, press) · Not investment advice",
        "pdf_created": "Incorporated", "pdf_staff": "Headcount", "pdf_hq": "Head office", "pdf_date": "Memo dated",
        "press": "Press sources [S3]",
        "err_api": "SIRENE API error", "err_engine": "Engine error",
        "noresult": "No company found for this name.",
        "disclaimer": "Educational tool based on public data. Not investment advice.",
        "memo_lang": "English",
        "sources": "Sources: SIRENE (data.gouv.fr), company website, Google News.",
    },
}

EN_LABELS = {"Dénomination": "Name", "Date de création": "Incorporated", "Statut": "Status",
             "Code NAF": "NAF code", "Effectif (tranche INSEE)": "Headcount (INSEE bracket)",
             "Siège": "Head office", "Établissements ouverts": "Open establishments"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');
:root { --ink:#0b0b0b; --paper:#f4f4f2; --mute:#8d8d8d; --line:rgba(255,255,255,.14); }
html, body, .stApp, [data-testid="stAppViewContainer"] { background:var(--ink); color:var(--paper); }
.stApp, .stMarkdown, p, li, label, input, button, textarea, td, th { font-family:'Inter',sans-serif !important; }
h1,h2,h3,h4, .serif { font-family:'Cormorant Garamond',serif !important; font-weight:500 !important; letter-spacing:.01em; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1rem; max-width:1320px; }
iframe { border:0 !important; display:block; }
a { color:var(--paper) !important; }

/* top bar */
.brand { font-family:'Cormorant Garamond',serif; font-size:1.35rem; letter-spacing:.42em; text-transform:uppercase; }

/* hero */
.stApp { overflow-x:hidden; }
.hero { position:relative; width:100vw; margin-left:calc(50% - 50vw); min-height:88vh; overflow:hidden;
        display:flex; align-items:flex-end; background-color:#111; }
.hero-bg { position:absolute; inset:-4%; background-size:cover; background-position:center 30%;
           animation:kenburns 26s ease-out both; }
@keyframes kenburns { from { transform:scale(1.14) translateY(2%); } to { transform:scale(1) translateY(0); } }
.hero-mist { position:absolute; inset:0; pointer-events:none; opacity:.55;
             background:radial-gradient(ellipse 60% 30% at 20% 70%,rgba(255,255,255,.18),transparent 70%),
                        radial-gradient(ellipse 50% 25% at 80% 60%,rgba(255,255,255,.14),transparent 70%);
             animation:drift 18s ease-in-out infinite alternate; }
@keyframes drift { from { transform:translateX(-4%); } to { transform:translateX(4%); } }
.hero::after { content:""; position:absolute; inset:0;
               background:linear-gradient(180deg,rgba(11,11,11,.05) 0%,rgba(11,11,11,.10) 40%,rgba(11,11,11,.75) 78%,#0b0b0b 100%); }
.hero-inner { position:relative; z-index:2; padding:0 6vw 9vh; max-width:880px; animation:rise 1.4s ease-out both; }
@keyframes rise { from { opacity:0; transform:translateY(24px);} to { opacity:1; transform:none;} }
.eyebrow { font-size:.72rem; letter-spacing:.32em; text-transform:uppercase; color:#cfcfcf; margin-bottom:1rem; }
.hero h1 { font-size:clamp(3.4rem,9vw,7.5rem); line-height:.95; margin:0 0 1.2rem; color:#fff; }
.lead { font-size:1.12rem; font-weight:300; line-height:1.65; color:#e6e6e6; max-width:620px; }
.cta { display:inline-block; margin-top:1.6rem; padding:.8rem 1.6rem; border:1px solid #fff; color:#fff !important;
       text-decoration:none; font-size:.75rem; letter-spacing:.22em; text-transform:uppercase; transition:all .3s; }
.cta:hover { background:#fff; color:#000 !important; }

/* sections */
.section-label { font-size:.7rem; letter-spacing:.32em; text-transform:uppercase; color:var(--mute);
                 border-top:1px solid var(--line); padding-top:1.1rem; margin:4.5rem 0 2rem; }
.step { border-top:1px solid var(--line); padding:1.4rem 0; transition:border-color .4s; }
.step:hover { border-color:#fff; }
.step .num { font-family:'Cormorant Garamond',serif; font-size:2.6rem; color:#fff; line-height:1; }
.step h3 { font-size:1.6rem; margin:.6rem 0 .5rem; }
.step p { color:#bdbdbd; font-weight:300; line-height:1.65; font-size:.95rem; }

.band { position:relative; height:58vh; width:100vw; margin:4.5rem 0 0 calc(50% - 50vw); background-size:cover;
        background-position:center 40%; background-attachment:fixed; display:flex; align-items:center; justify-content:center; text-align:center; }
.band::after { content:""; position:absolute; inset:0; background:linear-gradient(180deg,#0b0b0b 0%,rgba(11,11,11,.35) 25%,rgba(11,11,11,.45) 75%,#0b0b0b 100%); }
.band div { position:relative; z-index:2; padding:0 6vw; }
.band blockquote { font-family:'Cormorant Garamond',serif; font-size:clamp(1.6rem,3.4vw,2.7rem); font-style:italic;
                   color:#fff; margin:0 0 .8rem; border:none; }
.band cite { font-size:.7rem; letter-spacing:.28em; text-transform:uppercase; color:#cfcfcf; font-style:normal; }

/* widgets */
[data-testid="stTextInput"] input { background:transparent !important; border:none !important; border-bottom:1px solid #6b6b6b !important;
       border-radius:0 !important; color:#fff !important; font-size:1.05rem; padding-left:0; }
[data-testid="stTextInput"] input:focus { border-bottom-color:#fff !important; box-shadow:none !important; }
[data-testid="stTextInput"] div[data-baseweb="input"] { background:transparent !important; border:none !important; }
.stButton button, .stDownloadButton button { background:transparent; color:#fff; border:1px solid #fff; border-radius:0;
       font-size:.74rem; letter-spacing:.2em; text-transform:uppercase; padding:.75rem 1.5rem; transition:all .3s; }
.stButton button:hover, .stDownloadButton button:hover { background:#fff; color:#000; border-color:#fff; }
.stButton button:disabled { opacity:.35; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:0 !important; }
[data-testid="stTable"] table { color:var(--paper); }
[data-testid="stTable"] th { color:var(--mute) !important; font-weight:400; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }

/* climb progress */
.climb { display:flex; justify-content:space-between; position:relative; margin:2.2rem 0 1.4rem; }
.climb::before { content:""; position:absolute; top:9px; left:4%; right:4%; height:1px; background:var(--line); }
.camp { position:relative; z-index:1; flex:1; text-align:center; }
.camp .dot { width:19px; height:19px; margin:0 auto .7rem; border-radius:50%; border:1px solid #6b6b6b; background:var(--ink); }
.camp.done .dot { background:#fff; border-color:#fff; }
.camp.active .dot { border-color:#fff; animation:pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(255,255,255,.55);} 50% { box-shadow:0 0 0 10px rgba(255,255,255,0);} }
.camp .t { font-family:'Cormorant Garamond',serif; font-size:1.15rem; color:#6b6b6b; }
.camp .s { font-size:.65rem; letter-spacing:.2em; text-transform:uppercase; color:#555; }
.camp.done .t, .camp.active .t { color:#fff; } .camp.done .s, .camp.active .s { color:#bdbdbd; }

/* memo */
.memo-head { font-size:.7rem; letter-spacing:.32em; text-transform:uppercase; color:var(--mute); margin-top:1rem; }
.foot { border-top:1px solid var(--line); margin-top:5rem; padding:2rem 0 3rem; color:var(--mute); font-size:.78rem;
        display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem; }
@media (max-width:700px) { .hero { min-height:64vh; } .band { background-attachment:scroll; } }
</style>
"""


def climb_html(t: dict, current: int) -> str:
    camps = []
    for i, (title, sub) in enumerate(zip(t["climb"], t["climb_sub"])):
        state = "done" if i < current else "active" if i == current else ""
        camps.append(f'<div class="camp {state}"><div class="dot"></div>'
                     f'<div class="t">{title}</div><div class="s">{sub}</div></div>')
    return f'<div class="climb">{"".join(camps)}</div>'


# ----------------------------------------------------------------------------
# PDF export (1-2 pages, black & white, A4)
# ----------------------------------------------------------------------------
PDF_CHARS = {"→": "->", "←": "<-", "≈": "~", "≥": ">=", "≤": "<=", "✓": "-", "✔": "-", "✗": "x",
             "▪": "•", "●": "•", "◦": "-", "‑": "-", " ": " ", " ": " ", " ": " "}


def pdf_safe(text: str) -> str:
    text = "".join(PDF_CHARS.get(c, c) for c in text)
    return "".join(c if c.encode("cp1252", "ignore") else "" for c in text)


def md_inline(text: str) -> str:
    text = pdf_safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text.replace("`", "")


def build_pdf(memo: str, profile: dict, t: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    W, H = A4
    hero = find_image("hero.jpg")
    band_h = 40 * mm
    company = pdf_safe(str(profile.get("Dénomination") or ""))
    meta = " · ".join(x for x in [f"SIREN {profile.get('SIREN')}" if profile.get("SIREN") else "",
                                  f"NAF {profile.get('Code NAF')}" if profile.get("Code NAF") else "",
                                  f"{t['pdf_date']} {date.today().strftime('%d/%m/%Y')}"] if x)

    def first_page(c, doc):
        if hero:
            img = ImageReader(str(hero))
            iw, ih = img.getSize()
            draw_h = W * ih / iw
            c.saveState()
            p = c.beginPath(); p.rect(0, H - band_h, W, band_h); c.clipPath(p, stroke=0, fill=0)
            c.drawImage(img, 0, H - band_h - draw_h * 0.10, W, draw_h)  # show the dark lower ridges
            c.setFillColor(colors.black); c.setFillAlpha(.28); c.rect(0, H - band_h, W, band_h, stroke=0, fill=1)
            c.restoreState()
        else:
            c.setFillColor(colors.black); c.rect(0, H - band_h, W, band_h, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Times-Roman", 22); c.drawString(16 * mm, H - band_h + 9 * mm, "T E N Z I N G")
        c.setFont("Helvetica", 6.5); c.drawString(16 * mm, H - band_h + 5 * mm, pdf_safe(t["eyebrow"]).upper())
        later_pages(c, doc)

    def later_pages(c, doc):
        c.setStrokeColor(colors.HexColor("#bbbbbb")); c.setLineWidth(.4)
        c.line(16 * mm, 12 * mm, W - 16 * mm, 12 * mm)
        c.setFillColor(colors.HexColor("#666666")); c.setFont("Helvetica", 6.5)
        c.drawString(16 * mm, 8 * mm, pdf_safe(t["pdf_foot"]))
        c.drawRightString(W - 16 * mm, 8 * mm, f"{doc.page}")

    def story(size: float):
        lead = size * 1.32
        body = ParagraphStyle("b", fontName="Helvetica", fontSize=size, leading=lead, alignment=TA_LEFT,
                              textColor=colors.HexColor("#111111"), spaceAfter=size * .35)
        bullet = ParagraphStyle("li", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=size * .2)
        h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=size * 1.32, leading=size * 1.6,
                            spaceBefore=size * .9, spaceAfter=size * .3, textColor=colors.black)
        label = ParagraphStyle("lab", fontName="Helvetica", fontSize=6.8, leading=9, textColor=colors.HexColor("#777777"))
        title = ParagraphStyle("ti", fontName="Times-Roman", fontSize=21, leading=24, textColor=colors.black)
        small = ParagraphStyle("sm", fontName="Helvetica", fontSize=7.4, leading=9.5, textColor=colors.HexColor("#444444"))

        out = [Paragraph(md_inline(t["pdf_label"]).upper(), label), Spacer(1, 2),
               Paragraph(md_inline(company or "-"), title), Paragraph(md_inline(meta), small), Spacer(1, 6)]
        facts = [[t["pdf_created"], "Statut" if t["memo_lang"] == "français" else "Status", t["pdf_staff"], t["pdf_hq"]],
                 [str(profile.get("Date de création") or "-"), str(profile.get("Statut") or "-"),
                  str(profile.get("Effectif (tranche INSEE)") or "-"), str(profile.get("Siège") or "-")]]
        facts = [[Paragraph(md_inline(x).upper() if r == 0 else md_inline(x), label if r == 0 else small)
                  for x in row] for r, row in enumerate(facts)]
        ft = Table(facts, colWidths=[28 * mm, 26 * mm, 30 * mm, None])
        ft.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), .8, colors.black),
                                ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#999999")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        out += [ft, Spacer(1, 4)]

        lines, i = memo.splitlines(), 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("|"):  # markdown table
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                        rows.append([Paragraph(md_inline(c), small) for c in cells])
                    i += 1
                if rows:
                    n = max(len(r) for r in rows)
                    rows = [r + [""] * (n - len(r)) for r in rows]
                    tb = Table(rows, colWidths=[(W - 32 * mm) / n] * n)
                    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#cccccc")),
                                            ("LINEABOVE", (0, 0), (-1, 0), .8, colors.black),
                                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
                    out += [tb, Spacer(1, 4)]
                continue
            i += 1
            if not line or re.fullmatch(r"[-*_]{3,}", line):
                continue
            if line.startswith("#"):
                out.append(Paragraph(md_inline(line.lstrip("#").strip()), h2))
            elif re.match(r"^[-*•]\s+", line):
                out.append(Paragraph(md_inline(re.sub(r"^[-*•]\s+", "", line)), bullet, bulletText="•"))
            elif re.match(r"^\d+[.)]\s+", line):
                num = re.match(r"^(\d+)", line).group(1)
                out.append(Paragraph(md_inline(re.sub(r"^\d+[.)]\s+", "", line)), bullet, bulletText=f"{num}."))
            else:
                out.append(Paragraph(md_inline(line), body))
        return out

    from reportlab.platypus import BaseDocTemplate, Frame, NextPageTemplate, PageTemplate

    pdf = b""
    for size in (9.0, 8.3, 7.7, 7.2):  # shrink until the memo fits on 2 pages
        buf = io.BytesIO()
        pages = []
        doc = BaseDocTemplate(buf, pagesize=A4, title=f"Tenzing - {company}", author="Tenzing")
        m, bottom = 16 * mm, 17 * mm
        first = Frame(m, bottom, W - 2 * m, H - band_h - 7 * mm - bottom, id="first")
        rest = Frame(m, bottom, W - 2 * m, H - 16 * mm - bottom, id="rest")
        doc.addPageTemplates([
            PageTemplate("first", [first], onPage=lambda c, d: (pages.append(d.page), first_page(c, d))),
            PageTemplate("rest", [rest], onPage=lambda c, d: (pages.append(d.page), later_pages(c, d))),
        ])
        doc.build([NextPageTemplate("rest")] + story(size))
        pdf = buf.getvalue()
        if len(pages) <= 2:
            break
    return pdf


# ----------------------------------------------------------------------------
# Cinematic video blocks (videos served from static/, photo shown as poster/fallback)
# ----------------------------------------------------------------------------
# Public GitHub repo: jsDelivr serves its files with the right video type (Streamlit serves .mp4 as text).
GITHUB_REPO = "nicolasdejuvigny1-del/Tenzing-Due-Diligence"
VIDEO_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;1,400&family=Inter:wght@300;400&display=swap" rel="stylesheet">
<style>
html,body{margin:0;background:#0b0b0b;overflow:hidden;font-family:'Inter',sans-serif;}
.wrap{position:relative;width:100%;height:100vh;overflow:hidden;background:#111 center/cover no-repeat;}
video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.shade{position:absolute;inset:0;}
.inner{position:absolute;z-index:2;color:#fff;animation:rise 1.6s ease-out both;}
@keyframes rise{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}
.eyebrow{font-size:11px;letter-spacing:.32em;text-transform:uppercase;color:#d6d6d6;margin-bottom:14px;}
h1{font-family:'Cormorant Garamond',serif;font-weight:500;font-size:clamp(56px,9vw,124px);line-height:.95;margin:0 0 18px;}
.lead{font-weight:300;font-size:17px;line-height:1.65;color:#e8e8e8;max-width:600px;}
.cta{display:inline-block;margin-top:26px;padding:13px 26px;border:1px solid #fff;color:#fff;background:transparent;
     font:400 11px 'Inter',sans-serif;letter-spacing:.22em;text-transform:uppercase;cursor:pointer;transition:all .3s;}
.cta:hover{background:#fff;color:#000;}
blockquote{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:clamp(26px,3.4vw,44px);margin:0 0 14px;}
cite{font-style:normal;font-size:11px;letter-spacing:.28em;text-transform:uppercase;color:#d6d6d6;}
</style>"""


def video_block(video: str, poster: str, inner_html: str, shade: str, inner_css: str, height: int) -> None:
    has_video = (ROOT / "static" / video).exists()
    poster_b64 = img_b64(poster)
    poster_attr = f'poster="data:image/jpeg;base64,{poster_b64}"' if poster_b64 else ""
    bg = f"background-image:url('data:image/jpeg;base64,{poster_b64}')" if poster_b64 else ""
    video_tag = (f'<video autoplay muted loop playsinline preload="auto" {poster_attr}>'
                 f'<source src="https://cdn.jsdelivr.net/gh/{GITHUB_REPO}@main/static/{video}" type="video/mp4">'
                 f'<source src="app/static/{video}" type="video/mp4"></video>') if has_video else ""
    components.html(f"""{VIDEO_CSS}
<div class="wrap" style="{bg}">{video_tag}
  <div class="shade" style="background:{shade}"></div>
  <div class="inner" style="{inner_css}">{inner_html}</div>
</div>
<script>
  const v = document.querySelector('video');
  if (v) {{ v.muted = true; v.play().catch(() => {{}}); }}
  function goTool() {{
    const el = window.parent.document.getElementById('analysis');
    if (el) el.scrollIntoView({{behavior: 'smooth'}});
  }}
</script>""", height=height)


# ----------------------------------------------------------------------------
# 4. Page
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

bar_l, bar_r = st.columns([5, 1])
bar_l.markdown('<div class="brand">Tenzing</div>', unsafe_allow_html=True)
lang = bar_r.radio("Language", ["FR", "EN"], horizontal=True, label_visibility="collapsed").lower()
t = T[lang]
# Hero (video)
video_block("hero.mp4", "hero_poster.jpg",
            f"""<div class="eyebrow">{t['eyebrow']}</div><h1>Tenzing</h1>
            <div class="lead">{t['lead']}</div>
            <button class="cta" onclick="goTool()">{t['cta']}</button>""",
            "linear-gradient(180deg,rgba(11,11,11,0) 0%,rgba(11,11,11,.08) 40%,rgba(11,11,11,.78) 80%,#0b0b0b 100%)",
            "left:6%;right:6%;bottom:9%;max-width:860px;", 700)

# Method
st.markdown(f'<div class="section-label">{t["method"]}</div>', unsafe_allow_html=True)
for col, (num, title, text) in zip(st.columns(3, gap="large"), t["steps"]):
    col.markdown(f'<div class="step"><div class="num">{num}</div><h3>{title}</h3><p>{text}</p></div>',
                 unsafe_allow_html=True)

# Quote band (video)
st.markdown("<div style='height:4rem'></div>", unsafe_allow_html=True)
video_block("ridge.mp4", "ridge.jpg",
            f"""<blockquote>« {t['quote']} »</blockquote><cite>{t['quote_by']}</cite>""",
            "linear-gradient(180deg,#0b0b0b 0%,rgba(11,11,11,.30) 22%,rgba(11,11,11,.42) 78%,#0b0b0b 100%)",
            "left:8%;right:8%;top:50%;transform:translateY(-50%);text-align:center;animation:none;", 480)

# Tool
st.markdown(f'<div id="analysis" class="section-label">{t["tool"]}</div>', unsafe_allow_html=True)

with st.expander(t["settings"]):
    provider = st.radio(t["engine"], ["Groq", "Mistral", "Claude"], horizontal=True)
    secret_name = PROVIDERS[provider]["secret"] if provider in PROVIDERS else "ANTHROPIC_API_KEY"
    try:
        secret_key = st.secrets.get(secret_name, "")
    except Exception:  # no secrets file locally
        secret_key = ""
    api_key = secret_key or st.text_input(f"{t['key']} {provider}", type="password")

col1, col2 = st.columns(2, gap="large")
name = col1.text_input(t["name"], placeholder="Pennylane")
url = col2.text_input(t["url"], placeholder="pennylane.com")

if st.button(t["search"], disabled=not name):
    try:
        st.session_state["results"] = search_companies(name)
        st.session_state.pop("memo", None)
        if not st.session_state["results"]:
            st.warning(t["noresult"])
    except requests.RequestException as e:
        st.error(f"{t['err_api']} : {e}")
        st.session_state["results"] = []

results = st.session_state.get("results", [])
if results:
    labels = [f"{c.get('nom_complet')} — SIREN {c.get('siren')} — "
              f"{(c.get('siege') or {}).get('libelle_commune', '')} — NAF {c.get('activite_principale')}"
              for c in results]
    idx = st.selectbox(t["pick"], range(len(labels)), format_func=lambda i: labels[i])
    profile = company_profile(results[idx])

    with st.expander(t["registry"], expanded=True):
        shown = {k: v for k, v in profile.items() if k not in ("Dirigeants", "Finances publiées")}
        if lang == "en":
            shown = {EN_LABELS.get(k, k): v for k, v in shown.items()}
        st.table({k: [str(v)] for k, v in shown.items()})
        st.markdown(f"**{t['execs']} :** " + (", ".join(profile["Dirigeants"]) or t["none"]))
        if profile["Finances publiées"]:
            st.markdown(f"**{t['fin']} :**")
            st.dataframe(profile["Finances publiées"], hide_index=True)
        else:
            st.caption(t["nofin"])

    progress = st.empty()
    if st.button(t["generate"], disabled=not api_key):
        progress.markdown(climb_html(t, 1), unsafe_allow_html=True)
        website = fetch_website(url)
        progress.markdown(climb_html(t, 2), unsafe_allow_html=True)
        news = fetch_news(profile["Dénomination"] or name)
        progress.markdown(climb_html(t, 3), unsafe_allow_html=True)
        try:
            memo = generate_memo(provider, api_key, build_context(profile, website, news), t["memo_lang"])
        except Exception as e:
            progress.empty()
            st.error(f"{t['err_engine']} {provider} : {e}")
            st.stop()
        st.session_state.update(memo=memo, news=news, memo_profile=profile, memo_lang=lang,
                                memo_pdf=None)
    if st.session_state.get("memo"):
        progress.markdown(climb_html(t, 4), unsafe_allow_html=True)
    if not api_key:
        st.caption(t["needkey"])

if st.session_state.get("memo"):
    st.markdown(f'<div class="memo-head">{t["memo"]}</div>', unsafe_allow_html=True)
    st.markdown(st.session_state["memo"])
    if not st.session_state.get("memo_pdf"):
        try:
            st.session_state["memo_pdf"] = build_pdf(st.session_state["memo"],
                                                     st.session_state.get("memo_profile", {}),
                                                     T[st.session_state.get("memo_lang", lang)])
        except Exception as e:  # never block the memo because of the PDF
            st.caption(f"PDF indisponible : {e}")
    stem = re.sub(r"[^A-Za-z0-9]+", "_", str(st.session_state.get("memo_profile", {}).get("Dénomination") or "memo"))
    b1, b2, _ = st.columns([1.2, 1, 2])
    if st.session_state.get("memo_pdf"):
        b1.download_button(t["download_pdf"], st.session_state["memo_pdf"], mime="application/pdf",
                           file_name=f"Tenzing_{stem}_{date.today().isoformat()}.pdf")
    b2.download_button(t["download"], st.session_state["memo"],
                       file_name=f"Tenzing_{stem}_{date.today().isoformat()}.md")
    with st.expander(t["press"]):
        for n in st.session_state.get("news", []):
            st.markdown(f"- [{n['titre']}]({n['lien']}) — {n['source']}, {n['date']}")

st.markdown(f"""
<div class="foot">
  <span>{t['disclaimer']} {t['sources']}</span>
  <span>Nicolas Leonard de Juvigny · <a href="https://linkedin.com/in/nicolasjuvigny">LinkedIn</a></span>
</div>""", unsafe_allow_html=True)
