"""
Tenzing
From a French startup name to a one-page investment memo, using public sources only.
Sources: API Recherche d'entreprises (SIRENE, data.gouv.fr), company website, Google News RSS.
LLM: Groq (default, free plan), Mistral AI or Anthropic Claude.
"""

import base64
import time
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import date

import requests
import streamlit as st
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


@st.cache_data(show_spinner=False)
def img_b64(name: str) -> str:
    path = ASSETS / name
    return base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


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
        "memo": "Note d'investissement", "download": "Télécharger la note (.md)",
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
        "memo": "Investment memo", "download": "Download the memo (.md)",
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
.block-container { padding-top:1rem; max-width:1180px; }
a { color:var(--paper) !important; }

/* top bar */
.brand { font-family:'Cormorant Garamond',serif; font-size:1.35rem; letter-spacing:.42em; text-transform:uppercase; }

/* hero */
.hero { position:relative; min-height:78vh; border-radius:2px; overflow:hidden; display:flex; align-items:flex-end;
        background-size:cover; background-position:center 35%; animation:zoom 22s ease-out both; }
@keyframes zoom { from { background-size:112%; } to { background-size:100%; } }
.hero::after { content:""; position:absolute; inset:0;
               background:linear-gradient(180deg,rgba(11,11,11,.15) 0%,rgba(11,11,11,.35) 45%,rgba(11,11,11,.96) 100%); }
.hero-inner { position:relative; z-index:2; padding:0 4.5vw 7vh; max-width:880px; animation:rise 1.4s ease-out both; }
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

.band { position:relative; height:46vh; margin:4.5rem 0 0; background-size:cover; background-position:center;
        background-attachment:fixed; display:flex; align-items:center; justify-content:center; text-align:center; }
.band::after { content:""; position:absolute; inset:0; background:rgba(11,11,11,.55); }
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
# 4. Page
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

bar_l, bar_r = st.columns([5, 1])
bar_l.markdown('<div class="brand">Tenzing</div>', unsafe_allow_html=True)
lang = bar_r.radio("Language", ["FR", "EN"], horizontal=True, label_visibility="collapsed").lower()
t = T[lang]

# Hero
st.markdown(f"""
<div class="hero" style="background-image:url(data:image/jpeg;base64,{img_b64('hero.jpg')})">
  <div class="hero-inner">
    <div class="eyebrow">{t['eyebrow']}</div>
    <h1>Tenzing</h1>
    <div class="lead">{t['lead']}</div>
    <a class="cta" href="#analysis">{t['cta']}</a>
  </div>
</div>""", unsafe_allow_html=True)

# Method
st.markdown(f'<div class="section-label">{t["method"]}</div>', unsafe_allow_html=True)
for col, (num, title, text) in zip(st.columns(3, gap="large"), t["steps"]):
    col.markdown(f'<div class="step"><div class="num">{num}</div><h3>{title}</h3><p>{text}</p></div>',
                 unsafe_allow_html=True)

# Quote band
st.markdown(f"""
<div class="band" style="background-image:url(data:image/jpeg;base64,{img_b64('ridge.jpg')})">
  <div><blockquote>« {t['quote']} »</blockquote><cite>{t['quote_by']}</cite></div>
</div>""", unsafe_allow_html=True)

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
        st.session_state.update(memo=memo, news=news)
    if st.session_state.get("memo"):
        progress.markdown(climb_html(t, 4), unsafe_allow_html=True)
    if not api_key:
        st.caption(t["needkey"])

if st.session_state.get("memo"):
    st.markdown(f'<div class="memo-head">{t["memo"]}</div>', unsafe_allow_html=True)
    st.markdown(st.session_state["memo"])
    st.download_button(t["download"], st.session_state["memo"],
                       file_name=f"tenzing_memo_{date.today().isoformat()}.md")
    with st.expander(t["press"]):
        for n in st.session_state.get("news", []):
            st.markdown(f"- [{n['titre']}]({n['lien']}) — {n['source']}, {n['date']}")

st.markdown(f"""
<div class="foot">
  <span>{t['disclaimer']} {t['sources']}</span>
  <span>Nicolas Leonard de Juvigny · <a href="https://linkedin.com/in/nicolasjuvigny">LinkedIn</a></span>
</div>""", unsafe_allow_html=True)
