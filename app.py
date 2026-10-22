"""
Tenzing
From a French startup name to a one-page investment memo, using public sources only.
Sources: API Recherche d'entreprises (SIRENE, data.gouv.fr), company website, Google News RSS.
LLM: Groq
"""

import time
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

st.set_page_config(page_title="Tenzing", page_icon="🏔️", layout="wide")


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
# 3. Interface
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")
    provider = st.radio("Moteur d'analyse", ["Groq", "Mistral", "Claude"], horizontal=True,
                        help="Groq et Mistral : offres gratuites. Claude : payant.")
    secret_name = PROVIDERS[provider]["secret"] if provider in PROVIDERS else "ANTHROPIC_API_KEY"
    try:
        secret_key = st.secrets.get(secret_name, "")
    except Exception:  # no secrets file locally
        secret_key = ""
    api_key = secret_key or st.text_input(f"Clé API {provider}", type="password",
                                          help="Utilisée uniquement pour cette session, jamais stockée.")
    language = st.radio("Langue de la note", ["français", "English"], horizontal=True)
    st.divider()
    st.caption("Sources : SIRENE (data.gouv.fr), site web, Google News. Moteur : Groq, Mistral AI ou Claude. "
               "Outil pédagogique, ne constitue pas un conseil en investissement.")

st.title("🏔️ Tenzing")
st.write("Le sherpa de votre due diligence : du nom d'une startup française à une note d'investissement d'une page, à partir de sources publiques.")

col1, col2 = st.columns([2, 2])
name = col1.text_input("Nom de l'entreprise", placeholder="ex. Pennylane")
url = col2.text_input("Site web (optionnel)", placeholder="ex. pennylane.com")

if st.button("1. Rechercher dans le registre", type="primary", disabled=not name):
    with st.spinner("Interrogation du registre SIRENE..."):
        try:
            st.session_state["results"] = search_companies(name)
        except requests.RequestException as e:
            st.error(f"Erreur API SIRENE : {e}")
            st.session_state["results"] = []

results = st.session_state.get("results", [])
if results:
    labels = [f"{c.get('nom_complet')} — SIREN {c.get('siren')} — {(c.get('siege') or {}).get('libelle_commune', '')}"
              for c in results]
    idx = st.selectbox("Sélectionne la bonne entité juridique", range(len(labels)), format_func=lambda i: labels[i])
    profile = company_profile(results[idx])

    with st.expander("Fiche registre (SIRENE)", expanded=True):
        fin = profile.pop("Finances publiées")
        dirigeants = profile.pop("Dirigeants")
        st.table({k: [str(v)] for k, v in profile.items()})
        st.write("**Dirigeants :** " + (", ".join(dirigeants) or "non publiés"))
        if fin:
            st.write("**Finances publiées :**")
            st.dataframe(fin, hide_index=True)
        else:
            st.info("Aucun compte publié (fréquent pour les startups qui demandent la confidentialité).")
        profile["Dirigeants"] = dirigeants
        profile["Finances publiées"] = fin

    if st.button("2. Générer la note d'investissement", disabled=not api_key):
        with st.spinner("Collecte du site et de la presse..."):
            website = fetch_website(url)
            news = fetch_news(profile["Dénomination"] or name)
        with st.spinner("Rédaction de la note (jusqu'à 1 minute avec l'offre gratuite)..."):
            try:
                memo = generate_memo(provider, api_key, build_context(profile, website, news), language)
            except Exception as e:
                st.error(f"Erreur du moteur {provider} : {e}")
                st.stop()
        st.session_state["memo"] = memo
        st.session_state["news"] = news

    if not api_key:
        st.caption("Renseigne une clé API dans la barre latérale pour générer la note.")

if st.session_state.get("memo"):
    st.divider()
    st.markdown(st.session_state["memo"])
    st.download_button("Télécharger la note (.md)", st.session_state["memo"],
                       file_name=f"tenzing_memo_{date.today().isoformat()}.md")
    with st.expander("Sources presse utilisées [S3]"):
        for n in st.session_state.get("news", []):
            st.markdown(f"- [{n['titre']}]({n['lien']}) — {n['source']}, {n['date']}")
