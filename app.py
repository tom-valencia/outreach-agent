# app.py
# 🤖 AI Agentic Marketing Pipeline (Streamlit, Crawl4AI, Groq, SMTP) — Full Updated Code
# Changes in this version:
# - ADDED: "Select All" checkbox for recipients in the Writer tab.
# - ADDED: Settings for sender name and position to auto-append a signature.
# - ADDED: Email post-processing to replace/strip placeholders like [Name] and ensure no merge fields are sent.
# - Fix DuckDuckGo Instant Answer fallback by URL-encoding keywords
# - Enforce max results PER keyword (not global)
# - Add UI diagnostics when DDG returns zero results
# - Allow Research to run even if Crawl4AI is not installed (collect URLs only)
# - Fix md.split("", 1) bug -> use .splitlines()
# - Safer exception handling (surface useful info without crashing)
# - Minor copy tweaks for clearer errors
# - ADDED: Manual prospect entry (paste emails and optional fields)
# - FIX: Safe reset for manual input via flag + st.rerun() (avoid StreamlitAPIException)
# - SMTP: Support starttls/ssl/none, correct envelope sender, better errors, send_message()

import os
import re
import ssl
import smtplib
import sqlite3
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import List, Dict, Optional, Tuple
import time, random

import streamlit as st
from dotenv import load_dotenv
import asyncio

# Optional imports (Crawl4AI). The app will degrade gracefully if unavailable.
try:
    from crawl4ai import AsyncWebCrawler
    CRAWL_AVAILABLE = True
except Exception:
    CRAWL_AVAILABLE = False

# DuckDuckGo keyword search — prefer library, fallback to Instant Answer
DDG_AVAILABLE = False
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
    DDG_AVAILABLE = True
except Exception:
    pass

import aiohttp  # used for fallback Instant Answer API
from urllib.parse import quote_plus  # IMPORTANT: URL-encode DDG fallback queries

# ------- ENV & WINDOWS EVENT LOOP FIX -------
load_dotenv()

# On Windows, ensure Proactor loop so Playwright can spawn subprocesses
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

DB_PATH = os.environ.get("AGENT_DB_PATH", "agentic_marketing.db")

# ------- DB -------
def db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS prospects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                title TEXT,
                email TEXT UNIQUE,
                company TEXT,
                website TEXT,
                source_url TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prospect_id INTEGER,
                subject TEXT,
                body TEXT,
                status TEXT DEFAULT 'draft',  -- draft|approved|sent|failed
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (prospect_id) REFERENCES prospects(id)
            )
            """
        )
        conn.commit()

init_db()

# ------- UTILITIES -------
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
NAME_REGEX = re.compile(r"(?i)(CEO|Founder|Owner|Director|Head|Manager|Lead|CMO|Marketing|Growth)")

def extract_contacts(text: str) -> Tuple[List[str], Optional[str]]:
    emails = sorted(set(EMAIL_REGEX.findall(text)))
    title_match = NAME_REGEX.search(text)
    title = title_match.group(0) if title_match else None
    return emails, title

# ------- EMAIL POST-PROCESSING (no placeholders, add signature) -------
def _signature_from_settings() -> str:
    """Builds a signature from sender settings."""
    name = (st.session_state.get("sender_name") or os.environ.get("SENDER_NAME") or "").strip()
    title = (st.session_state.get("sender_title") or os.environ.get("SENDER_TITLE") or "").strip()
    if not name and not title:
        return ""
    if name and title:
        return f"\n\n—\n{name}\n{title}"
    return f"\n\n—\n{name or title}"

def _fill_and_strip_placeholders(text: str, prospect: Dict) -> str:
    """Replaces common tokens with data, then strips any leftover bracketed tokens."""
    repl = {
        "[Name]": (prospect.get("name") or "").strip(),
        "[FirstName]": (prospect.get("name") or "").split()[0] if prospect.get("name") else "",
        "[Company]": (prospect.get("company") or "").strip(),
        "[Title]": (prospect.get("title") or "").strip(),
        "[Website]": (prospect.get("website") or "").strip(),
    }
    # Specific replacements for common AI-generated placeholders
    for k, v in repl.items():
        if v:
            text = text.replace(k, v)

    # Generic templating variants: {{name}}, {name}
    text = re.sub(r"\{\{\s*name\s*\}\}|\{\s*name\s*\}", repl["[Name]"], text, flags=re.IGNORECASE)
    text = re.sub(r"\{\{\s*company\s*\}\}|\{\s*company\s*\}", repl["[Company]"], text, flags=re.IGNORECASE)

    # Strip any remaining single-word bracket tokens like [Something] or {{Something}}
    text = re.sub(r"\[[A-Za-z][A-Za-z_ -]{0,24}\]", "", text)
    text = re.sub(r"\{\{[A-Za-z][A-Za-z_ -]{0,24}\}\}", "", text)

    # Collapse extra spaces/newlines introduced by removals
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def finalize_email(subject: str, body: str, prospect: Dict) -> Tuple[str, str]:
    """Cleans placeholders and adds a signature to the email body."""
    final_subject = _fill_and_strip_placeholders(subject or "", prospect)
    final_body = _fill_and_strip_placeholders(body or "", prospect)

    sig = _signature_from_settings()
    if sig:
        clean_sig_name = (st.session_state.get("sender_name") or os.environ.get("SENDER_NAME") or "xxxxx").strip()
        if clean_sig_name not in final_body:
            final_body += sig

    return final_subject, final_body

# ------- DUCKDUCKGO SEARCH -------
def _ddg_text_with_backoff(ddgs: "DDGS", query: str, max_results: int, retries: int = 6):
    """
    Run ddgs.text with exponential backoff + jitter when DDG rate-limits (HTTP 202 / 'Ratelimit').
    Returns a list of result dicts (possibly empty).
    """
    delay = 1.5
    for _ in range(retries):
        try:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                safesearch="moderate",
                region="wt-wt",
                timelimit="y"
            ))
            return results
        except Exception as e:
            msg = str(e)
            if "Ratelimit" in msg or "202" in msg:
                sleep_for = delay + random.uniform(0, 0.8)
                time.sleep(sleep_for)
                delay = min(delay * 2, 20)
                continue
            raise
    return []

async def ddg_search_via_library(keywords: List[str], max_per_keyword: int) -> List[str]:
    urls: List[str] = []
    if not DDG_AVAILABLE:
        return urls
    with DDGS() as ddgs:
        for kw in [k.strip() for k in keywords if k.strip()]:
            try:
                count_for_kw = 0
                for r in ddgs.text(kw, max_results=max_per_keyword, safesearch="moderate"):
                    u = (r.get("href") or r.get("url") or "").strip()
                    if u:
                        urls.append(u)
                        count_for_kw += 1
                        if count_for_kw >= max_per_keyword:
                            break
            except Exception:
                continue
    return urls

def _collect_firsturls_from_related_topics(related_topics: list, max_count: int) -> List[str]:
    """DDG Instant Answer can nest results inside items with 'Topics'."""
    found: List[str] = []
    def add_url(u: str):
        if u and u not in found:
            found.append(u)
    for t in related_topics:
        if isinstance(t, dict):
            if "FirstURL" in t and t["FirstURL"]:
                add_url(t["FirstURL"])
                if len(found) >= max_count:
                    break
            sub = t.get("Topics")
            if isinstance(sub, list):
                for s in sub:
                    if isinstance(s, dict) and "FirstURL" in s and s["FirstURL"]:
                        add_url(s["FirstURL"])
                        if len(found) >= max_count:
                            break
        if len(found) >= max_count:
            break
    return found

async def ddg_search_via_instant_answer(keywords: List[str], max_per_keyword: int) -> List[str]:
    urls: List[str] = []
    async with aiohttp.ClientSession() as session:
        for kw in [k.strip() for k in keywords if k.strip()]:
            try:
                q = quote_plus(kw)
                api = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
                async with session.get(api, timeout=20) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        topics = data.get("RelatedTopics", []) or []
                        per_kw = _collect_firsturls_from_related_topics(topics, max_per_keyword)
                        urls.extend(per_kw)
                    else:
                        continue
            except Exception:
                continue
    return urls

async def search_duckduckgo(keywords: List[str], max_per_keyword: int = 25, rpm_limit: Optional[int] = 30) -> List[str]:
    """
    Throttled multi-keyword search using duckduckgo-search with backoff.
    rpm_limit: max keyword-requests per minute (None to disable). We simply sleep between keywords.
    """
    urls: List[str] = []
    spacing = 0.0
    if rpm_limit and rpm_limit > 0:
        spacing = max(60.0 / float(rpm_limit), 0.0)
    max_per_keyword = max(1, min(int(max_per_keyword), 50))

    if not DDG_AVAILABLE:
        return []

    with DDGS() as ddgs:
        for idx, kw in enumerate([k.strip() for k in keywords if k.strip()]):
            if idx > 0 and spacing > 0:
                time.sleep(spacing + random.uniform(0, spacing * 0.3))
            results = _ddg_text_with_backoff(ddgs, kw, max_per_keyword)
            if not results:
                continue
            for r in results:
                u = (r.get("href") or r.get("url") or "").strip()
                if u:
                    urls.append(u)

    seen, deduped = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped

# ------- GROQ (Writer Agent) -------
from groq import Groq

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and "groq_api_key" in st.session_state:
        api_key = st.session_state["groq_api_key"]
        os.environ["GROQ_API_KEY"] = api_key
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Fill it in Settings.")
    return Groq(api_key=api_key)

def generate_email(prospect: Dict, product_pitch: str, angle: str, tone: str, max_tokens: int = 800) -> Tuple[str, str]:
    client = get_groq_client()
    sys_prompt = (
        "You are a precise, friendly B2B email copywriter. "
        "Write concise, compliant outreach. NEVER use placeholders like [Name], {name}, or {{company}}. "
        "If the recipient’s name is unknown, write a natural opener without any placeholder."
    )
    user_prompt = f"""
Write a cold email to {{name}} (title: {{title}}) at {{company}} about: {product_pitch}
Goal/angle: {angle}
Tone: {tone}
Constraints:
- 120-160 words body
- Clear subject (<=8 words)
- Personalize with website/company cues if provided: website={{website}}, notes={{notes}}
- Avoid spammy phrasing; include a soft CTA and a one-line opt-out.
- Canadian audience: be mindful of CASL compliance.
- Do not include any placeholders or merge fields (e.g., [Name], {{Company}}). Write fully resolved copy.
- Do not invent a sender name/title — these will be appended by the system.
"""
    content_vars = {
        "name": prospect.get("name") or "there",
        "title": prospect.get("title") or "",
        "company": prospect.get("company") or "",
        "website": prospect.get("website") or "",
        "notes": prospect.get("notes") or "",
    }
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": json.dumps(content_vars)},
    ]
    resp = client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
    )
    full_text = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    subject = ""
    body_lines = []
    for l in lines:
        if l.lower().startswith("subject:") and not subject:
            subject = l.split(":", 1)[1].strip()
        else:
            body_lines.append(l)
    if not subject and lines:
        subject = lines[0][:80]
        body_lines = lines[1:]
    body = "\n".join(body_lines)
    return subject, body

def save_draft(prospect_id: int, subject: str, body: str, model: str = "moonshotai/kimi-k2-instruct") -> int:
    with db_conn() as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO drafts (prospect_id, subject, body, model, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (prospect_id, subject, body, model),
        )
        conn.commit()
        return c.lastrowid

# ------- SMTP Sending -------
def get_smtp_settings():
    return {
        "host": (st.session_state.get("smtp_host") or os.environ.get("SMTP_HOST") or "").strip(),
        "port": int(st.session_state.get("smtp_port") or os.environ.get("SMTP_PORT") or 587),
        "user": (st.session_state.get("smtp_user") or os.environ.get("SMTP_USER") or "").strip(),
        "password": st.session_state.get("smtp_pass") or os.environ.get("SMTP_PASS") or "",
        "from_name": st.session_state.get("smtp_from_name") or os.environ.get("SMTP_FROM_NAME") or "Outreach Bot",
        "from_email": (st.session_state.get("smtp_from_email") or os.environ.get("SMTP_FROM_EMAIL") or "").strip(),
        # NEW options for robust delivery:
        "security": (st.session_state.get("smtp_security") or os.environ.get("SMTP_SECURITY") or "starttls").lower(),  # starttls|ssl|none
        "envelope_from": (st.session_state.get("smtp_envelope_from") or os.environ.get("SMTP_ENVELOPE_FROM") or "").strip(),
        "timeout": 30,
    }

def send_email_smtp(to_email: str, subject: str, html_body: str) -> Tuple[bool, str]:
    cfg = get_smtp_settings()
    required = [cfg["host"], cfg["user"], cfg["password"], (cfg["from_email"] or cfg["user"])]
    if not all(required):
        return False, "Missing SMTP settings. Host/User/Pass/From are required."

    # Header From (displayed to recipients)
    header_from_email = cfg["from_email"] or cfg["user"]
    from_header = formataddr((cfg["from_name"], header_from_email))

    # Envelope MAIL FROM (best to use authenticated user)
    envelope_from = cfg["envelope_from"] or cfg["user"]

    # Build message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_header
    msg["To"] = to_email

    text_body = re.sub("<[^<]+?>", "", html_body or "")
    msg.attach(MIMEText(text_body or "", "plain", "utf-8"))
    msg.attach(MIMEText(html_body or "", "html", "utf-8"))

    context = ssl.create_default_context()
    try:
        # Choose transport based on security/port
        if cfg["security"] == "ssl" or int(cfg["port"]) == 465:
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=cfg["timeout"], context=context)
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=cfg["timeout"])

        with server:
            server.ehlo()
            if cfg["security"] == "starttls":
                server.starttls(context=context)
                server.ehlo()
            # 'none' means plain (use only on trusted networks/smarthosts)
            if cfg["user"]:
                server.login(cfg["user"], cfg["password"])

            # send_message ensures headers are used; set explicit envelope sender
            server.send_message(msg, from_addr=envelope_from, to_addrs=[to_email])

        return True, "Sent"
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Auth failed: {e}"
    except smtplib.SMTPRecipientsRefused as e:
        return False, f"Recipient refused: {e.recipients}"
    except smtplib.SMTPResponseException as e:
        return False, f"SMTP error {e.smtp_code}: {e.smtp_error.decode() if isinstance(e.smtp_error, bytes) else e.smtp_error}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# ------- RESEARCH AGENT (CRAWL) -------
async def crawl_and_extract(url: str) -> Dict:
    res = {"url": url, "emails": [], "company": None, "title_hint": None, "raw": ""}
    if not CRAWL_AVAILABLE:
        return res
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url=url)
            md = result.markdown or ""
            res["raw"] = md
            emails, title_hint = extract_contacts(md)
            res["emails"] = emails
            res["title_hint"] = title_hint
            title_line = md.splitlines()[0].strip("# ") if md else ""
            if title_line and len(title_line) <= 80:
                res["company"] = title_line
        except Exception:
            pass
    return res

async def research_pipeline(seed_urls: List[str]) -> List[Dict]:
    sem = asyncio.Semaphore(10)

    async def bounded_crawl(u: str) -> Dict:
        async with sem:
            return await crawl_and_extract(u)

    tasks = [bounded_crawl(u) for u in seed_urls]
    results = await asyncio.gather(*tasks)
    prospects = []
    for r in results:
        for email in r.get("emails", []):
            pid = upsert_prospect(
                email=email,
                name=None,
                title=r.get("title_hint"),
                company=r.get("company"),
                website=r.get("url"),
                source_url=r.get("url"),
                notes="auto-imported",
            )
            if pid:
                prospects.append({"id": pid, "email": email, "company": r.get("company"), "website": r.get("url")})
    return prospects

# ------- PROSPECT UPSERT -------
def upsert_prospect(email: str, name: Optional[str], title: Optional[str], company: Optional[str], website: Optional[str], source_url: str, notes: str = "") -> Optional[int]:
    if not email:
        return None
    with db_conn() as conn:
        c = conn.cursor()
        try:
            c.execute(
                """
                INSERT INTO prospects (name, title, email, company, website, source_url, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (name or "", title or "", email.lower(), company or "", website or "", source_url, notes or ""),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        c.execute("SELECT id FROM prospects WHERE email = ?", (email.lower(),))
        row = c.fetchone()
        return row[0] if row else None

# ------- MANUAL PROSPECT ENTRY HELPERS -------
def email_exists(email: str) -> bool:
    with db_conn() as conn:
        row = conn.execute("SELECT 1 FROM prospects WHERE email = ?", (email.lower(),)).fetchone()
        return bool(row)

def _parse_manual_line(line: str) -> Dict[str, Optional[str]]:
    """
    Accepts flexible formats, returns a dict with email + optional fields.
    Supported examples (one per line):
      - email@example.com
      - Name <email@example.com>
      - email@example.com, Company
      - email@example.com, Name, Title, Company, https://website.com, Notes here
    CSV-style splits by comma. Only email is required.
    """
    line = (line or "").strip()
    if not line:
        return {}

    m_angle = re.search(r"^(.*?)<\s*(" + EMAIL_REGEX.pattern + r")\s*>$", line)
    if m_angle:
        name = m_angle.group(1).strip().strip('"').strip()
        email = m_angle.group(2).strip().lower()
        return {"email": email, "name": name or None, "title": None, "company": None, "website": None, "notes": "manual"}

    m_email = EMAIL_REGEX.search(line)
    if not m_email:
        return {}
    email = m_email.group(0).strip().lower()

    parts = [p.strip() for p in line.split(",")]
    if len(parts) > 1:
        try:
            idx = next(i for i, p in enumerate(parts) if EMAIL_REGEX.fullmatch(p))
        except StopIteration:
            return {"email": email, "name": None, "title": None, "company": None, "website": None, "notes": "manual"}

        name = parts[idx + 1] if (idx + 1) < len(parts) and parts[idx + 1] else None
        title = parts[idx + 2] if (idx + 2) < len(parts) and parts[idx + 2] else None
        company = parts[idx + 3] if (idx + 3) < len(parts) and parts[idx + 3] else None
        website = parts[idx + 4] if (idx + 4) < len(parts) and parts[idx + 4] else None
        notes = parts[idx + 5] if (idx + 5) < len(parts) and parts[idx + 5] else "manual"
        return {
            "email": email,
            "name": name or None,
            "title": title or None,
            "company": company or None,
            "website": website or None,
            "notes": notes or "manual",
        }

    return {"email": email, "name": None, "title": None, "company": None, "website": None, "notes": "manual"}

def add_manual_prospects(input_text: str) -> Tuple[int, int, int, int]:
    """
    Parses text area input and upserts prospects.
    Returns (attempted_lines, valid_emails, inserted, duplicates).
    """
    lines = [l.strip() for l in (input_text or "").splitlines() if l.strip()]
    attempted = len(lines)
    valid = 0
    inserted = 0
    duplicates = 0

    seen_emails_session = set()

    for line in lines:
        data = _parse_manual_line(line)
        email = data.get("email")
        if not email:
            continue
        if email in seen_emails_session:
            duplicates += 1
            continue
        seen_emails_session.add(email)
        valid += 1

        if email_exists(email):
            duplicates += 1
            continue

        pid = upsert_prospect(
            email=email,
            name=data.get("name"),
            title=data.get("title"),
            company=data.get("company"),
            website=data.get("website"),
            source_url="manual",
            notes=data.get("notes") or "manual",
        )
        if pid is not None:
            inserted += 1

    return attempted, valid, inserted, duplicates

# ------- STREAMLIT UI -------
st.set_page_config(page_title="AI Agentic Marketing", layout="wide")
st.title("🤖 AI Agentic Marketing Pipeline")

with st.sidebar:
    st.header("Settings")

    # Put all inputs inside a real <form>
    with st.form("settings_form", clear_on_submit=False):
        st.text_input("GROQ_API_KEY", type="password", key="groq_api_key")
        st.subheader("SMTP")
        st.text_input("SMTP_HOST", key="smtp_host", value=os.environ.get("SMTP_HOST", ""))
        st.number_input("SMTP_PORT", key="smtp_port", value=int(os.environ.get("SMTP_PORT", 587)))
        st.text_input("SMTP_USER", key="smtp_user", value=os.environ.get("SMTP_USER", ""))
        st.text_input("SMTP_PASS", type="password", key="smtp_pass", value=os.environ.get("SMTP_PASS", ""))
        st.text_input("SMTP_FROM_NAME", key="smtp_from_name", value=os.environ.get("SMTP_FROM_NAME", "Outreach Bot"))
        st.text_input("SMTP_FROM_EMAIL", key="smtp_from_email", value=os.environ.get("SMTP_FROM_EMAIL", ""))

        # NEW fields for robust SMTP behavior
        default_sec = (os.environ.get("SMTP_SECURITY") or "starttls").lower()
        sec_options = ["starttls", "ssl", "none"]
        sec_index = sec_options.index(default_sec) if default_sec in sec_options else 0
        st.selectbox(
            "SMTP_SECURITY",
            sec_options,
            index=sec_index,
            key="smtp_security",
            help="587: starttls, 465: ssl, smarthost/relay: sometimes none."
        )
        st.text_input(
            "SMTP_ENVELOPE_FROM (optional)",
            key="smtp_envelope_from",
            value=os.environ.get("SMTP_ENVELOPE_FROM", ""),
            help="MAIL FROM (envelope sender). Leave blank to default to SMTP_USER."
        )

        st.subheader("Sender")
        st.text_input("Sender name", key="sender_name", value=os.environ.get("SENDER_NAME", ""))
        st.text_input("Sender position/title", key="sender_title", value=os.environ.get("SENDER_TITLE", ""))

        submitted = st.form_submit_button("Save settings")

    if submitted:
        st.toast("Settings saved")

st.markdown("""
This tool helps you:
1) **Research**: collect URLs from keywords (DuckDuckGo) and/or seed URLs, then crawl for contacts.
2) **Write**: generate tailored outreach drafts with Groq.
3) **Review/Send**: edit drafts, approve, and send via **SMTP**.

Quickstart
1) Python 3.10+
2) Install deps:

pip install -U streamlit groq crawl4ai python-dotenv duckduckgo-search aiohttp pandas
crawl4ai-setup
python -m playwright install --with-deps chromium

3) Env (or fill in Settings):

GROQ_API_KEY=...
SMTP_HOST=...
SMTP_PORT=587
SMTP_USER=you@example.com
SMTP_PASS=app_password
SMTP_FROM_NAME="Your Name"
SMTP_FROM_EMAIL=you@example.com
SMTP_SECURITY=starttls
# Optional:
SMTP_ENVELOPE_FROM=you@example.com
SENDER_NAME="Your Name"
SENDER_TITLE="Your Position"

4) Run:  streamlit run app.py
""")

research_tab, write_tab, drafts_tab = st.tabs(["🔎 Research", "✍️ Write", "📬 Drafts & Sending"])

with research_tab:
    st.subheader("Research Agent")
    if not CRAWL_AVAILABLE:
        st.warning("Crawl4AI not available. You can still collect URLs from keywords and seed URLs; crawling will be skipped.")

    st.markdown("**Input sources**")
    urls_input = st.text_area("Seed URLs (one per line)")

    use_ddg = st.checkbox("Search by keyword with DuckDuckGo", value=True)
    keywords_input = st.text_area(
        "Keywords (one per line)",
        placeholder="e.g. dentists toronto clinic email OR SR&ED consultants Toronto"
    )
    max_per_kw = st.number_input(
        "Max results per keyword",
        min_value=1, max_value=200, value=50, step=1,
        help="Limits DuckDuckGo results for each keyword."
    )

    # >>> Research action (above manual section)
    if st.button("Run Research"):
        seed_urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        urls = list(seed_urls)

        if use_ddg:
            keywords = [k.strip() for k in keywords_input.splitlines() if k.strip()]
            if keywords:
                st.write("Searching DuckDuckGo…")
                try:
                    ddg_urls = asyncio.run(search_duckduckgo(keywords, int(max_per_kw)))
                except RuntimeError:
                    ddg_urls = asyncio.get_event_loop().run_until_complete(
                        search_duckduckgo(keywords, int(max_per_kw))
                    )
                st.info(f"DuckDuckGo returned {len(ddg_urls)} URL(s) for {len(keywords)} keyword(s).")
                if not ddg_urls:
                    st.warning("No results from DuckDuckGo for the provided keywords. Try different terms or add seed URLs.")
                urls.extend(ddg_urls)
            else:
                st.caption("No keywords provided; skipping DuckDuckGo.")

        seen = set()
        all_urls = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)

        if not all_urls:
            st.error("No URLs to crawl. Provide seed URLs and/or keywords that return results.")
        else:
            st.write(f"Collected **{len(all_urls)}** URL(s).")
            if CRAWL_AVAILABLE:
                st.write("Crawling and extracting contacts…")
                with st.spinner("Crawling…"):
                    try:
                        prospects_added = asyncio.run(research_pipeline(all_urls))
                    except RuntimeError:
                        prospects_added = asyncio.get_event_loop().run_until_complete(research_pipeline(all_urls))
                st.success(f"Added {len(prospects_added)} prospect(s) from {len(all_urls)} URL(s).")
            else:
                st.info("Crawling is disabled (Crawl4AI not installed). URLs collected above.")

    # --- Manually add prospects (emails) ---
    # If last click asked to clear, do it *before* creating the widget this run
    if st.session_state.pop("reset_manual_emails", False):
        st.session_state["manual_emails"] = ""

    st.markdown("### Manually add prospects (emails)")
    st.caption("One per line. Accepted: `email@example.com`, `Name <email@example.com>`, or CSV-like: `email@example.com, Name, Title, Company, Website, Notes`.")
    manual_input = st.text_area(
        "Paste emails (and optional fields) here",
        key="manual_emails",
        height=140,
        placeholder="jane@acme.com\nJohn Smith <john@contoso.com>\nsales@widget.io, Widget Inc.\nowner@abc.com, Alice Chen, Owner, ABC Dental, https://abcdental.ca, inbound lead"
    )
    col_m1, col_m2 = st.columns([1, 4])
    with col_m1:
        if st.button("Add Prospects"):
            if not manual_input.strip():
                st.error("Please paste at least one line with a valid email.")
            else:
                attempted, valid, inserted, duplicates = add_manual_prospects(manual_input)
                if inserted:
                    st.success(f"Processed {attempted} line(s). Valid emails: {valid}. Added {inserted} new prospect(s).")
                if valid - inserted > 0 or duplicates > 0:
                    st.info(f"Duplicates or re-pasted within this submission: {duplicates}. Invalid lines: {attempted - valid}.")
                # Ask next run to clear the widget value, then rerun immediately
                st.session_state["reset_manual_emails"] = True
                st.rerun()
    with col_m2:
        st.caption("Tip: Include company/title/website to save time. Existing emails will not be duplicated.")

    st.divider()
    st.caption("Prospects in database")
    with db_conn() as conn:
        try:
            import pandas as pd
            df = pd.read_sql_query(
                "SELECT id, name, title, email, company, website, created_at FROM prospects ORDER BY created_at DESC",
                conn
            )
        except Exception:
            df = None
        if df is not None and not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No prospects yet.")

# ----- Writer Tab -----
with write_tab:
    st.subheader("Writer Agent")
    with db_conn() as conn:
        prospects = conn.execute(
            "SELECT id, name, title, email, company, website, notes FROM prospects ORDER BY created_at DESC"
        ).fetchall()
    if not prospects:
        st.info("No prospects. Add some in Research tab.")
    else:
        labels = [f"#{p[0]} {p[3]} — {p[4] or ''}" for p in prospects]

        select_all = st.checkbox("Select all prospects", value=False, key="select_all_prospects")

        if "prospect_sel" not in st.session_state:
            st.session_state["prospect_sel"] = []

        default_selection = st.session_state.get("prospect_sel", [])
        if select_all:
            default_selection = list(range(len(labels)))
            st.session_state["prospect_sel"] = default_selection

        sel = st.multiselect(
            "Select prospects",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            default=default_selection,
            key="prospect_sel",
        )

        product_pitch = st.text_input("What are you pitching?", value="SR&ED credits funding opportunities")
        angle = st.text_input("Angle/Goal", value="Book a 15-min call to assess eligibility and timelines")
        tone = st.selectbox("Tone", ["Professional", "Warm", "Concise", "Upbeat"], index=1)

        if st.button("Generate Draft(s)"):
            if not sel:
                st.error("Select at least one prospect.")
            else:
                results = []
                with st.spinner(f"Generating {len(sel)} drafts..."):
                    for i in sel:
                        p = prospects[i]
                        prospect_dict = {
                            "id": p[0],
                            "name": p[1],
                            "title": p[2],
                            "email": p[3],
                            "company": p[4],
                            "website": p[5],
                            "notes": p[6],
                        }
                        try:
                            subject, body = generate_email(prospect_dict, product_pitch, angle, tone)
                            subject, body = finalize_email(subject, body, prospect_dict)
                            draft_id = save_draft(prospect_dict["id"], subject, body)
                            results.append((prospect_dict["email"], draft_id))
                        except Exception as e:
                            st.error(f"Error for {prospect_dict['email']}: {e}")
                st.success(f"Created {len(results)} draft(s). See 'Drafts & Sending'.")

# ----- Drafts Tab -----
with drafts_tab:
    st.subheader("Manage Drafts")
    with db_conn() as conn:
        drafts = conn.execute(
            """
            SELECT drafts.id, drafts.prospect_id, drafts.subject, drafts.body, drafts.status, prospects.email
            FROM drafts
            JOIN prospects ON drafts.prospect_id = prospects.id
            ORDER BY drafts.created_at DESC
            """
        ).fetchall()

    if not drafts:
        st.info("No drafts yet.")
    else:
        for d in drafts:
            draft_id, prospect_id, subject, body, status, email = d
            with st.expander(f"Draft #{draft_id} → {email} [{status}]", expanded=False):
                new_subject = st.text_input("Subject", value=subject, key=f"sub_{draft_id}")
                new_body = st.text_area("Body (supports HTML)", value=body, height=220, key=f"body_{draft_id}")
                col1, col2, col3, col4 = st.columns([1,1,1,2])
                with col1:
                    if st.button("Save", key=f"save_{draft_id}"):
                        with db_conn() as conn:
                            conn.execute(
                                "UPDATE drafts SET subject=?, body=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (new_subject, new_body, draft_id),
                            )
                            conn.commit()
                        st.toast("Saved.")
                with col2:
                    if st.button("Approve", key=f"approve_{draft_id}"):
                        with db_conn() as conn:
                            conn.execute(
                                "UPDATE drafts SET status='approved', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (draft_id,),
                            )
                            conn.commit()
                        st.toast("Approved.")
                        st.rerun()
                with col3:
                    send_now = st.button("Send", key=f"send_{draft_id}")
                    if send_now:
                        ok, msg = send_email_smtp(email, new_subject, new_body)
                        with db_conn() as conn:
                            if ok:
                                conn.execute("UPDATE drafts SET status='sent', updated_at=CURRENT_TIMESTAMP WHERE id=?", (draft_id,))
                            else:
                                conn.execute("UPDATE drafts SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?", (draft_id,))
                            conn.commit()
                        if ok:
                            st.success("Email sent.")
                        else:
                            st.error(f"Send failed: {msg}")
                        st.rerun()
                with col4:
                    st.caption("Tip: Approved drafts are ready for batch sending.")

        st.divider()
        if st.button("Send All Approved"):
            sent, failed = 0, 0
            with db_conn() as conn:
                rows = conn.execute(
                    """
                    SELECT drafts.id, drafts.subject, drafts.body, prospects.email
                    FROM drafts JOIN prospects ON drafts.prospect_id = prospects.id
                    WHERE drafts.status='approved'
                    """
                ).fetchall()
            with st.spinner(f"Sending {len(rows)} approved emails..."):
                for rid, subj, bdy, em in rows:
                    ok, msg = send_email_smtp(em, subj, bdy)
                    with db_conn() as conn:
                        if ok:
                            conn.execute("UPDATE drafts SET status='sent', updated_at=CURRENT_TIMESTAMP WHERE id=?", (rid,))
                            sent += 1
                        else:
                            conn.execute("UPDATE drafts SET status='failed', updated_at=CURRENT_TIMESTAMP WHERE id=?", (rid,))
                            failed += 1
                        conn.commit()
            st.success(f"Batch done. Sent: {sent}, Failed: {failed}")
            st.rerun()

st.divider()
st.caption("Built with Streamlit · Crawl4AI · Groq · SMTP · DuckDuckGo | Demo only — ensure compliance with anti-spam laws and obtain consent where required.")
