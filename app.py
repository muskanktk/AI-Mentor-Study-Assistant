import os
import io
import json
from io import BytesIO
from datetime import datetime
from urllib.parse import quote_plus

import streamlit as st
import pdfplumber
from gtts import gTTS
import httpx, certifi
from dotenv import load_dotenv

# -------- Optional OCR (kept off by default) ----------
try:
    import pypdfium2 as pdfium
    import pytesseract
    from PIL import Image  # noqa: F401
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# -------- Page & env ----------
st.set_page_config(page_title="AI Generated Study Assistant", layout="centered")
load_dotenv()
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

# -------- Secrets / API Keys ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")
    st.stop()

# -------- OpenAI client (SDK v1.x) ----------
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Choose your model:
# - "gpt-4o-mini" (cheap/fast, great for summaries)
# - "gpt-4o" (higher quality, more expensive)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------- Styling ----------
st.markdown("""
<style>
:root{ --bg:#f7f3e9; --card:#fff; --text:#111; --muted:#4b4b4b; --accent:#ff7a00; --accent-hover:#cc6300; --border:#e9e3d5;}
.stApp{ background:var(--bg); color:var(--text); }
p,span,label,li,div,code,.stMarkdown,.stText,.stCaption{ color:var(--text); }
a,a:visited{ color:var(--accent)!important; } a:hover{ color:var(--accent-hover)!important; }
h1,h2,h3,h4{ color:var(--text); font-weight:800; letter-spacing:.2px;}
.hero-sub{ color:var(--muted); }
.stTabs [role="tablist"]{ border-bottom:1px solid var(--border); gap:.25rem;}
.stTabs [role="tab"]{ background:transparent; color:var(--muted); border:none; padding:10px 16px; font-weight:700;}
.stTabs [role="tab"]:hover{ color:var(--text);}
.stTabs [role="tab"][aria-selected="true"]{ color:var(--text); border-bottom:3px solid var(--accent);}
.stButton > button{ background:var(--accent); color:#fff; border-radius:10px; border:1px solid var(--accent);
  padding:10px 18px; font-weight:800; letter-spacing:.3px; box-shadow:0 2px 8px rgba(0,0,0,.08);}
.stButton > button:hover{ background:var(--accent-hover); border-color:var(--accent-hover); color:#fff;}
.stSelectbox label,.stCheckbox label{ color:var(--text);}
.stFileUploader{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:14px; box-shadow:0 4px 16px rgba(0,0,0,.06);}
.card{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px; box-shadow:0 6px 20px rgba(0,0,0,.07); margin-bottom:20px;}
.card h3{ color:var(--accent-hover); margin-top:0; font-weight:900;}
.stDownloadButton > button{ background:#fff; color:var(--text); border:1px solid var(--border); border-radius:10px; padding:8px 14px;}
.stDownloadButton > button:hover{ border-color:var(--accent); color:var(--accent-hover);}
</style>
""", unsafe_allow_html=True)

# -------- Hero ----------
st.markdown("""
<div style="text-align:center; padding: 32px 0 8px 0;">
  <h1 style="font-size: 3rem; margin: 0;">
    <span>Chat with </span><span style="color:#ff7a00;">AI</span>
  </h1>
  <p class="hero-sub" style="font-size: 1.05rem; margin-top: 8px;">
    Simply upload your notes and get <b>visual</b>, <b>verbal</b>, and <b>hands-on</b> study resources.
  </p>
</div>
""", unsafe_allow_html=True)

# -------- UI helpers ----------
def start_card(title: str):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)

def end_card():
    st.markdown("</div>", unsafe_allow_html=True)

# -------- Upload ----------
start_card("Upload Lecture Notes")
uploaded = st.file_uploader("Upload Lecture Notes (PDF)", type=["pdf"], key="notes_pdf")
end_card()

# -------- Non-AI helpers ----------
def fetch_videos_for_topics(topics, per_topic=3):
    results = {}
    for t in topics:
        q = quote_plus(f"{t} explained for beginners")
        url = f"https://www.youtube.com/results?search_query={q}"
        results[t] = [{"title": f"YouTube search: {t}", "url": url, "note": "Open for multiple videos"}]
    return results

def build_sim_search_links(topic: str):
    sites = [
        ('PhET (science sims)',       'site:phet.colorado.edu "{q}" simulation'),
        ('Khan Academy practice',     'site:khanacademy.org "{q}" practice'),
        ('Desmos activities',         'site:teacher.desmos.com "{q}" activity'),
        ('GeoGebra interactives',     'site:geogebra.org "{q}"'),
        ('FreeCodeCamp (coding)',     'site:freecodecamp.org "{q}" tutorial'),
        ('W3Schools Try-It (coding)', 'site:w3schools.com "Tryit" "{q}"'),
        ('MDN Playground (JS)',       'site:developer.mozilla.org "{q}" example'),
    ]
    items = []
    for title, template in sites:
        query = template.replace("{q}", topic)
        url = "https://www.google.com/search?q=" + quote_plus(query)
        items.append({"title": f"{title} — {topic}", "url": url, "note": "Search results"})
    return items

def truncate_chars(s: str, max_chars: int = 12000) -> str:
    if s is None:
        return ""
    return s[:max_chars]

# -------- PDF extraction (OCR off by default) ----------
def extract_pdf_text(uploaded_file, use_ocr=False, min_text_chars=1200) -> str:
    if uploaded_file is None:
        return ""
    raw = uploaded_file.getvalue()

    text_plumber = ""
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t:
                    text_plumber += t + "\n"
    except Exception:
        text_plumber = ""

    need_ocr = use_ocr or (len(text_plumber.strip()) < min_text_chars)
    if not need_ocr:
        return text_plumber.strip()

    if not OCR_AVAILABLE:
        return (text_plumber + "\n\n(OCR not available — install pypdfium2, pytesseract, pillow)").strip()

    ocr_text = []
    try:
        doc = pdfium.PdfDocument(raw)
        n = len(doc)
        for i in range(n):
            page = doc[i]
            pil_img = page.render(scale=2.5).to_pil()
            ocr_text.append(pytesseract.image_to_string(pil_img))
    except Exception as e:
        ocr_text.append(f"(OCR failed: {e})")

    combined = (text_plumber + "\n" + "\n".join(ocr_text)).strip()
    return combined if combined else text_plumber.strip()

# -------- OpenAI helpers ----------
def openai_complete(prompt: str, temperature=0.25, max_tokens=900) -> str:
    """
    Uses OpenAI Chat Completions API (SDK v1.x).
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        st.error(f"OpenAI Error: {type(e).__name__}: {e}")
        return "(AI failed)"

def split_text(text: str, max_chars: int = 15000, overlap: int = 800) -> list:
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunks.append(text[i:end])
        if end == n: break
        i = max(0, end - overlap)
    return chunks

def smart_complete(prompt: str, temperature=0.25, max_output_tokens=900) -> str:
    # Always use OpenAI in this cloud-only version
    return openai_complete(prompt, temperature=temperature, max_tokens=max_output_tokens)

def ai_summarize_full(full_text: str) -> str:
    chunks = split_text(full_text, max_chars=15000, overlap=800)
    if not chunks:
        return "No text found in the PDF."
    st.write(f"Detected {len(chunks)} chunks. Summarizing with **OpenAI ({OPENAI_MODEL})**…")
    prog = st.progress(0)

    chunk_summaries = []
    for i, ch in enumerate(chunks, start=1):
        prompt = (
            "You are an expert CS tutor. Summarize these notes into a short, "
            "clear mini-section in Markdown that includes:\n"
            "• Key ideas (bullets)\n"
            "• ONE vivid analogy\n"
            "• ONE concrete example\n"
            "• ONE mini ASCII sketch if helpful (<= 6 lines)\n"
            "• Common pitfalls\n"
            "• Key terms\n"
            "Keep it ≤ 220 words. Be accurate and simple.\n\n"
            f"Chunk {i} of {len(chunks)}:\n\n{truncate_chars(ch, 11800)}"
        )
        s = smart_complete(prompt, temperature=0.2, max_output_tokens=600)
        chunk_summaries.append(f"### Chunk {i}\n{s}")
        prog.progress(i / len(chunks))

    merged_input = "\n\n".join(chunk_summaries)
    prompt = (
        "Merge these chunk summaries into ONE cohesive study guide in Markdown. "
        "Include these exact sections:\n"
        "## Overview\n"
        "## Big-picture Analogies (2-3)\n"
        "## Visual Intuition (ASCII Sketches)\n"
        "## Key Concepts\n"
        "## How It Works (Step-by-step)\n"
        "## Worked Examples in C\n"
        "## Common Pitfalls\n"
        "## Quick Reference\n"
        "## Self-check Questions\n\n"
        "Rules:\n"
        "- 800-1200 words total\n"
        "- Use **bold** for key terms\n"
        "- Include 1-2 small ASCII diagrams\n"
        "- Include C code snippets for arrays/strings\n\n"
        f"Content to merge:\n\n{truncate_chars(merged_input, 28000)}"
    )
    final_summary = smart_complete(prompt, temperature=0.25, max_output_tokens=1200)
    return final_summary

def ai_extract_topics(full_text: str, max_topics: int = 6) -> list:
    prompt = (
        "From these notes, extract 3-8 concise study topics as a JSON list of strings. "
        'Format: ["topic1", "topic2"]. Only return JSON.\n\n'
        f"Notes:\n{truncate_chars(full_text, 8000)}"
    )
    try:
        response = smart_complete(prompt, temperature=0.2, max_output_tokens=280)
        clean_json = response.replace("```json", "").replace("```", "").strip()
        topics = json.loads(clean_json)
        return [str(t).strip() for t in topics if t and len(str(t)) <= 60][:max_topics]
    except Exception:
        return []

# -------- Main flow ----------
def start_card(title: str):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)

def end_card():
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded:
    tabs = st.tabs(["Visual", "Verbal", "Hands-on", "Export"])

    if "extracted_text" not in st.session_state:
        st.session_state["extracted_text"] = None
    if "full_summary" not in st.session_state:
        st.session_state["full_summary"] = None

    def ensure_text():
        if st.session_state["extracted_text"] is None:
            with st.spinner("Extracting text from PDF…"):
                st.session_state["extracted_text"] = extract_pdf_text(uploaded, use_ocr=False)
        return st.session_state["extracted_text"]

    @st.cache_data(show_spinner=False)
    def tts_bytes(text: str, lang: str = "en", slow: bool = False) -> bytes:
        if not text or not text.strip():
            return b""
        fp = BytesIO()
        gTTS(text, lang=lang, slow=slow).write_to_fp(fp)
        fp.seek(0)
        return fp.read()

    def safe_slug(prefix: str = "summary") -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}"

    # --- Visual ---
    with tabs[0]:
        st.subheader("🎥 Visual (Videos)")
        if st.button("Detect Topics"):
            text = ensure_text()
            detected_topics = ai_extract_topics(text, max_topics=6)
            if not detected_topics:
                words = text.split()
                unique_words = list(dict.fromkeys(words))
                detected_topics = unique_words[:5] if unique_words else ["Algorithms"]
            st.success("Topics detected:")
            st.write(", ".join(detected_topics))

            videos = fetch_videos_for_topics(detected_topics)
            for topic, items in videos.items():
                st.markdown(f"**{topic}**")
                for v in items:
                    st.markdown(f"- [{v['title']}]({v['url']}) — {v['note']}")
            st.session_state["detected_topics"] = detected_topics
            st.session_state["videos"] = videos
        else:
            st.info("Click **Detect Topics** to populate video resources.")

    # --- Verbal ---
    with tabs[1]:
        st.subheader("📝 Verbal (Rich FULL Summary + Voiceover)")
        text = ensure_text()

        if st.button("Generate Rich FULL Summary (entire PDF)"):
            with st.spinner("Summarizing the whole PDF (map → reduce)…"):
                st.session_state["full_summary"] = ai_summarize_full(text)

        if st.session_state["full_summary"]:
            full_summary = st.session_state["full_summary"]
            st.markdown(full_summary)

            full_name = safe_slug("full_summary") + ".md"
            st.download_button(
                "Download full_summary.md",
                data=full_summary.encode("utf-8"),
                file_name=full_name,
                mime="text/markdown",
            )

            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                lang = st.selectbox("Voice", ["en", "ur", "hi"], index=0)
            with col2:
                slow = st.checkbox("Slow", value=False)

            if st.button("Generate Voiceover (from FULL summary)"):
                with st.spinner("Generating voiceover…"):
                    audio_data = tts_bytes(full_summary, lang=lang, slow=slow)
                if audio_data:
                    st.audio(audio_data, format="audio/mp3")
                    st.download_button(
                        "Download voiceover.mp3",
                        data=audio_data,
                        file_name="voiceover.mp3",
                        mime="audio/mpeg",
                    )
        else:
            st.info("Click **Generate Rich FULL Summary (entire PDF)** to create a comprehensive, analogy-filled summary and voiceover.")

    # --- Hands-on ---
    with tabs[2]:
        st.subheader("🧪 Hands-on (Interactives)")
        detected_topics = st.session_state.get("detected_topics", [])
        if not detected_topics:
            st.info("Detect topics first (in the Visual tab) to get interactive resources.")
        else:
            sims = {topic: build_sim_search_links(topic) for topic in detected_topics}
            for topic, items in sims.items():
                st.markdown(f"**{topic}**")
                for it in items:
                    st.markdown(f"- [{it['title']}]({it['url']}) — {it['note']}")
            st.session_state["sims"] = sims

    # --- Export ---
    with tabs[3]:
        st.subheader("📦 Export Study Pack")
        detected_topics = st.session_state.get("detected_topics", [])
        videos = st.session_state.get("videos", {})
        sims = st.session_state.get("sims", {})

        topics_md = "\n".join(f"- {t}" for t in detected_topics) if detected_topics else "_No topics detected yet._"

        visual_lines = []
        for topic, items in videos.items():
            visual_lines.append(f"### {topic}")
            for v in items:
                visual_lines.append(f"- [{v['title']}]({v['url']}) — {v['note']}")
        visual_md = "\n".join(visual_lines) if visual_lines else "_No visual links._"

        hands_lines = []
        for topic, items in sims.items():
            hands_lines.append(f"### {topic}")
            for it in items:
                hands_lines.append(f"- [{it['title']}]({it['url']}) — {it['note']}")
        hands_md = "\n".join(hands_lines) if hands_lines else "_No interactives._"

        full_text = st.session_state.get("extracted_text", "") or ""
        full_summary = st.session_state.get("full_summary", "")

        study_md = f"""# Study Pack

## Detected Topics
{topics_md}

---

## Visual Resources
{visual_md}

---

## Hands-on Interactives
{hands_md}

---

## Full Summary
{(full_summary if full_summary else "_(Generate the full summary in the Verbal tab to include it here.)_")}

---

## Notes Preview
{(full_text[:800] + "…") if len(full_text) > 800 else full_text}
"""
        st.write("Click to download your current study pack (Markdown).")
        st.download_button(
            "Download study_pack.md",
            data=study_md.encode("utf-8"),
            file_name="study_pack.md",
            mime="text/markdown",
        )

# -------- Footer ----------
st.caption(f"AI-powered study assistant — OpenAI ({OPENAI_MODEL})")
