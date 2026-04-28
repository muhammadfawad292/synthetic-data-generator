import streamlit as st
import google.generativeai as genai
import json
import csv
import uuid
import os
import re
import io
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="SynthChat · Data Generator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ─────────────────────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── API Key Resolution ────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    """Try .env first, then Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

API_KEY = get_api_key()

# ── Prompt Builder ────────────────────────────────────────────────────────────
LENGTH_MAP = {"Short": 3, "Medium": 6, "Long": 10}

def build_prompt(niche: str, instructions: str, num_convos: int, length: str) -> str:
    turns = LENGTH_MAP[length]
    return f"""You are an expert synthetic training data generator for conversational AI systems.

Generate exactly {num_convos} realistic, distinct conversations for a chatbot in the **{niche}** domain.

Additional instructions: {instructions if instructions else "None — use sensible defaults."}

Rules:
- Each conversation has exactly {turns} user-assistant exchange pairs (turns).
- Vary topics, user personas, emotional tones, and complexity across conversations.
- Keep conversations natural, human-like, and contextually accurate.
- Assistant responses should be helpful, on-topic, and appropriately detailed.
- Never repeat the same scenario across conversations.

Return ONLY a valid JSON array. No markdown fences, no explanation, no extra text.
Use this exact structure:

[
  {{
    "id": "uuid-string",
    "niche": "{niche}",
    "messages": [
      {{"role": "user", "content": "..."}},
      {{"role": "assistant", "content": "..."}},
      ...
    ]
  }},
  ...
]
"""

# ── Gemini Call ───────────────────────────────────────────────────────────────
def generate_conversations(niche, instructions, num_convos, length):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = build_prompt(niche, instructions, num_convos, length)
    response = model.generate_content(prompt)
    return response.text

# ── Response Cleaner ──────────────────────────────────────────────────────────
def clean_and_parse(raw: str) -> list[dict]:
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    data = json.loads(raw)
    # Inject UUIDs if missing
    for conv in data:
        if not conv.get("id"):
            conv["id"] = str(uuid.uuid4())
    return data

# ── Export Helpers ────────────────────────────────────────────────────────────
def to_json_bytes(data: list[dict]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

def to_csv_bytes(data: list[dict]) -> bytes:
    rows = []
    for conv in data:
        for msg in conv["messages"]:
            rows.append({
                "conversation_id": conv["id"],
                "niche": conv.get("niche", ""),
                "role": msg["role"],
                "content": msg["content"],
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🧬 AI Training Data</div>
  <h1 class="hero-title">SynthChat<span class="accent">.</span></h1>
  <p class="hero-sub">Generate realistic synthetic conversations to train your chatbots — fast, structured, domain-specific.</p>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.markdown("""
    <div class="alert-box">
      ⚠️ <strong>Gemini API key not found.</strong><br>
      Add <code>GEMINI_API_KEY=your_key</code> to a <code>.env</code> file locally,
      or add it under <em>Secrets</em> in your Streamlit Community app settings.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Input Panel ───────────────────────────────────────────────────────────────
# st.markdown('<div class="section-label">⚙️ Configuration</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    niche = st.text_input(
        "Niche / Topic Domain",
        placeholder="e.g. healthcare, e-commerce, legal, customer support…",
        help="The domain your chatbot will operate in.",
    )
    instructions = st.text_area(
        "Additional Instructions",
        placeholder="e.g. Friendly tone, include frustrated users, avoid medical jargon…",
        height=130,
        help="Guide the style, tone, personas, or edge cases you want covered.",
    )

with col2:
    num_convos = st.slider("Number of Conversations", min_value=1, max_value=20, value=5)
    length = st.radio(
        "Conversation Length",
        options=["Short", "Medium", "Long"],
        horizontal=True,
        help="Short = 3 turns · Medium = 6 turns · Long = 10 turns",
    )
    total_messages = LENGTH_MAP[length] * num_convos
    st.markdown(f"""
    <div class="summary-box">
      <div class="summary-row"><span>Messages per conversation</span><span class="val">{LENGTH_MAP[length]}</span></div>
      <div class="summary-row"><span>Total messages</span><span class="val">{total_messages}</span></div>
      <!--<div class="summary-row"><span>= {LENGTH_MAP[length]} turns × {num_convos} convos × 2 msg/turn</span></div>-->
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
generate_btn = st.button("✦ Generate Conversations", use_container_width=True, type="primary")

# ── Generation ────────────────────────────────────────────────────────────────
if generate_btn:
    if not niche.strip():
        st.warning("Please enter a niche / topic domain.")
        st.stop()

    with st.spinner("SynthChat is Crafting conversations..."):
        try:
            raw = generate_conversations(niche.strip(), instructions.strip(), num_convos, length)
            conversations = clean_and_parse(raw)
            st.session_state["conversations"] = conversations
            st.session_state["meta"] = {
                "niche": niche,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "count": len(conversations),
            }
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse Gemini response as JSON: {e}")
            with st.expander("Raw Gemini output"):
                st.code(raw)
            st.stop()
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

# ── Output ────────────────────────────────────────────────────────────────────
if "conversations" in st.session_state:
    convos = st.session_state["conversations"]
    meta = st.session_state["meta"]

    st.markdown(f"""
    <div class="output-header">
      <div class="output-meta">
        <span class="tag">📁 {meta['niche']}</span>
        <span class="tag">💬 {meta['count']} conversations</span>
        <span class="tag">🕐 {meta['generated_at']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Download buttons
    dcol1, dcol2, _ = st.columns([2, 2, 4])
    with dcol1:
        st.download_button(
            "⬇ Download JSON",
            data=to_json_bytes(convos),
            file_name=f"synthchat_{meta['niche'].replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with dcol2:
        st.download_button(
            "⬇ Download CSV",
            data=to_csv_bytes(convos),
            file_name=f"synthchat_{meta['niche'].replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Render each conversation
    for i, conv in enumerate(convos, 1):
        with st.expander(f"Conversation {i}  ·  `{conv['id'][:8]}…`", expanded=(i == 1)):
            for msg in conv["messages"]:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f"""
                    <div class="bubble user-bubble">
                      <span class="bubble-label">👤 User</span>
                      <p>{content}</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bubble bot-bubble">
                      <span class="bubble-label">🤖 Assistant</span>
                      <p>{content}</p>
                    </div>""", unsafe_allow_html=True)
