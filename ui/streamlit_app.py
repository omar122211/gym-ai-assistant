"""Chat UI for the Gym AI Assistant.  Run the API first, then:  streamlit run ui/streamlit_app.py

The UI only talks to the FastAPI backend (API_URL), it never sees the Gemini key.
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
EXAMPLES = [
    "How much is the yearly membership?",
    "I'd like to book a boxing class",
    "متى حصص اليوغا؟",
    "Kick boks için hangi antrenörle çalışmalıyım?",
]
ACTION_LABELS = {
    "booked": "✅ Booked", "need_info": "📝 Needs details", "full": "⛔ Class full", "already_booked": "ℹ️ Already booked",
    "not_found": "🔎 Not found", "schedule": "📅 Schedule", "ticket": "🎫 Ticket created",
}

st.set_page_config(page_title="Fitness Abbas · AI Assistant", page_icon="💪", layout="centered")
# Static CSS only (no user text): Arabic paragraphs read right-to-left, others left-to-right
st.markdown("<style>[data-testid='stChatMessage'] p{unicode-bidi:plaintext;text-align:start}</style>",
            unsafe_allow_html=True)


def new_conversation() -> None:
    st.session_state.conversation_id = uuid.uuid4().hex  # random: it is the only key to this chat's memory
    st.session_state.messages = []


if "conversation_id" not in st.session_state:
    new_conversation()


def call_api(method: str, path: str, **kwargs) -> dict:
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
    except httpx.HTTPError:
        return {"error": "The assistant service is not reachable. Is the API running?"}
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        return {"error": detail if isinstance(detail, str) else "Something went wrong — please try again."}
    return r.json()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("💪 Fitness Abbas")
    st.caption("AI receptionist · answers in Arabic, Turkish or English")
    st.button("New conversation", on_click=new_conversation, width="stretch")

    st.subheader("Class schedule")
    data = call_api("GET", "/schedule")
    if "error" in data:
        st.warning(data["error"])
    else:
        st.dataframe(
            [{"Class": s["class_name"], "Day": s["day"][:3], "Time": s["time"], "Spots": s["spots_left"]}
             for s in data["sessions"]],
            hide_index=True, width="stretch",
        )
        st.caption("Demo timetable — the real gym has no public schedule.")

    st.divider()
    st.caption("🔒 **Privacy:** this demo uses the free Gemini API, where Google may use messages to improve "
               "its products. Don't share personal information — a first name is enough for bookings.")

# ---------------------------------------------------------------- chat
st.title("Gym AI Assistant")
st.caption("Ask about plans, trainers, workouts and nutrition — or book a class and report a problem.")

if not st.session_state.messages:
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
            st.session_state.pending = example
            st.rerun()


def show_meta(meta: dict) -> None:
    badges = []
    if meta.get("action_type") in ACTION_LABELS:
        badges.append(ACTION_LABELS[meta["action_type"]])
    if meta.get("grounded"):
        badges.append("📚 From the knowledge base")
    if meta.get("latency_ms") is not None:
        badges.append(f"{meta['latency_ms']} ms")
    if badges:
        st.caption(" · ".join(badges))
    if meta.get("sources"):
        with st.expander("Sources"):
            for s in meta["sources"]:
                st.write(f"- `{s['file']}` › {s['section']} (similarity {s['similarity']:.2f})")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])  # Streamlit escapes HTML by default
        if m.get("meta"):
            show_meta(m["meta"])

prompt = st.chat_input("Type your message…", max_chars=500) or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = call_api("POST", "/chat", json={"message": prompt,
                                                     "conversation_id": st.session_state.conversation_id})
        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {result['error']}"})
        else:
            meta = {"action_type": (result.get("action") or {}).get("type"), "grounded": result["grounded"],
                    "sources": result["sources"], "latency_ms": result["latency_ms"]}
            st.markdown(result["reply"])
            show_meta(meta)
            st.session_state.messages.append({"role": "assistant", "content": result["reply"], "meta": meta})
            if meta["action_type"] == "booked":
                st.rerun()  # refresh the free spots in the sidebar
