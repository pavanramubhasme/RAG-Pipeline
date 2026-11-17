"""Streamlit demo UI. Talks to the API over HTTP like any other client.

uv run streamlit run ui/app.py      (the API must be running first)

With access control on, the sidebar's "Signed in as" switcher simulates logging in as one
of the demo users: it sends that user's API key, and the server decides what it may see.
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and can add ~2 s per request
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TRACES_URL = os.getenv("RAG_TRACES_URL", "http://127.0.0.1:6006")
DEFAULT_KEY = os.getenv("RAG_API_KEY", "")
IDENTITIES_FILE = Path(os.getenv("RAG_IDENTITIES_FILE", "config/identities.json"))
HISTORY_TURNS = 6  # earlier messages sent with each question so follow-ups work
LOW_CONFIDENCE = 0.3  # warn when the best cited passage has a lower reranker score than this

st.set_page_config(page_title="RAG Showcase", page_icon="📚", layout="wide")


@st.cache_resource
def api() -> httpx.Client:
    # trust_env=False: talk to the API directly. A company proxy from HTTP(S)_PROXY would otherwise
    # intercept http://127.0.0.1:8000 and answer with an HTML error page instead of our JSON.
    return httpx.Client(base_url=API_URL, trust_env=False)


def demo_users() -> dict[str, dict]:
    """Demo users from the identities file (a stand-in for a real login)."""
    try:
        return json.loads(IDENTITIES_FILE.read_text(encoding="utf-8")).get("users", {})
    except (OSError, ValueError):
        return {}


def auth() -> dict[str, str]:
    """The credential of the signed-in demo user; the server resolves it to a role."""
    return {"X-API-Key": st.session_state.get("api_key", DEFAULT_KEY)}


st.title("📚 Ask your documents")
st.caption("LangGraph agent · hybrid search (BM25 + dense) · cross-encoder rerank · local open-source LLM")

with st.sidebar:
    st.header("System")
    try:
        response = api().get("/health", timeout=5)
        health = response.json()
        (st.success if health["status"] == "ok" else st.warning)(f"Status: {health['status']}")
        st.write(f"LLM: `{health['llm_model']}` ({'up' if health['llm'] else 'down'})")
        st.write(f"Chunks indexed: **{health['chunks_indexed']}**")
        if health.get("cached_answers") is not None:
            st.write(f"Cached answers: **{health['cached_answers']}**")
        access_control = bool(health.get("access_control"))
    except httpx.HTTPError:
        access_control = False
        st.error(f"API not reachable at {API_URL}. Is it running (Terminal 2)?")
    except (ValueError, KeyError):  # something other than our API answered
        access_control = False
        st.error(
            f"{API_URL}/health answered HTTP {response.status_code} without the expected JSON. "
            "Is RAG_API_URL right, and is the API (not another program) on that port?"
        )
    st.markdown(f"[Open traces (Phoenix)]({TRACES_URL}) · [API docs]({API_URL}/docs)")

    if access_control:
        st.header("Signed in as")
        users = demo_users()
        if users:
            names = list(users)
            by_key = {spec["key"]: name for name, spec in users.items()}
            default = by_key.get(DEFAULT_KEY, "employee_demo" if "employee_demo" in users else names[0])
            current = st.session_state.get("user", default)
            choice = st.selectbox(
                "Demo user (simulates a login)", names, index=names.index(current) if current in names else 0
            )
            if choice != st.session_state.get("user"):
                if "user" in st.session_state:  # switched user: start a new conversation
                    st.session_state.history = []
                st.session_state.user = choice
                st.session_state.api_key = users[choice]["key"]
        me = api().get("/whoami", headers=auth(), timeout=5)
        if me.is_success:
            who = me.json()
            st.write(f"Role: **{who['role']}**")
            st.caption("May see: " + ", ".join(who["scopes"]))
        else:
            st.error("This key is not accepted by the API.")

    st.header("Add documents")
    uploads = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Ingest"):
        files = [("files", (f.name, f.getvalue())) for f in uploads]
        with st.spinner("Chunking and embedding..."):
            r = api().post("/ingest", files=files, headers=auth(), timeout=600)
        if r.is_success:
            st.json(r.json())
        elif r.status_code == 403:
            st.error("Your role may not add documents (an upload decides what others read). Try admin_demo.")
        else:
            st.error(r.text)

    st.header("Conversation")
    st.caption(f"Follow-up questions use the last {HISTORY_TURNS} messages as context.")
    if st.button("New conversation"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if question := st.chat_input("e.g. What's the WFH quota?"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if len(question.strip()) < 3:
            st.info("Please type a full question about the documents (at least 3 characters).")
            st.stop()
        previous_turns = st.session_state.history[-HISTORY_TURNS:]
        payload = {"question": question, "history": previous_turns}
        with st.spinner("Retrieving, reasoning, verifying..."):
            r = api().post("/query", json=payload, headers=auth(), timeout=300)
        if r.status_code == 422:
            st.warning("Please ask a question between 3 and 1000 characters.")
        elif r.status_code == 401:
            st.error("The API did not accept this user's key.")
        elif r.status_code == 503:
            st.error("The LLM is not reachable. Is Ollama running? (`ollama list`)")
        elif not r.is_success:
            st.error(f"Error {r.status_code}: {r.text[:300]}")
        else:
            data = r.json()
            if data.get("blocked"):
                st.warning(data["answer"])
                st.caption("Guardrails: " + ", ".join(data["guardrails"]))
                st.stop()
            st.markdown(data["answer"])
            if data["grounded"] is False:
                st.warning("Grounding check failed: verify this answer against the sources.")
            for c in data["citations"]:
                label = f"[{c['ref']}] {c['source']}" + (f" · {c['section']}" if c["section"] else "")
                if c.get("page"):
                    label += f" · page {c['page']}"
                if c.get("classification"):
                    label += f" · 🔒 {c['classification']}"
                with st.expander(f"{label}  —  relevance {c['score'] or 0:.2f}"):
                    st.write(c["text"])
            best = max((c["score"] or 0 for c in data["citations"]), default=None)
            if best is not None and best < LOW_CONFIDENCE:
                st.info(
                    f"Low confidence: the best matching passage scored only {best:.2f}. "
                    "Check the source, or ask more specifically (name the register, section or product)."
                )
            if data.get("standalone_question"):
                st.caption(f"Understood as: *{data['standalone_question']}*")
            if data.get("guardrails"):
                st.caption("Guardrails: " + ", ".join(data["guardrails"]))
            if data.get("access"):
                access = data["access"]
                st.caption(
                    f"🔐 Searched as **{access['user']}** ({access['role']}), "
                    f"only in: {', '.join(access['scopes'])}"
                )
            if data.get("cache_similar_to"):
                st.caption(
                    f"⚡ Answered from cache: same meaning as your earlier question "
                    f'*"{data["cache_similar_to"]}"* (similarity {data["cache_similarity"]:.2f}), '
                    "so no retrieval and no LLM call."
                )
            elif data.get("cached"):
                st.caption(
                    "⚡ Answered from cache: asked before, so no retrieval and no LLM call. "
                    "The cache is cleared whenever documents are added."
                )
            path = " → ".join(data["steps"])
            if data["search_query"]:
                path += f"  (rewritten query: *{data['search_query']}*)"
            st.caption(f"Agent path: {path}")
            st.caption(" · ".join(f"{k} {v:.0f} ms" for k, v in data["timings_ms"].items()))
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "assistant", "content": data["answer"]})
