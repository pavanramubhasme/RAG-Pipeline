"""Streamlit demo UI. Talks to the API over HTTP like any other client.

uv run python -m streamlit run ui/app.py      (the API must be running first)

Chat box: Up/Down walk through the questions asked this session; "Stop" ends the wait for an
answer and puts the question back into the box to be edited (ui/chat_state.py).

With access control on, the sidebar's "Signed in as" switcher simulates logging in as one
of the demo users: it sends that user's API key, and the server decides what it may see.
"""

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import httpx
import streamlit as st
from chat_state import api_history, browser_script, finish, interrupted, start, stop

# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and can add ~2 s per request
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")
TRACES_URL = os.getenv("RAG_TRACES_URL", "http://127.0.0.1:6006")
DEFAULT_KEY = os.getenv("RAG_API_KEY", "")
IDENTITIES_FILE = Path(os.getenv("RAG_IDENTITIES_FILE", "config/identities.json"))
HISTORY_TURNS = 6  # earlier messages sent with each question so follow-ups work
LOW_CONFIDENCE = 0.3  # warn when the best cited passage has a lower reranker score than this

st.set_page_config(page_title="RAG Showcase", page_icon="📚", layout="wide")
# The previous run ended before its answer arrived (Stop, a new question, any click): keep the
# question in the conversation as stopped.
interrupted(st.session_state)


@st.cache_resource
def api() -> httpx.Client:
    # trust_env=False: talk to the API directly. A company proxy from HTTP(S)_PROXY would otherwise
    # intercept http://127.0.0.1:8000 and answer with an HTML error page instead of our JSON.
    return httpx.Client(base_url=API_URL, trust_env=False)


@st.cache_resource
def workers() -> ThreadPoolExecutor:
    # The request runs in a thread while the page keeps updating, so Stop is handled at once.
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="query")


def wait(future: Future, status) -> httpx.Response:
    """Wait for the answer, updating the page every 0.25 s: each update is a point where
    Streamlit can end this run (Stop, a new question). The thread itself only does HTTP."""
    started = time.monotonic()
    while not future.done():
        status.caption(f"⏳ Retrieving, reasoning, verifying... {time.monotonic() - started:.0f} s")
        time.sleep(0.25)
    status.empty()
    return future.result()


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
    health: dict = {}
    try:
        response = api().get("/health", timeout=5)
        health = response.json()
        (st.success if health["status"] == "ok" else st.warning)(f"Status: {health['status']}")
        st.write(f"LLM: `{health['llm_model']}` ({'up' if health['llm'] else 'down'})")
        if health.get("grounding_model") and health["grounding_model"] != health["llm_model"]:
            st.write(f"Answers checked by: `{health['grounding_model']}`")
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
    tracing = health.get("tracing") if isinstance(health, dict) else None
    if tracing:
        name = "LangSmith" if tracing == "langsmith" else "Phoenix"
        st.markdown(
            f"[Open traces ({name})]({health.get('traces_url') or TRACES_URL}) · [API docs]({API_URL}/docs)"
        )
    else:
        st.markdown(f"Tracing off · [API docs]({API_URL}/docs)")

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
    st.caption("Tip: ↑ / ↓ in the chat box recall your earlier questions.")
    if st.button("New conversation"):
        st.session_state.history = []
        st.session_state.pop("prefill", None)
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("stopped"):
            st.caption("⏹ Stopped before an answer")

question = st.chat_input("e.g. What's the WFH quota?  (↑ for earlier questions)")
if question:
    start(st.session_state, question)
with st.sidebar:  # 1 px high: arrow keys, the Stop refill, and the send arrow as a stop square
    browser = st.empty()


def show_browser_state(busy: bool) -> None:
    state = st.session_state
    browser.iframe(browser_script(state.get("asked", []), state.get("prefill"), busy=busy), height=1)


def answer(question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if len(question.strip()) < 3:
            st.info("Please type a full question about the documents (at least 3 characters).")
            finish(st.session_state, question, None)
            return
        payload = {"question": question, "history": api_history(st.session_state, HISTORY_TURNS)}
        stop_slot, status = st.empty(), st.empty()
        stop_slot.button(
            "⏹ Stop",
            key=f"stop-{len(st.session_state.asked)}",
            on_click=stop,
            args=(st.session_state, question),
            help="Stop waiting and put the question back into the chat box to edit it",
        )
        future = workers().submit(api().post, "/query", json=payload, headers=auth(), timeout=300)
        try:
            r = wait(future, status)
        except httpx.HTTPError as exc:
            stop_slot.empty()
            st.error(f"API not reachable: {exc}")
            finish(st.session_state, question, None)
            return
        stop_slot.empty()
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
                finish(st.session_state, question, None)
                return
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
                    # As written in the document: rendered as Markdown, a line could become a
                    # giant heading and code would lose its indentation.
                    st.code(c["text"], language=None, wrap_lines=True)
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
            finish(st.session_state, question, data["answer"])
        if st.session_state.get("in_flight") == question:  # an error was shown: no answer to keep
            finish(st.session_state, question, None)


show_browser_state(busy=bool(question))
if question:
    answer(question)
    show_browser_state(busy=False)  # the answer is here: the stop square becomes the send arrow again