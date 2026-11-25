"""Chat bookkeeping for the Streamlit UI: question recall (arrow keys) and Stop.

Kept free of Streamlit calls, so it works on any dict-like ``state`` (``st.session_state`` in the
app, a plain dict in the tests).

* ``asked``: every question sent this session, oldest first. Up/Down in the chat box walks
  through it, like a shell or ChatGPT.
* ``in_flight``: the question whose answer is still being computed. If a run ends without an
  answer (Stop, a new question, any click that reruns the page), it is kept in the conversation
  as "stopped" and, after Stop, put back into the chat box to be edited.
* ``history``: what the conversation shows. Stopped questions have no answer, so they are left
  out of the history sent to the API.

While an answer is being computed, the chat box's send arrow turns into a stop square, like
ChatGPT: Streamlit's send button can't be changed, so a stop button is drawn exactly over it and
clicks the (hidden) Streamlit "Stop" button, whose callback does the bookkeeping.
"""

import json
import re
import uuid
from collections.abc import MutableMapping


def remember(state: MutableMapping, question: str) -> None:
    """Add to the arrow-key history (no immediate repeats)."""
    asked = state.setdefault("asked", [])
    if not asked or asked[-1] != question:
        asked.append(question)


def start(state: MutableMapping, question: str) -> None:
    remember(state, question)
    state["in_flight"] = question
    state.pop("prefill", None)  # a new question replaces an edit in progress


def interrupted(state: MutableMapping) -> str | None:
    """Record an unanswered question as stopped; returns it (None if nothing was running)."""
    question = state.pop("in_flight", None)
    if question:
        state.setdefault("history", []).append({"role": "user", "content": question, "stopped": True})
    return question


def stop(state: MutableMapping, question: str) -> None:
    """The Stop button: keep the question as stopped and put it back into the chat box."""
    if state.get("in_flight") == question:
        interrupted(state)
    state["prefill"] = {"text": question, "id": uuid.uuid4().hex}


def finish(state: MutableMapping, question: str, answer: str | None) -> None:
    """The run ended normally; ``answer`` None = no answer to keep (an error or a refusal to answer)."""
    state.pop("in_flight", None)
    if answer is not None:
        history = state.setdefault("history", [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def api_history(state: MutableMapping, turns: int) -> list[dict]:
    """The last ``turns`` answered messages, as the API expects them (no stopped questions)."""
    answered = [m for m in state.get("history", []) if not m.get("stopped")]
    return [{"role": m["role"], "content": m["content"]} for m in answered[-turns:]]


# --------------------------------------------------------------------------- browser side
# Installed once into the Streamlit page (not the component's iframe, which is replaced on
# every rerun), so the key handler keeps working. Setting a React textarea needs the native
# value setter plus an "input" event, or React ignores the change.
_INSTALL = r"""
(function () {
  if (window.__ragChat) return;
  const SELECTOR = '[data-testid="stChatInput"] textarea, textarea[data-testid="stChatInputTextArea"]';
  const chat = { history: [], index: null, draft: "", setting: false, selector: SELECTOR };
  window.__ragChat = chat;
  chat.set = function (box, text) {
    const proto = window.HTMLTextAreaElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
    chat.setting = true;
    setter.call(box, text);
    box.dispatchEvent(new Event("input", { bubbles: true }));
    chat.setting = false;
    box.selectionStart = box.selectionEnd = text.length;
  };
  document.addEventListener("keydown", function (event) {
    const box = event.target;
    if (!(box instanceof HTMLTextAreaElement) || !box.matches(SELECTOR)) return;
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    if (event.shiftKey || event.altKey || event.ctrlKey || event.metaKey || event.isComposing) return;
    if (!chat.history.length) return;
    const before = box.value.slice(0, box.selectionStart);
    const after = box.value.slice(box.selectionEnd);
    if (event.key === "ArrowUp") {
      if (before.includes("\n")) return; // inside a multi-line message: move the cursor first
      if (chat.index === null) { chat.draft = box.value; chat.index = chat.history.length; }
      if (chat.index === 0) { event.preventDefault(); return; } // oldest question reached
      chat.index -= 1;
    } else {
      if (chat.index === null || after.includes("\n")) return;
      chat.index += 1;
    }
    event.preventDefault();
    if (chat.index >= chat.history.length) { chat.set(box, chat.draft); chat.index = null; }
    else chat.set(box, chat.history[chat.index]);
  }, true);
  // ---- stop square over the send arrow while an answer is being computed
  const style = document.createElement("style");
  style.textContent = '[class*="st-key-stop-"] { display: none !important; }' +
    "#rag-stop { position: fixed; z-index: 1000; display: none;" +
    " align-items: center; justify-content: center;" +
    " border: none; border-radius: 0.5rem; cursor: pointer; background: rgb(255, 75, 75); padding: 0; }" +
    "#rag-stop span { width: 0.7rem; height: 0.7rem; background: white; border-radius: 2px; }";
  document.head.appendChild(style);
  const stop = document.createElement("button");
  stop.id = "rag-stop";
  stop.title = "Stop: the question goes back into the box";
  stop.setAttribute("aria-label", "Stop");
  stop.appendChild(document.createElement("span"));
  stop.addEventListener("click", function () {
    const trigger = document.querySelector('[class*="st-key-stop-"] button');
    if (trigger) trigger.click();
  });
  document.body.appendChild(stop);
  function place() {
    const send = document.querySelector('[data-testid="stChatInputSubmitButton"]');
    if (!send) return;
    const r = send.getBoundingClientRect();
    stop.style.left = r.left + "px";
    stop.style.top = r.top + "px";
    stop.style.width = r.width + "px";
    stop.style.height = r.height + "px";
  }
  chat.setBusy = function (busy) {
    stop.style.display = busy ? "flex" : "none";
    clearInterval(chat.follow);
    if (busy) { place(); chat.follow = setInterval(place, 150); } // follow resizes and scrolling
  };
  document.addEventListener("input", function (event) {
    if (!chat.setting && event.target instanceof HTMLTextAreaElement && event.target.matches(SELECTOR)) {
      chat.index = null; // typing ends the walk through the history
    }
  }, true);
})();
"""

_UPDATE = """
<script>
(function () {
  const page = window.parent;
  if (!page.__ragChat) {
    const script = page.document.createElement("script");
    script.textContent = __INSTALL__;
    page.document.head.appendChild(script);
  }
  const chat = page.__ragChat;
  if (!chat) return;
  chat.setBusy(__BUSY__);
  chat.history = __HISTORY__;
  chat.index = null;
  const prefill = __PREFILL__;
  if (!prefill || chat.prefilled === prefill.id) return;
  (function fill(tries) {
    const box = page.document.querySelector(chat.selector);
    if (box) { chat.set(box, prefill.text); box.focus(); chat.prefilled = prefill.id; }
    else if (tries > 0) setTimeout(function () { fill(tries - 1); }, 100);
  })(30);
})();
</script>
"""


def _js(value) -> str:
    """JSON for inside a <script> tag: "</script>" in a question must not end it."""
    return json.dumps(value).replace("</", "<\\/")


def browser_script(asked: list[str], prefill: dict | None, busy: bool = False) -> str:
    """HTML for a 1 px component: arrow-key history, the one-time Stop prefill, and the send
    arrow shown as a stop square while ``busy``."""
    values = {
        "__INSTALL__": _js(_INSTALL),
        "__HISTORY__": _js(list(asked)),
        "__PREFILL__": _js(prefill),
        "__BUSY__": _js(bool(busy)),
    }
    # one pass, so a question that happens to contain "__PREFILL__" is never replaced itself
    return re.sub("__INSTALL__|__HISTORY__|__PREFILL__|__BUSY__", lambda m: values[m.group(0)], _UPDATE)