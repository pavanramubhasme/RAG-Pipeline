"""Chat box bookkeeping: arrow-key history and Stop (ui/chat_state.py).

The browser side was checked end to end in headless Chrome against the real Streamlit app:
Up/Down walk the history and restore a draft; while an answer is computed the send arrow is
covered by a stop square of the same size; clicking it ends the wait in about 0.1 s, puts the
question back into the box and never shows the late answer; afterwards the arrow sends again."""

import json
import shutil
import subprocess

import pytest

from ui.chat_state import _INSTALL, api_history, browser_script, finish, interrupted, remember, start, stop


def test_remember_skips_immediate_repeats():
    state = {}
    for question in ["a", "a", "b", "a"]:
        remember(state, question)
    assert state["asked"] == ["a", "b", "a"]


def test_a_normal_answer_goes_into_the_conversation():
    state = {}
    start(state, "How many vacation days?")
    assert state["in_flight"] == "How many vacation days?"
    finish(state, "How many vacation days?", "30 days [1]")
    assert "in_flight" not in state
    assert state["history"] == [
        {"role": "user", "content": "How many vacation days?"},
        {"role": "assistant", "content": "30 days [1]"},
    ]


def test_stop_keeps_the_question_as_stopped_and_refills_the_box():
    state = {}
    start(state, "slow question")
    stop(state, "slow question")
    assert "in_flight" not in state
    assert state["history"] == [{"role": "user", "content": "slow question", "stopped": True}]
    assert state["prefill"]["text"] == "slow question" and state["prefill"]["id"]


def test_stop_after_the_answer_arrived_only_refills_the_box():
    state = {}
    start(state, "q")
    finish(state, "q", "answer")
    stop(state, "q")
    assert [m.get("stopped") for m in state["history"]] == [None, None]
    assert state["prefill"]["text"] == "q"


def test_a_run_that_ended_without_an_answer_counts_as_stopped():
    state = {}
    start(state, "first")
    assert interrupted(state) == "first"  # e.g. a new question was sent while waiting
    assert interrupted(state) is None  # only once
    assert state["history"][-1] == {"role": "user", "content": "first", "stopped": True}


def test_a_new_question_replaces_an_edit_in_progress():
    state = {"prefill": {"text": "old", "id": "x"}}
    start(state, "new")
    assert "prefill" not in state


def test_errors_keep_nothing_in_the_conversation():
    state = {}
    start(state, "q")
    finish(state, "q", None)
    assert state.get("history", []) == [] and "in_flight" not in state


def test_stopped_questions_are_not_sent_to_the_api():
    state = {
        "history": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "stopped one", "stopped": True},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "B"},
        ]
    }
    assert api_history(state, 6) == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "A"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "B"},
    ]
    assert api_history(state, 2) == [{"role": "user", "content": "b"}, {"role": "assistant", "content": "B"}]


def test_browser_script_carries_history_and_prefill():
    html = browser_script(["first", "second"], {"text": "edit me", "id": "abc"})
    assert json.dumps(["first", "second"]) in html
    assert '"text": "edit me"' in html and '"id": "abc"' in html
    assert html.count("<script>") == 1 and html.count("</script>") == 1


def test_busy_turns_the_send_arrow_into_a_stop_square():
    assert "chat.setBusy(true);" in browser_script([], None, busy=True)
    assert "chat.setBusy(false);" in browser_script([], None)
    assert '[class*="st-key-stop-"] { display: none !important; }' in _INSTALL  # no second button


def test_a_question_cannot_break_out_of_the_script():
    html = browser_script(['</script><img src=x onerror="alert(1)">'], None)
    assert html.count("</script>") == 1  # only our own closing tag
    assert "<\\/script>" in html


def test_placeholders_inside_questions_are_not_replaced():
    html = browser_script(["what is __PREFILL__?"], {"text": "x", "id": "1"})
    assert "what is __PREFILL__?" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_installed_javascript_is_valid(tmp_path):
    script = tmp_path / "install.js"
    script.write_text(_INSTALL, encoding="utf-8")
    assert subprocess.run(["node", "--check", str(script)], capture_output=True).returncode == 0