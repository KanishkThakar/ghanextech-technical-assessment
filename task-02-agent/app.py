"""Streamlit UI for the tool-calling agent.

The page owns the chat transcript and the rendering; `agent.loop.run_turn` owns
the reasoning. Every tool call the model makes is shown under its answer, so the
tool use is visible rather than hidden behind the reply.
"""

from __future__ import annotations

import json

import streamlit as st

from agent.loop import MAX_STEPS, run_turn
from agent.registry import describe_tools

st.set_page_config(page_title="Tool-Calling Agent", page_icon="🛠️", layout="centered")

EXAMPLES = [
    "What's the weather in Mumbai for the next 3 days, and should I carry an umbrella?",
    "If I split a ₹4,850 dinner between 4 people and add 5% service charge, what does each pay?",
    "Who was Ada Lovelace, and how much is 250 USD in INR?",
    "What is 17.5% of 2,340, then convert that from INR to EUR?",
]

st.title("Tool-Calling Agent")
st.caption(
    "A multi-turn chat where the model decides which tools to call, can call several in one "
    "turn, and recovers from tool errors."
)

with st.sidebar:
    st.subheader("Tools")
    for tool in describe_tools():
        with st.expander(tool["name"]):
            st.write(tool["description"])
    st.caption(f"Up to {MAX_STEPS} tool steps per turn.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.traces = []
        st.rerun()

# --- session state -----------------------------------------------------------
# `messages` is the plain transcript sent back to the model (user/assistant only).
# `traces` holds the tool calls made for each assistant reply, for display.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []

# --- history -----------------------------------------------------------------
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and index < len(st.session_state.traces):
            steps = st.session_state.traces[index]
            if steps:
                with st.expander(f"Tool calls ({len(steps)})"):
                    for step in steps:
                        st.markdown(f"**{step['tool']}**")
                        st.code(json.dumps(step["arguments"], indent=2, default=str), language="json")
                        st.code(json.dumps(step["result"], indent=2, default=str), language="json")

# --- input -------------------------------------------------------------------
prompt = st.chat_input("Ask something that needs a tool…")
if not prompt:
    if not st.session_state.messages:
        st.write("Try one of these:")
        for example in EXAMPLES:
            st.caption(f"• {example}")
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

with st.chat_message("assistant"):
    with st.spinner("Thinking…"):
        turn = run_turn(st.session_state.messages[:-1], prompt)
    st.markdown(turn.answer)
    if turn.steps:
        with st.expander(f"Tool calls ({len(turn.steps)})", expanded=True):
            for step in turn.steps:
                st.markdown(f"**{step['tool']}**")
                st.code(json.dumps(step["arguments"], indent=2, default=str), language="json")
                st.code(json.dumps(step["result"], indent=2, default=str), language="json")

st.session_state.messages.append({"role": "assistant", "content": turn.answer})
st.session_state.traces.append(turn.steps)
