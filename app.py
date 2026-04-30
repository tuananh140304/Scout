#app.py
import streamlit as st
from streamlit.elements.widgets.chat import ChatInputValue

from Scout_step_5 import ScoutAgent

company_logo = "images/the_keepsake_bow_v5.svg"
scout_icon = "images/scout_icon_keepsake_bg.svg"


st.set_page_config(
    page_title="Scout – Business Analyst",
    page_icon="🎁",
    layout="centered"
)

st.image(company_logo)

st.title("Scout — The Keepsake’s Business Analyst")

st.markdown(
    "Ask Scout questions about ecommerce, customers, products, or performance."
)

# Initialize agent in session state
if "agent" not in st.session_state:
    st.session_state.agent = ScoutAgent()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input: str | None | ChatInputValue = st.chat_input("Ask Scout a business question...")

if user_input:
    # Show user message
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Get Scout's response
    response = st.session_state.agent.ask(user_input)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response}
    )

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Reset button
st.divider()
if st.button("Reset conversation"):
    st.session_state.agent.reset()
    st.session_state.chat_history = []
    st.experimental_rerun()