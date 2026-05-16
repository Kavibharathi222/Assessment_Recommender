from __future__ import annotations

import asyncio
import streamlit as st

from app.catalog import load_catalog
from app.gemini_client import generate_answer
from app.recommender import (
    apply_user_edits,
    is_confirmation,
    recommend,
    should_ask_clarifying_question,
)


st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --panel: #141820;
        --panel-soft: #1b202b;
        --line: #2c3442;
        --text-muted: #aab3c2;
        --accent: #ff4b5c;
        --accent-2: #2dd4bf;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    h1 {
        font-size: 2.7rem !important;
        letter-spacing: 0 !important;
        margin-bottom: .25rem !important;
    }

    .hero-subtitle {
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: 1rem 0 1.25rem;
    }

    .status-card {
        background: linear-gradient(180deg, var(--panel-soft), var(--panel));
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .9rem 1rem;
    }

    .status-card span {
        display: block;
        color: var(--text-muted);
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .04em;
    }

    .status-card strong {
        display: block;
        margin-top: .25rem;
        font-size: 1.15rem;
    }

    .example-row {
        display: flex;
        gap: .55rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }

    .example-chip {
        border: 1px solid var(--line);
        background: #11151d;
        color: #d9dee8;
        border-radius: 999px;
        padding: .45rem .7rem;
        font-size: .86rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.025);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 8px;
        padding: .8rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
    }

    .api-box {
        border-left: 3px solid var(--accent-2);
        background: rgba(45, 212, 191, .08);
        padding: .75rem .9rem;
        border-radius: 6px;
        color: #d9fff8;
        font-size: .9rem;
    }

    @media (max-width: 760px) {
        .status-grid {
            grid-template-columns: 1fr;
        }
        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading SHL catalog...")
def bootstrap():
    return load_catalog()


products = bootstrap()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

st.title("SHL Assessment Recommender")
st.markdown(
    '<div class="hero-subtitle">Find relevant SHL assessments from role, skills, seniority, and constraints.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-grid">
        <div class="status-card"><span>Catalog</span><strong>{len(products)} products</strong></div>
        <div class="status-card"><span>Retrieval</span><strong>ChromaDB + reranker</strong></div>
        <div class="status-card"><span>API</span><strong>/health and /chat</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="example-row">
        <div class="example-chip">Junior AI Developer</div>
        <div class="example-chip">Senior Java backend engineer</div>
        <div class="example-chip">Graduate management trainee</div>
        <div class="example-chip">Sales talent audit</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Project Status")
    st.metric("Catalog Products", len(products))
    st.caption("RAG-style recommender using the SHL product catalog.")
    st.markdown(
        """
        <div class="api-box">
            API server:<br>
            <code>uvicorn api:app --reload --port 8000</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.subheader("Try prompts")
    st.caption("Ask for a role, level, skills, duration, language, or assessment type.")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.recommendations = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Describe the role, skills, seniority, and constraints...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    edited = apply_user_edits(user_query, st.session_state.recommendations)
    turn_count = len(st.session_state.messages) // 2
    end_of_conversation = is_confirmation(user_query) or turn_count >= 8

    if edited != st.session_state.recommendations:
        recommendations = edited
    elif should_ask_clarifying_question(user_query):
        recommendations = None
    else:
        recommendations = recommend(user_query, products)

    if recommendations is not None:
        st.session_state.recommendations = recommendations

    answer = asyncio.run(generate_answer(user_query, recommendations, end_of_conversation))
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        if recommendations:
            st.dataframe(
                [
                    {
                        "#": index + 1,
                        "Name": item["name"],
                        "Test Type": item["test_type"],
                        "Keys": ", ".join(item["keys"]),
                        "Duration": item["duration"],
                        "Languages": ", ".join(item["languages"][:4])
                        + (" +" if len(item["languages"]) > 4 else ""),
                        "Remote": item["remote"],
                        "Adaptive": item["adaptive"],
                        "URL": item["url"],
                    }
                    for index, item in enumerate(recommendations)
                ],
                hide_index=True,
                use_container_width=True,
            )
