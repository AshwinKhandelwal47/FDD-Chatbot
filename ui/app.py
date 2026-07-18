import logging
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

import sys
# Add project root to sys.path to allow importing src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 1. Force find and load the .env file from the parent directory before project imports
ENV_PATH = project_root / '.env'
load_dotenv(dotenv_path=ENV_PATH)

# 2. Local module imports can now safely read the loaded environment variables
from src.chat.chatbot import ask_with_rag
from src.llm.factory import (
    default_provider_for_role,
    get_registered_model,
    list_provider_names,
)
from src.rag.vector_store import load_chroma_store
from src.rag.retriever import get_retriever

logging.basicConfig(level=logging.INFO)


def _init_provider_session_state() -> None:
    provider_names = list_provider_names()
    if "chat_provider" not in st.session_state:
        try:
            st.session_state.chat_provider = default_provider_for_role("chat")
        except ValueError:
            st.session_state.chat_provider = provider_names[0]
    if "extraction_provider" not in st.session_state:
        try:
            st.session_state.extraction_provider = default_provider_for_role("extraction")
        except ValueError:
            st.session_state.extraction_provider = provider_names[0]


def _render_provider_sidebar() -> None:
    provider_names = list_provider_names()

    with st.sidebar:
        st.header("LLM Providers")
        st.selectbox("Chat provider", provider_names, key="chat_provider")
        st.selectbox("Extraction provider", provider_names, key="extraction_provider")

        chat_model = get_registered_model(st.session_state.chat_provider)
        st.caption(f"Active chat: {st.session_state.chat_provider} · {chat_model}")


def _get_retriever():
    """Build a retriever from the shared vector store (cached per Streamlit session)."""
    store_path = Path(__file__).resolve().parent.parent / "data" / "vector_store"
    collection = load_chroma_store(store_path)
    return get_retriever(collection)


def main() -> None:
    _init_provider_session_state()
    _render_provider_sidebar()

    st.title("Financial Due Diligence Chatbot")
    st.write("Upload financial documents, run ingestion, and ask questions.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_files = st.file_uploader(
        "Upload Documents", type=["pdf", "xlsx", "xls", "csv"], accept_multiple_files=True
    )
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files.")

    if st.button("Start ingestion"):
        if not uploaded_files:
            st.warning("Please upload a file first.")
        else:
            from pathlib import Path
            from src.ingestion.pipeline import ingest_document

            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)

            for uploaded_file in uploaded_files:
                # Save the file to data/raw/
                save_path = raw_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Call the orchestrator
                try:
                    with st.spinner(f"Ingesting {uploaded_file.name}..."):
                        summary = ingest_document(str(save_path))
                    st.success(f"Successfully ingested {uploaded_file.name}")
                    st.json(summary)
                except Exception as e:
                    st.error(f"Error ingesting {uploaded_file.name}: {e}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    st.divider()
    st.subheader("Chat")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("provider_label"):
                st.caption(f"answered by: {message['provider_label']}")
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        extra = ""
                        if src.get("row_range"):
                            extra += f" · Table {src.get('table_index')} ({src.get('row_range')})"
                        st.markdown(
                            f"**[{src['index']}]** {src['source_file']} "
                            f"· Page: {src['page_number']} · Type: {src['chunk_type']}{extra}"
                        )

    prompt = st.chat_input("Ask a financial due diligence question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if m["role"] in ("user", "assistant")
        ]
        prior_history = history[:-1]

        retriever = _get_retriever()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, provider_label, sources_list = ask_with_rag(
                    prompt,
                    history=prior_history,
                    retriever=retriever,
                    chat_provider=st.session_state.chat_provider,
                )
            st.markdown(answer)
            st.caption(f"answered by: {provider_label}")
            if sources_list:
                with st.expander("Sources"):
                    for src in sources_list:
                        extra = ""
                        if src.get("row_range"):
                            extra += f" · Table {src.get('table_index')} ({src.get('row_range')})"
                        st.markdown(
                            f"**[{src['index']}]** {src['source_file']} "
                            f"· Page: {src['page_number']} · Type: {src['chunk_type']}{extra}"
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "provider_label": provider_label,
                "sources": sources_list,
            }
        )


if __name__ == "__main__":
    main()