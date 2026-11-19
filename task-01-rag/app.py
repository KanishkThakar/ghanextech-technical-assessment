"""Streamlit UI for the RAG document assistant.

The UI is deliberately thin: it owns session state and rendering, and delegates
every real step to `rag.pipeline.RagPipeline`.
"""

from __future__ import annotations

import streamlit as st

from rag.ingest import IngestError
from rag.llm import LLMError
from rag.pipeline import RagPipeline

st.set_page_config(page_title="Document Assistant (RAG)", page_icon="📄", layout="wide")


def get_pipeline() -> RagPipeline:
    """One pipeline per browser session, kept across Streamlit reruns."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RagPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role", "content", "sources"}]
    return st.session_state.pipeline


def render_sources(sources) -> None:
    """Show the retrieved passages under an answer, with citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, item in enumerate(sources, start=1):
            chunk = item.chunk
            st.markdown(
                f"**[{i}] {chunk.citation}** · chunk {chunk.chunk_index} · "
                f"similarity {item.score:.3f}"
            )
            st.caption(chunk.text[:600] + ("…" if len(chunk.text) > 600 else ""))


def main() -> None:
    pipeline = get_pipeline()

    st.title("Document Assistant")
    st.caption(
        "Upload PDFs, ask questions, get answers grounded in the documents with page-level citations."
    )

    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Load uploaded files", type="primary", use_container_width=True):
            if not uploads:
                st.info("Choose at least one PDF first.")
            else:
                for uploaded in uploads:
                    if uploaded.name in pipeline.loaded_documents:
                        continue
                    try:
                        added = pipeline.add_pdf(uploaded.getvalue(), uploaded.name)
                        st.success(f"{uploaded.name}: {added} chunks indexed.")
                    except IngestError as exc:
                        st.error(str(exc))

        if pipeline.loaded_documents:
            st.write("**Indexed:**")
            for name in pipeline.loaded_documents:
                st.write(f"- {name} ({pipeline.store.documents.get(name, 0)} chunks)")
            st.caption(f"{len(pipeline.store.chunks)} chunks in the index")
        else:
            st.caption("No documents loaded yet.")

        for warning in pipeline.warnings:
            st.warning(warning)

        st.divider()
        top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 4)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset everything", use_container_width=True):
            pipeline.reset()
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))

    question = st.chat_input("Ask something about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not pipeline.has_documents:
            st.warning("Upload and load at least one PDF first.")
            return
        with st.spinner("Retrieving and answering…"):
            try:
                answer = pipeline.ask(
                    question,
                    top_k=top_k,
                    history=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ],
                )
            except (IngestError, LLMError) as exc:
                st.error(str(exc))
                return
        st.markdown(answer.text)
        render_sources(answer.sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer.text, "sources": answer.sources}
    )


if __name__ == "__main__":
    main()
