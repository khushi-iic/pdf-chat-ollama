import os
import streamlit as st
import chromadb
import ollama
from pypdf import PdfReader


# ---------- Ollama client (cloud if key is set, else local) ----------
def get_api_key():
    try:
        return st.secrets["OLLAMA_API_KEY"]
    except Exception:
        return os.environ.get("OLLAMA_API_KEY", "")


API_KEY = get_api_key()
if API_KEY:
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + API_KEY},
    )
else:
    client = ollama.Client()


# ---------- Helpers ----------
def get_chat_models():
    try:
        names = [m.model for m in client.list().models]
    except Exception:
        names = []
    if API_KEY and not names:
        names = ["gpt-oss:120b"]
    return [n for n in names if "embed" not in n.lower()]


def extract_pages(file):
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages, size=800, overlap=150):
    chunks = []
    for page_num, text in pages:
        text = text.strip()
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            start += size - overlap
    return chunks


def build_index(chunks, progress):
    db = chromadb.Client()  # in-memory, uses ChromaDB's built-in embeddings
    try:
        db.delete_collection("pdf")
    except Exception:
        pass
    col = db.create_collection("pdf")

    batch_size = 32
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        col.add(
            ids=[str(i + j) for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
        )
        progress.progress(min((i + batch_size) / total, 1.0))
    return col


def retrieve(question, collection, k=4):
    results = collection.query(query_texts=[question], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [{"text": d, "page": m["page"]} for d, m in zip(docs, metas)]


def answer_stream(question, sources, history, model):
    context = "\n\n---\n\n".join(
        f"[Page {s['page']}]\n{s['text']}" for s in sources
    )
    system = (
        "You are a helpful assistant that answers questions about a PDF document. "
        "Use ONLY the provided context. If the answer is not in the context, "
        "say: \"I couldn't find that in the document.\""
    )
    messages = [{"role": "system", "content": system}]
    messages += history[-6:]
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    stream = client.chat(model=model, messages=messages, stream=True)
    for part in stream:
        yield part["message"]["content"]


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**Page {s['page']}**")
            st.caption(s["text"])


# ---------- UI ----------
st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")
st.sidebar.caption("Mode: " + ("Ollama Cloud" if API_KEY else "Local Ollama"))

models = get_chat_models()
if not models:
    st.error(
        "No chat models found. For local mode, start Ollama and run "
        "`ollama pull llama3.2`. For cloud mode, check your OLLAMA_API_KEY."
    )
    st.stop()

chat_model = st.sidebar.selectbox("Chat model", models)
top_k = st.sidebar.slider("Chunks to retrieve", 2, 8, 4)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.session_state.get("file_name") != uploaded.name:
    pages = extract_pages(uploaded)
    chunks = chunk_pages(pages)

    if not chunks:
        st.error(
            "No text found in this PDF. It may be a scanned/image PDF, "
            "which needs OCR. Try a PDF with selectable text."
        )
        st.stop()

    st.write("Indexing your PDF (the first time can take a minute)...")
    bar = st.progress(0.0)
    try:
        st.session_state.collection = build_index(chunks, bar)
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        st.stop()

    st.session_state.file_name = uploaded.name
    st.session_state.messages = []
    st.success(f"Indexed {len(chunks)} chunks from {len(pages)} pages. Ask away!")

if "collection" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("sources"):
                show_sources(m["sources"])

    if question := st.chat_input("Ask something about the PDF"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                sources = retrieve(question, st.session_state.collection, k=top_k)
                reply = st.write_stream(
                    answer_stream(question, sources, history, chat_model)
                )
                show_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "sources": sources}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")