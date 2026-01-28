import streamlit as st
import re

from langchain_community.document_loaders import PDFMinerLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Resume AI Agent", layout="centered")

st.title("🤖 Smart Resume AI Agent")
st.caption("Ask anything about your resume")


# ---------------- CLEAN TEXT ---------------- #

def clean_resume_text(text):

    if not text:
        return ""

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\u00a0', ' ', text)

    return text.strip()

# ---------------- HEADERS ---------------- #

SECTION_HEADERS = [
    "EDUCATION",
    "RESEARCH",
    "PROJECTS",
    "SKILLS",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "EXTRACURRICULAR"
]


# ---------------- HEADER SPLITTER ---------------- #

def split_by_headers(text):

    sections = []

    current_section = "GENERAL"
    buffer = ""

    for line in text.split("\n"):

        clean_line = line.strip().upper()

        if clean_line in SECTION_HEADERS:

            if buffer.strip():
                sections.append((current_section, buffer.strip()))

            current_section = clean_line
            buffer = ""

        else:
            buffer += line + "\n"

    if buffer.strip():
        sections.append((current_section, buffer.strip()))

    return sections


# ---------------- SMART CHUNKER ---------------- #

def smart_resume_chunker(sections, max_chars=350):

    chunks = []

    for section, content in sections:

        if not content.strip():
            continue

        lines = content.split("\n")
        buffer = ""

        for line in lines:

            if len(buffer) + len(line) <= max_chars:
                buffer += line + "\n"

            else:
                chunks.append(
                    Document(
                        page_content=buffer.strip(),
                        metadata={"section": section}
                    )
                )
                buffer = line + "\n"

        if buffer.strip():
            chunks.append(
                Document(
                    page_content=buffer.strip(),
                    metadata={"section": section}
                )
            )

    return chunks


# ---------------- LOAD RESUME ---------------- #

@st.cache_resource(show_spinner=True)
def load_resume():

    loader = PDFMinerLoader("resume.pdf")
    docs = loader.load()

    if not docs:
        st.error("❌ Resume PDF could not be loaded")
        st.stop()

    raw_text = "\n".join([doc.page_content for doc in docs])

    cleaned_text = clean_resume_text(raw_text)

    if len(cleaned_text) < 50:
        st.error("❌ Resume text extraction failed. PDF may be image based.")
        st.stop()

    sections = split_by_headers(cleaned_text)

    chunks = smart_resume_chunker(sections)

    if len(chunks) == 0:
        st.error("❌ No resume chunks created. Please check resume formatting.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

    retriever = vector_db.as_retriever(search_kwargs={"k": 6})

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256,
        temperature=0.0
    )

    return retriever, generator, chunks


retriever, generator, resume_chunks = load_resume()


# ---------------- RAG AGENT ---------------- #

def resume_agent(query):

    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "Not available in resume"

    context = "\n".join(
        [f"[{d.metadata['section']}]\n{d.page_content}" for d in docs]
    )

    prompt = f"""
You are a Resume AI Assistant.

Rules:
- Only answer from resume content
- If not present say "Not available in resume"
- Be short and accurate

Resume:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt)[0]["generated_text"]

    return result.strip()


# ---------------- CHAT UI ---------------- #

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question:")

if st.button("Ask"):

    if user_input.strip():

        with st.spinner("Processing..."):

            answer = resume_agent(user_input)

            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("AI", answer))


# ---------------- CHUNK VIEWER ---------------- #

st.subheader("📄 Resume Chunk Inspector")

with st.expander("View extracted chunks"):

    st.write(f"Total Chunks: {len(resume_chunks)}")

    for i, chunk in enumerate(resume_chunks):

        st.markdown(f"### Chunk {i+1} ({chunk.metadata['section']})")
        st.code(chunk.page_content)
        st.divider()


# ---------------- CHAT HISTORY ---------------- #

st.subheader("💬 Chat History")

for role, msg in st.session_state.history:

    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")
