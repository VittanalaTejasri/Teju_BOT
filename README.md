# 🤖 Smart Resume AI Agent (RAG-Based Application)

## 📌 Overview

The **Smart Resume AI Agent** is a **Retrieval-Augmented Generation (RAG)** based Streamlit web application that allows users to ask natural language questions about their resume PDF and receive accurate answers strictly grounded in the resume content.

This project combines:

* **PDF text extraction**
* **Smart resume chunking**
* **Semantic vector search**
* **Transformer-based answer generation**

The system ensures that responses are factual, concise, and derived only from the uploaded resume.

---

## 🧠 What is RAG (Retrieval-Augmented Generation)?

**RAG (Retrieval-Augmented Generation)** is an AI architecture that enhances text generation models by combining them with an external knowledge retrieval system.

### Traditional LLM Problem

Large Language Models (LLMs) may:

* Hallucinate answers
* Provide outdated information
* Generate responses not grounded in user data

### RAG Solution

RAG solves this by adding a retrieval step before generation:

1. **Retrieve** relevant information from a knowledge source (resume)
2. **Augment** the prompt with retrieved context
3. **Generate** an answer using a language model

### In My Project

My application:

* Retrieves relevant resume chunks using **FAISS + embeddings**
* Feeds those chunks to **Flan-T5**
* Generates answers only from resume content

This makes your system:

* Accurate
* Explainable
* Resume-grounded
* Production-ready

---

## 🏗️ Project Architecture

```
User Question
     ↓
Streamlit UI
     ↓
Resume Retriever (FAISS)
     ↓
Relevant Resume Chunks
     ↓
Prompt Construction
     ↓
Flan-T5 Generator
     ↓
Final Answer
```

---

## 🔄 Complete Workflow Explanation

### Step 1 — Resume Upload & Loading

Your system uses **PDFMinerLoader** to extract text from resume PDFs.

### Package Used

```
langchain_community.document_loaders.PDFMinerLoader
```

### Why PDFMiner?

* Better text layout preservation
* Accurate extraction for resumes
* Handles multi-column PDFs

---

### Step 2 — Resume Text Cleaning

After extraction, raw text contains unwanted characters.

### Cleaning Operations

Your function:

* Removes extra new lines
* Removes unnecessary spaces
* Handles Unicode spacing issues

### Purpose

* Improve embedding quality
* Avoid chunk noise
* Better retrieval accuracy

---

### Step 3 — Section Detection (Header Based Splitting)

You manually define common resume section headers:

```
EDUCATION
PROJECTS
SKILLS
RESEARCH
CERTIFICATIONS
ACHIEVEMENTS
EXTRACURRICULAR
```

### Why Header-Based Splitting?

Instead of random chunking:

* Your system preserves resume structure
* Maintains semantic meaning
* Improves contextual retrieval

---

### Step 4 — Smart Resume Chunking

Your custom chunker:

* Splits resume by sections
* Breaks content into 350 character chunks
* Preserves section metadata

### Benefits

✔ Smaller chunks → Better embeddings
✔ Section metadata → Context awareness
✔ Faster retrieval

---

### Step 5 — Embedding Generation

You convert resume chunks into numerical vectors using:

### Model Used

```
BAAI/bge-large-en-v1.5
```

### Why This Model?

* State-of-the-art sentence embedding model
* High semantic similarity accuracy
* Strong performance in retrieval tasks

---

### Step 6 — Vector Database Creation

You store embeddings using:

### Tool Used

```
FAISS (Facebook AI Similarity Search)
```

### Purpose

* Fast similarity search
* Low memory usage
* Optimized nearest neighbor retrieval

---

### Step 7 — Retriever Setup

Retriever configuration:

```
k = 6
```

### What It Means

For every user query:

* Top 6 most relevant resume chunks are fetched
* These chunks form the knowledge context

---

### Step 8 — Prompt Engineering

You construct a controlled prompt:

### Prompt Rules

* Only answer from resume content
* If missing → return "Not available in resume"
* Keep responses short and accurate

### Why This Matters

This prevents:

❌ Hallucination
❌ Guessing
❌ External knowledge leakage

---

### Step 9 — Answer Generation

You use a Transformer model for generation.

### Model Used

```
google/flan-t5-large
```

### Why Flan-T5?

* Instruction-tuned model
* Strong at question answering
* Lightweight compared to GPT models
* Free to use locally

### Configuration

* Max tokens: 256
* Temperature: 0.0 (Deterministic output)

---

### Step 10 — Streamlit Frontend

Your UI is built using:

```
Streamlit
```

### UI Features

✔ Chat interface
✔ Resume chunk inspector
✔ Chat history
✔ Spinner loading indicator

---

## 🧩 Libraries Used

| Library                  | Purpose                     |
| ------------------------ | --------------------------- |
| Streamlit                | Web UI                      |
| LangChain Community      | Document loaders and schema |
| PDFMiner                 | PDF text extraction         |
| HuggingFace Transformers | Text generation model       |
| HuggingFace Embeddings   | Vector embedding creation   |
| FAISS                    | Vector database             |
| Regex (re)               | Resume cleaning             |

---

##

---

✅ Built by: Tejasri
