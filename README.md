# Build Your Own RAG System from Scratch
## AWS Well-Architected Assistant — Complete Step-by-Step Guide

> **YouTube Tutorial by Rahul Chaubey**
> Senior Cloud Architect | 20+ Years IT Experience | Ex-AWS | Ex-Microsoft | Ex-Oracle
> Director of Cloud Strategy | Cloud & Agentic AI Transformation Leader

---

##  About This Tutorial

This guide teaches you how to build a **production-grade RAG (Retrieval-Augmented Generation) system** from scratch using:
- **LangChain** — the RAG orchestration framework
- **OpenAI** — for embeddings and answer generation
- **FAISS** — for vector storage and similarity search
- **AWS Well-Architected Framework** — as the knowledge domain

By the end of this guide, you will have a working AI assistant that reads AWS documentation and answers your architecture questions — citing the exact source page for every answer.

**GitHub Repository:** https://github.com/chrahul/aws-wellarchitected-rag

---

##  Table of Contents

- [What Is RAG and Why Does It Matter](#what-is-rag-and-why-does-it-matter)
- [Phase 0 — Prerequisites](#phase-0--prerequisites)
- [Phase 1 — Install Python](#phase-1--install-python)
- [Phase 2 — Get Your OpenAI API Key](#phase-2--get-your-openai-api-key)
- [Phase 3 — Set Up the Project](#phase-3--set-up-the-project)
- [Phase 4 — Download the AWS Documents](#phase-4--download-the-aws-documents)
- [Phase 5 — Write the Ingestion Pipeline](#phase-5--write-the-ingestion-pipeline)
- [Phase 6 — Write the RAG Chatbot](#phase-6--write-the-rag-chatbot)
- [Phase 7 — Run the System](#phase-7--run-the-system)
- [Phase 8 — Push to GitHub](#phase-8--push-to-github)
- [Architecture Explained](#architecture-explained)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)
- [What to Build Next](#what-to-build-next)

---

##  What Is RAG and Why Does It Matter

### The Problem with Plain LLMs

When you ask GPT-4 a question, it answers from its **training memory** — data it learned during training, which has a cutoff date and does not include your private or domain-specific documents.

This creates three problems:
1. **Stale knowledge** — the model does not know about updates after its training cutoff
2. **No domain knowledge** — it has never read your company docs, AWS whitepapers, or internal policies
3. **Hallucination** — when uncertain, it confidently makes up facts

### The RAG Solution

RAG gives the LLM a **search engine over your documents**. Before answering, the system:
1. Converts your question into a vector (a list of numbers representing meaning)
2. Searches your document collection for the most relevant passages
3. Hands those passages to the LLM as context
4. The LLM answers **only from those passages** — not from training memory

**One sentence:** RAG = retrieve relevant evidence first, then generate an answer grounded in that evidence.

### Why AWS Well-Architected Framework?

- 5 public whitepapers covering all cloud architecture best practices
- Perfect for follow-up questions: "How do I design for HA?" → "What about multi-AZ?"
- You can verify every answer against the source PDF
- Directly applicable to real cloud architecture work

---

## Phase 0 — Prerequisites

### What You Need Before Starting

| Requirement | Details |
|-------------|---------|
| A laptop or desktop | Windows, Mac, or Linux |
| Internet connection | For downloading Python, packages, and calling OpenAI API |
| OpenAI account | Free to create at platform.openai.com |
| OpenAI API key | You will need credit (~$1 is enough for this project) |
| Git installed | For version control and pushing to GitHub |
| GitHub account | Free at github.com |
| Text editor | VS Code recommended (free at code.visualstudio.com) |

### What You Do NOT Need

-  A GPU or special hardware — this runs on any laptop
-  Docker or Kubernetes
-  A server or cloud VM
-  Prior AI/ML experience (this guide explains everything)

### Knowledge Assumptions

This guide assumes you know:
- How to open a terminal / command prompt
- Basic file system navigation (cd, ls/dir)
- What a Python file is
- How to copy and paste code

---

## Phase 1 — Install Python

Python is the programming language this project runs on. You need Python 3.10 or newer.

### Step 1.1 — Check if Python is already installed

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
python --version
```

If you see `Python 3.10.x` or newer — skip to Phase 2.

If you see an error or `Python 2.x` — follow the steps below.

### Step 1.2 — Download Python

Go to: **https://www.python.org/downloads/**

Click the yellow **Download Python 3.11.x** button (3.11 is recommended for stability with LangChain).

### Step 1.3 — Install Python (Windows)

1. Run the downloaded installer
2. **CRITICAL:** Check the box that says **"Add Python to PATH"** before clicking Install
3. Click **Install Now**
4. Wait for installation to complete
5. Click **Close**

### Step 1.4 — Install Python (Mac)

```bash
# If you have Homebrew installed:
brew install python@3.11

# Or use the installer from python.org (same as Windows steps above)
```

### Step 1.5 — Verify Python is installed

Close your terminal and open a new one. Run:

```bash
python --version
```

You should see: `Python 3.11.x`

Also verify pip (Python package manager) is installed:

```bash
pip --version
```

You should see something like: `pip 23.x from .../site-packages/pip (python 3.11)`

### Step 1.6 — Install Git (if not installed)

Check if Git is installed:

```bash
git --version
```

If not installed, download from: **https://git-scm.com/downloads**

During Windows install, select **"Git Bash"** — this gives you a Unix-style terminal on Windows.

---

## Phase 2 — Get Your OpenAI API Key

This project calls the OpenAI API for:
1. **Embeddings** — converting text to vectors (`text-embedding-3-small`)
2. **Chat completion** — generating answers (`gpt-4o-mini`)

**Estimated cost for this project:** Less than $1 USD

### Step 2.1 — Create an OpenAI account

Go to: **https://platform.openai.com**

Click **Sign Up** and create a free account.

### Step 2.2 — Add billing credits

Go to: **platform.openai.com/account/billing**

Add $5 USD — this is more than enough for this entire project and many more runs.

### Step 2.3 — Create an API key

1. Go to: **platform.openai.com/api-keys**
2. Click **Create new secret key**
3. Give it a name: `rag-assistant`
4. Click **Create secret key**
5. **Copy the key immediately** — you cannot see it again after closing the dialog
6. Save it somewhere safe (password manager or secure note)

Your key looks like: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx`

###  Security Warning

Never share your API key publicly. Never paste it directly in code files that go to GitHub. We will use a `.env` file to keep it secure.

---

## Phase 3 — Set Up the Project

### Step 3.1 — Create a GitHub repository

1. Go to **github.com** and log in
2. Click the **+** icon → **New repository**
3. Fill in:
   - Repository name: `aws-wellarchitected-rag`
   - Description: `AWS Well-Architected RAG Assistant — Build Your Own RAG System`
   - Visibility: **Public**
   - Check: **Add a README file**
   - Add .gitignore: **Python**
4. Click **Create repository**
5. Copy the repository URL (HTTPS or SSH)

### Step 3.2 — Clone the repository locally

Open Git Bash (Windows) or Terminal (Mac/Linux):

```bash
# Navigate to where you want the project
cd ~/Documents

# Clone your repo (replace YOUR_USERNAME with your GitHub username)
git clone https://github.com/YOUR_USERNAME/aws-wellarchitected-rag.git

# Enter the project directory
cd aws-wellarchitected-rag
```

### Step 3.3 — Create the folder structure

```bash
# Create required directories
mkdir documents faiss_index

# Create required files
touch ingest.py chatbot.py requirements.txt .env

# Verify the structure
ls -la
```

You should see:
```
documents/
faiss_index/
ingest.py
chatbot.py
requirements.txt
.env
README.md
.gitignore
```

### Step 3.4 — Update .gitignore

Add these entries to your `.gitignore` file to prevent secrets and generated files from being pushed to GitHub:

```bash
cat >> .gitignore << 'EOF'
.env
faiss_index/
__pycache__/
*.pyc
.venv/
EOF
```

### Step 3.5 — Create a virtual environment

A virtual environment keeps your project dependencies isolated from your system Python installation.

```bash
# Create the virtual environment
python -m venv .venv

# Activate it (Windows Git Bash)
source .venv/Scripts/activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

Your terminal prompt will now show `(.venv)` at the start — this confirms the virtual environment is active.

**Important:** Always activate the virtual environment before running any project commands.

### Step 3.6 — Install required packages

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu pypdf python-dotenv openai
```

This will take 2–5 minutes depending on your internet speed.

**What each package does:**

| Package | Purpose |
|---------|---------|
| `langchain` | Core RAG orchestration framework |
| `langchain-openai` | LangChain integration with OpenAI API |
| `langchain-community` | Community loaders including PyPDFDirectoryLoader |
| `langchain-text-splitters` | Document chunking utilities |
| `faiss-cpu` | Facebook AI Similarity Search — local vector store |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Load API keys from .env file securely |
| `openai` | OpenAI Python SDK |

### Step 3.7 — Save your dependencies

```bash
pip freeze > requirements.txt
```

This records all installed packages so anyone can recreate your environment.

### Step 3.8 — Set your API key

```bash
# Add your actual OpenAI API key to .env
echo "OPENAI_API_KEY=your-actual-key-here" > .env
```

Replace `your-actual-key-here` with the key you copied in Phase 2.

Also create a template file for other users:

```bash
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env.example
```

---

## Phase 4 — Download the AWS Documents

These are the 5 official AWS Well-Architected Framework whitepapers. All are publicly available and free to download.

### Step 4.1 — Navigate to the documents folder

```bash
cd documents
```

### Step 4.2 — Download all 5 PDFs

Run these commands one by one:

```bash
curl -L -o operational-excellence.pdf "https://docs.aws.amazon.com/wellarchitected/latest/operational-excellence-pillar/wellarchitected-operational-excellence-pillar.pdf"
```

```bash
curl -L -o security.pdf "https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/wellarchitected-security-pillar.pdf"
```

```bash
curl -L -o reliability.pdf "https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/wellarchitected-reliability-pillar.pdf"
```

```bash
curl -L -o performance-efficiency.pdf "https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/wellarchitected-performance-efficiency-pillar.pdf"
```

```bash
curl -L -o cost-optimization.pdf "https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/wellarchitected-cost-optimization-pillar.pdf"
```

### Step 4.3 — Verify the downloads

```bash
ls -lh
```

Expected output:
```
-rw-r--r-- 1 user 197620 1.9M cost-optimization.pdf
-rw-r--r-- 1 user 197620 2.7M operational-excellence.pdf
-rw-r--r-- 1 user 197620 1.4M performance-efficiency.pdf
-rw-r--r-- 1 user 197620 6.8M reliability.pdf
-rw-r--r-- 1 user 197620 2.8M security.pdf
```

Total: 5 PDFs, approximately 16MB.

### Step 4.4 — Go back to project root

```bash
cd ..
```

---

## Phase 5 — Write the Ingestion Pipeline

The ingestion pipeline reads your PDFs, splits them into chunks, converts each chunk to a vector (embedding), and saves everything to disk. This runs **once** — not on every query.

### Understanding the Pipeline

```
5 PDF Files
    ↓  PyPDFDirectoryLoader
1,019 pages of raw text
    ↓  RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
2,749 text chunks
    ↓  OpenAIEmbeddings (text-embedding-3-small)
2,749 vectors (1,536 dimensions each)
    ↓  FAISS.save_local()
faiss_index/ saved to disk
```

### Step 5.1 — Write ingest.py

Open `ingest.py` in VS Code and paste this complete code:

```python
# ingest.py
# ============================================================
# Pipeline 1: Document Ingestion
# Load → Chunk → Embed → Save
# Run this ONCE to build the FAISS index from your documents
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Step 0: Load API key from .env file ───────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

print("=" * 60)
print("AWS Well-Architected RAG — Document Ingestion Pipeline")
print("=" * 60)

# ── Step 1: Load all PDFs from documents/ folder ──────────────
print("\n[1/4] Loading PDF documents...")
loader = PyPDFDirectoryLoader("./documents/")
documents = loader.load()
num_docs = len(set(d.metadata['source'] for d in documents))
print(f"      Loaded {len(documents)} pages from {num_docs} documents")

# ── Step 2: Split into chunks ──────────────────────────────────
# chunk_size=1000: each chunk is at most 1000 characters
# chunk_overlap=200: 200 characters shared between adjacent chunks
# This overlap prevents losing answers that span chunk boundaries
print("\n[2/4] Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
avg_size = sum(len(c.page_content) for c in chunks) // len(chunks)
print(f"      Created {len(chunks)} chunks")
print(f"      Average chunk size: {avg_size} characters")

# ── Step 3: Generate embeddings using OpenAI ───────────────────
# text-embedding-3-small: 1536-dimensional vectors, cost-efficient
# This step calls the OpenAI API — takes 1-2 minutes
print("\n[3/4] Generating embeddings (calling OpenAI API)...")
print("      This may take 1-2 minutes for large document sets...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"      Embeddings generated for {len(chunks)} chunks")

# ── Step 4: Save FAISS index to disk ──────────────────────────
# Creates two files:
# - faiss_index/index.faiss: binary vector data (for fast search)
# - faiss_index/index.pkl: original text + metadata (for retrieval)
print("\n[4/4] Saving FAISS index to disk...")
vectorstore.save_local("./faiss_index")
print("      Saved: faiss_index/index.faiss")
print("      Saved: faiss_index/index.pkl")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ingestion complete!")
print(f"  Documents : {num_docs}")
print(f"  Pages     : {len(documents)}")
print(f"  Chunks    : {len(chunks)}")
print(f"  Avg chunk : {avg_size} characters")
print("  Index     : ./faiss_index/")
print("=" * 60)
print("\nNext step: python chatbot.py")
```

### Step 5.2 — Understanding the key concepts

**What is chunking?**
You cannot embed an entire 300-page PDF as one vector — the embedding loses all specificity. Instead, you split the document into small overlapping pieces (chunks). Each chunk gets its own embedding vector. When a user asks a question, you find the chunks most similar to their question and retrieve just those pieces.

**What is chunk_overlap?**
If a chunk ends mid-sentence and the answer continues in the next chunk, you would miss it without overlap. By repeating 200 characters at the start of each new chunk, you ensure answers that span boundaries are captured.

**What is an embedding?**
An embedding converts text into a list of numbers (a vector) that captures the meaning of the text. Similar meanings produce similar vectors. This allows mathematical similarity search — finding the most relevant chunks for any question.

**Why FAISS?**
FAISS (Facebook AI Similarity Search) is an open-source library that stores vectors and searches them extremely fast using approximate nearest neighbor algorithms. For 2,749 vectors it returns results in milliseconds. It saves to two files on disk and loads in under one second.

---

## Phase 6 — Write the RAG Chatbot

The chatbot loads the pre-built FAISS index, retrieves relevant chunks for each question, and generates grounded answers using GPT-4o-mini.

### Understanding the Pipeline

```
User question
    ↓  OpenAIEmbeddings (same model as ingestion)
Query vector (1,536 dimensions)
    ↓  FAISS similarity search (top-4 chunks)
4 most relevant text passages
    ↓  System prompt (grounding + refusal instructions)
Full prompt: system + context + history + question
    ↓  ChatOpenAI (gpt-4o-mini, temperature=0)
Grounded answer with source citations
```

### Step 6.1 — Write chatbot.py

Open `chatbot.py` in VS Code and paste this complete code:

```python
# chatbot.py
# ============================================================
# Pipeline 2: RAG Chatbot
# Load → Retrieve → Generate
# Runs on every user query using the pre-built FAISS index
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# ── Load API key ───────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

# ── System prompt ──────────────────────────────────────────────
# This is the most important part of any RAG system.
# It enforces two critical behaviours:
# 1. GROUNDING: Answer only from retrieved context
# 2. REFUSAL: Say "I don't have enough information" when context is insufficient
SYSTEM_PROMPT = """You are an expert assistant for the AWS Well-Architected Framework.
You help cloud architects, DevOps engineers, and developers understand AWS best practices
across the five pillars: Operational Excellence, Security, Reliability,
Performance Efficiency, and Cost Optimization.

STRICT RULES:
1. Answer ONLY using the context provided below from the AWS Well-Architected documents.
2. If the context does not contain enough information to answer the question,
   respond exactly with:
   "I don't have enough information in the provided documents."
3. Never use your general knowledge or make up facts.
4. Always cite which pillar or document your answer comes from when possible.
   Example: "According to the Security Pillar..." or "The Reliability Pillar states..."
5. Be concise, professional, and technically accurate.

Context from AWS Well-Architected documents:
{context}"""

# ── Load FAISS index from disk ─────────────────────────────────
def load_vectorstore():
    """Load the pre-built FAISS index from disk."""
    if not os.path.exists("./faiss_index"):
        raise FileNotFoundError(
            "faiss_index/ not found. Please run: python ingest.py"
        )
    print("Loading FAISS index from disk...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    vectorstore = FAISS.load_local(
        "./faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Index loaded successfully.\n")
    return vectorstore

# ── Format source citations ────────────────────────────────────
def format_sources(docs):
    """Extract and deduplicate source citations from retrieved chunks."""
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        filename = os.path.basename(source)
        key = f"{filename}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"  - {filename} (page {page})")
    return "\n".join(sources)

# ── Main chat loop ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AWS Well-Architected RAG Assistant")
    print("Powered by LangChain + OpenAI + FAISS")
    print("=" * 60)
    print("Commands: 'quit' = exit | 'clear' = reset history")
    print("-" * 60)

    # Load the vector store (once at startup)
    vectorstore = load_vectorstore()

    # Create retriever: finds top-4 most similar chunks per query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Create the LLM
    # temperature=0: deterministic output — critical for factual RAG
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    # In-memory conversation history for follow-up questions
    # Stores last 3 turns (6 messages) to keep prompt length manageable
    chat_history = []

    print("Ready. Ask your first question:\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_history = []
            print("Conversation history cleared.\n")
            continue

        try:
            # Step 1: Retrieve relevant chunks
            docs = retriever.invoke(user_input)
            context = "\n\n".join([d.page_content for d in docs])

            # Step 2: Build the prompt with context
            system_msg = SYSTEM_PROMPT.format(context=context)

            # Step 3: Include conversation history for follow-up handling
            # Last 3 turns = last 6 messages (user + assistant alternating)
            messages = [{"role": "system", "content": system_msg}]
            for h in chat_history[-6:]:
                messages.append({"role": "user", "content": h["question"]})
                messages.append({"role": "assistant", "content": h["answer"]})
            messages.append({"role": "user", "content": user_input})

            # Step 4: Generate the answer
            response = llm.invoke(messages)
            answer = response.content

            # Step 5: Store in history for next turn
            chat_history.append({"question": user_input, "answer": answer})

            # Step 6: Display answer and sources
            print(f"\nAssistant: {answer}")
            if docs:
                print(f"\nSources:")
                print(format_sources(docs))
            print("-" * 60)

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
```

### Step 6.2 — Understanding key concepts

**Why temperature=0?**
Temperature controls how creative/random the LLM's output is. Higher temperature = more creative but less accurate. For a RAG system that must stay grounded in source documents, temperature=0 produces the most faithful, deterministic responses.

**Why k=4 chunks?**
Retrieving 4 chunks gives the LLM enough context to answer most questions without overwhelming the prompt. Too few chunks = missing context. Too many chunks = the LLM's attention gets diluted.

**How does follow-up handling work?**
The chat history (last 3 turns) is prepended to every new message. So when you ask "What about multi-AZ?" after asking about high availability, the LLM sees both the previous question and answer as context, enabling it to understand what "this" and "that" refer to.

**Why refuse when context is missing?**
In a technical domain like AWS architecture, a confident wrong answer is worse than saying "I don't know." The refusal instruction in the system prompt forces the LLM to be honest about what the documents don't cover.

---

## Phase 7 — Run the System

### Step 7.1 — Activate your virtual environment

Every time you open a new terminal, activate the venv first:

```bash
# Windows Git Bash
source .venv/Scripts/activate

# Mac/Linux
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### Step 7.2 — Run the ingestion pipeline

This runs once. It processes all 5 PDFs and builds the FAISS index.

```bash
python ingest.py
```

Expected output:
```
============================================================
AWS Well-Architected RAG — Document Ingestion Pipeline
============================================================

[1/4] Loading PDF documents...
      Loaded 1019 pages from 5 documents
[2/4] Splitting into chunks...
      Created 2749 chunks
      Average chunk size: 820 characters
[3/4] Generating embeddings (calling OpenAI API)...
      This may take 1-2 minutes for large document sets...
      Embeddings generated for 2749 chunks
[4/4] Saving FAISS index to disk...
      Saved: faiss_index/index.faiss
      Saved: faiss_index/index.pkl

============================================================
Ingestion complete!
  Documents : 5
  Pages     : 1019
  Chunks    : 2749
  Avg chunk : 820 characters
  Index     : ./faiss_index/
============================================================

Next step: python chatbot.py
```

> **Note:** This step takes 1–2 minutes because it calls the OpenAI embeddings API for 2,749 chunks. You only run this once — or whenever you add new documents.

### Step 7.3 — Run the chatbot

```bash
python chatbot.py
```

Expected startup output:
```
============================================================
AWS Well-Architected RAG Assistant
Powered by LangChain + OpenAI + FAISS
============================================================
Commands: 'quit' = exit | 'clear' = reset history
------------------------------------------------------------
Loading FAISS index from disk...
Index loaded successfully.

Ready. Ask your first question:

You:
```

### Step 7.4 — Test with these 5 questions

**Test 1 — Direct question (Reliability Pillar):**
```
How should I design for high availability on AWS?
```
Expected: Answer citing reliability.pdf with page numbers

**Test 2 — Follow-up question (conversation history test):**
```
What about multi-AZ vs multi-Region? Which one should I choose?
```
Expected: Context-aware answer referencing the previous question

**Test 3 — Cost Optimization Pillar:**
```
How do I reduce costs for EC2 instances on AWS?
```
Expected: Answer citing cost-optimization.pdf

**Test 4 — Security Pillar:**
```
What are the best practices for IAM in AWS?
```
Expected: Answer citing security.pdf with specific IAM recommendations

**Test 5 — Out of scope (refusal test):**
```
What is the capital of France?
```
Expected:
```
I don't have enough information in the provided documents.
```

### Step 7.5 — Exit the chatbot

```
quit
```

---

## Phase 8 — Push to GitHub

### Step 8.1 — Stage all files

```bash
git add .
```

### Step 8.2 — Check what will be committed

```bash
git status
```

Make sure `.env` and `faiss_index/` are NOT in the list (they should be in .gitignore).

### Step 8.3 — Commit

```bash
git commit -m "Complete RAG pipeline — ingest.py + chatbot.py + 5 AWS PDFs"
```

### Step 8.4 — Push

```bash
git push origin main
```

### Step 8.5 — Verify on GitHub

Open your browser and go to your GitHub repository. You should see:
```
aws-wellarchitected-rag/
├── documents/          ← 5 AWS PDFs
├── chatbot.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Note: `faiss_index/` and `.env` should NOT be visible on GitHub.

---

## 🏗️ Architecture Explained

### The Two-Pipeline Design

This is the most important architectural decision in the entire project.

```
┌─────────────────────────────────────────────────────────┐
│  PIPELINE 1: INGESTION (ingest.py) — runs ONCE          │
│                                                          │
│  documents/ ──► PyPDFDirectoryLoader ──► 1,019 pages   │
│       ↓                                                  │
│  RecursiveCharacterTextSplitter ──► 2,749 chunks         │
│       ↓                                                  │
│  OpenAIEmbeddings ──► 2,749 vectors (1,536 dims each)   │
│       ↓                                                  │
│  FAISS.save_local() ──► faiss_index/ on disk             │
└─────────────────────────────────────────────────────────┘
                           │
                    [HANDOFF ARTIFACT]
                    faiss_index/ on disk
                           │
┌─────────────────────────────────────────────────────────┐
│  PIPELINE 2: RETRIEVAL + GENERATION (chatbot.py)         │
│                                                          │
│  User question                                           │
│       ↓                                                  │
│  FAISS.load_local() ──► index loaded in memory           │
│       ↓                                                  │
│  OpenAIEmbeddings ──► query vector                       │
│       ↓                                                  │
│  FAISS similarity search ──► top-4 relevant chunks       │
│       ↓                                                  │
│  System prompt + context + history + question            │
│       ↓                                                  │
│  ChatOpenAI (gpt-4o-mini, temperature=0)                 │
│       ↓                                                  │
│  Grounded answer + source citations                      │
└─────────────────────────────────────────────────────────┘
```

**Why separate pipelines?**
Embedding 2,749 chunks costs money and takes 1–2 minutes. You do this once. The chatbot loads the saved index in under a second on every startup and costs only the price of the chat completion per query.

---

##  Key Design Decisions

### 1. Why chunk_size=1000 with chunk_overlap=200?

At 1,000 characters per chunk, you get specific enough content for precise retrieval while maintaining enough context for the LLM to understand the passage. The 200-character overlap ensures that answers spanning two adjacent chunks are not lost.

### 2. Why text-embedding-3-small?

It is OpenAI's most cost-efficient embedding model with 1,536 dimensions — strong enough for semantic search over technical documentation at a fraction of the cost of larger models.

### 3. Why gpt-4o-mini at temperature=0?

gpt-4o-mini provides GPT-4 quality reasoning at significantly lower cost. Temperature=0 produces deterministic, factual responses that stay as close as possible to the retrieved context — essential for a grounded RAG system.

### 4. Why FAISS instead of a cloud vector database?

For fewer than 10 million vectors, FAISS running locally is faster, cheaper, and simpler than managed databases like Pinecone or Weaviate. No API key, no server, no cost. For this project with 2,749 vectors, FAISS returns results in under 1 millisecond.

### 5. Why the refusal instruction?

In technical domains, a confident wrong answer is more dangerous than saying "I don't know." If the AWS documents don't cover a topic, the system should be honest — not hallucinate a plausible-sounding but incorrect AWS best practice.

---

##  Troubleshooting

### "ModuleNotFoundError: No module found"

You are running Python outside the virtual environment. Activate it:
```bash
source .venv/Scripts/activate   # Windows
source .venv/bin/activate       # Mac/Linux
```

Then run again.

### "OPENAI_API_KEY not found"

Your `.env` file is missing or incorrectly formatted. Check:
```bash
cat .env
```

It should contain exactly:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

No quotes, no spaces around the `=`.

### "faiss_index/ not found"

You have not run the ingestion pipeline yet. Run:
```bash
python ingest.py
```

### "AuthenticationError: Incorrect API key"

Your OpenAI API key is wrong or expired. Check platform.openai.com/api-keys and create a new key.

### "RateLimitError"

You have exceeded your OpenAI API rate limit. Wait 60 seconds and try again. If persistent, add billing credits at platform.openai.com/account/billing.

### Ingestion is slow

This is normal — embedding 2,749 chunks via the OpenAI API takes 1–2 minutes. It only runs once. Future runs of the chatbot load the saved index instantly.

### Chatbot gives generic answers

The retrieval is working but your question may be too vague. Try more specific questions:
- Instead of: "Tell me about AWS"
- Try: "What are the best practices for S3 bucket security in the Well-Architected Framework?"

---

##  What to Build Next

This project is a foundation. Here are extensions you can build on top of it:

### 1. Add a Streamlit UI
Replace the command-line interface with a web UI:
```bash
pip install streamlit
```

### 2. Add more documents
Drop any PDF into the `documents/` folder and re-run `ingest.py`. The system immediately gains knowledge of those documents.

### 3. Add source filtering
Let users ask questions about a specific pillar:
```
"According to the Security Pillar only, what are the IAM best practices?"
```

### 4. Add evaluation
Measure retrieval quality — are the right chunks being retrieved? Tools like RAGAS provide automated RAG evaluation.

### 5. Deploy to the cloud
Package chatbot.py as a FastAPI endpoint and deploy on AWS Lambda, ECS, or EC2. Add authentication and rate limiting.

### 6. Connect to a real knowledge base
Replace the AWS PDFs with your company's internal documentation, runbooks, or architecture decision records.

### 7. Add multi-modal support
Include architecture diagrams by using GPT-4o's vision capabilities alongside text retrieval.

---

##  Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| RAG Framework | LangChain | 0.3.x |
| LLM | GPT-4o-mini (OpenAI) | latest |
| Embeddings | text-embedding-3-small (OpenAI) | latest |
| Vector Store | FAISS (faiss-cpu) | 1.8.x |
| PDF Loader | PyPDFDirectoryLoader | via langchain-community |
| Text Splitter | RecursiveCharacterTextSplitter | via langchain-text-splitters |
| Secrets | python-dotenv | 1.0.x |
| Version Control | Git + GitHub | latest |

---

##  Project Stats

| Metric | Value |
|--------|-------|
| Documents | 5 AWS Well-Architected PDFs |
| Total pages | 1,019 |
| Total chunks | 2,749 |
| Average chunk size | 820 characters |
| Embedding model | text-embedding-3-small |
| Vector dimensions | 1,536 |
| Retrieval k | 4 chunks per query |
| Estimated cost (full run) | < $1 USD |

---

##  License

This project is open source. The AWS Well-Architected Framework documents are publicly available from Amazon Web Services.

---

##  Connect with Rahul

If this tutorial helped you, connect with me:

- **GitHub:** https://github.com/chrahul
- **LinkedIn:** Rahul Chaubey — Director of Cloud Strategy
- **YouTube:** Subscribe for more Cloud & Agentic AI tutorials

> *"The best way to learn AI engineering is to build real systems — not toy demos."*
> — Rahul Chaubey

---

*Built as part of IITM Agentic AI Program — Week 15 Graded Mini Project*