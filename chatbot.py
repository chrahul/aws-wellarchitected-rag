# chatbot.py
# Pipeline 2: Load → Retrieve → Generate
# Conversational RAG assistant for AWS Well-Architected Framework

import os
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ── System prompt — grounding + refusal ───────────────────────────────────
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

# ── Build prompt template ──────────────────────────────────────────────────
def build_prompt():
    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    return ChatPromptTemplate.from_messages(messages)

# ── Load FAISS index from disk ─────────────────────────────────────────────
def load_vectorstore():
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
    print(f"Index loaded successfully.")
    return vectorstore

# ── Build the conversational RAG chain ────────────────────────────────────
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": build_prompt()},
        verbose=False
    )
    return chain

# ── Format source citations ────────────────────────────────────────────────
def format_sources(source_documents):
    seen = set()
    sources = []
    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        filename = os.path.basename(source)
        key = f"{filename}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"  - {filename} (page {page})")
    return "\n".join(sources)

# ── Main chat loop ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("AWS Well-Architected RAG Assistant")
    print("Powered by LangChain + OpenAI + FAISS")
    print("=" * 60)
    print("Ask questions about AWS Well-Architected best practices.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to reset conversation history.")
    print("-" * 60)

    # Load index + build chain
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)

    print("\nReady. Ask your first question:\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            chain.memory.clear()
            print("Conversation history cleared.\n")
            continue

        # Run the RAG chain
        try:
            response = chain({"question": user_input})
            answer = response["answer"]
            source_docs = response.get("source_documents", [])

            print(f"\nAssistant: {answer}")

            if source_docs:
                print(f"\nSources:")
                print(format_sources(source_docs))

            print("-" * 60)

        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()
