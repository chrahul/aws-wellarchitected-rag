import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

SYSTEM_PROMPT = """You are an expert assistant for the AWS Well-Architected Framework.
You help cloud architects, DevOps engineers, and developers understand AWS best practices
across the five pillars: Operational Excellence, Security, Reliability,
Performance Efficiency, and Cost Optimization.

STRICT RULES:
1. Answer ONLY using the context provided below from the AWS Well-Architected documents.
2. If the context does not contain enough information to answer the question, respond exactly with: I don't have enough information in the provided documents.
3. Never use your general knowledge or make up facts.
4. Always cite which pillar your answer comes from when possible.
5. Be concise, professional, and technically accurate.

Context from AWS Well-Architected documents:
{context}"""

def load_vectorstore():
    print("Loading FAISS index from disk...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("Index loaded successfully.")
    return vectorstore

def format_sources(docs):
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

def main():
    print("=" * 60)
    print("AWS Well-Architected RAG Assistant")
    print("Powered by LangChain + OpenAI + FAISS")
    print("=" * 60)
    print("Type 'quit' to exit | 'clear' to reset history")
    print("-" * 60)

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    chat_history = []

    print("\nReady. Ask your first question:\n")

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
            print("History cleared.\n")
            continue

        try:
            docs = retriever.invoke(user_input)
            context = "\n\n".join([d.page_content for d in docs])
            system_msg = SYSTEM_PROMPT.format(context=context)
            messages = [{"role": "system", "content": system_msg}]
            for h in chat_history[-6:]:
                messages.append({"role": "user", "content": h["question"]})
                messages.append({"role": "assistant", "content": h["answer"]})
            messages.append({"role": "user", "content": user_input})
            response = llm.invoke(messages)
            answer = response.content
            chat_history.append({"question": user_input, "answer": answer})
            print(f"\nAssistant: {answer}")
            if docs:
                print(f"\nSources:")
                print(format_sources(docs))
            print("-" * 60)
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
