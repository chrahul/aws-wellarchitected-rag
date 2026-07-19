import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print("=" * 60)
print("AWS Well-Architected RAG — Document Ingestion Pipeline")
print("=" * 60)

print("\n[1/4] Loading PDF documents...")
loader = PyPDFDirectoryLoader("./documents/")
documents = loader.load()
print(f"      Loaded {len(documents)} pages from {len(set(d.metadata['source'] for d in documents))} documents")

print("\n[2/4] Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"      Created {len(chunks)} chunks")
print(f"      Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

print("\n[3/4] Generating embeddings (calling OpenAI API)...")
print("      This may take 1-2 minutes...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"      Embeddings generated for {len(chunks)} chunks")

print("\n[4/4] Saving FAISS index to disk...")
vectorstore.save_local("./faiss_index")
print("      Saved: faiss_index/index.faiss")
print("      Saved: faiss_index/index.pkl")

print("\n" + "=" * 60)
print("Ingestion complete!")
print(f"  Documents : {len(set(d.metadata['source'] for d in documents))}")
print(f"  Pages     : {len(documents)}")
print(f"  Chunks    : {len(chunks)}")
print("  Index     : ./faiss_index/")
print("=" * 60)
print("\nYou can now run: python chatbot.py")