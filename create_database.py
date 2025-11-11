# create_database.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, shutil, re

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
COLLECTION = "policy_rag"

def infer_category(path: str) -> str:
    name = os.path.basename(path).lower()
    if "wwft" in name or "wwft-guidelines" in name:
        return "WWFT"
    if "code-of-conduct" in name or "small-business" in name or "lending" in name:
        return "CodeOfConduct"
    return "CompanyPolicy"

def clean_meta(meta: dict) -> dict:
    # make simple, consistent metadata
    out = dict(meta or {})
    src = out.get("source") or out.get("file_path") or ""
    out["source"] = src
    out["doc_name"] = os.path.basename(src) if src else "unknown"
    out["doc_category"] = infer_category(src)
    # keep page label if present
    if "page" in out and "page_label" not in out:
        out["page_label"] = str(out["page"] + 1)
    return out

def load_documents() -> list[Document]:
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()
    # normalize metadata + tag categories
    for d in docs:
        d.metadata = clean_meta(d.metadata)
    return docs

def split_text(documents: list[Document]) -> list[Document]:
    # token-aware splitter (better for PDFs). fallback to char if not installed.
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=200, add_start_index=True
        )
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if chunks:
        i = min(5, len(chunks) - 1)
        print(chunks[i].page_content[:300])
        print(chunks[i].metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # fresh build for this POC
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
    )

    # Chroma 0.4+ auto-persist
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH} (collection={COLLECTION}).")

def main():
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
