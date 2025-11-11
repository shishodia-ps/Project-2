# analyze_gaps.py
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os, textwrap, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma")
COLLECTION = "policy_rag"

WWFT_TOPICS = [
    "risk assessment methodology", "customer due diligence (CDD)", "enhanced due diligence (EDD)",
    "beneficial ownership", "PEPs", "sanctions screening", "transaction monitoring",
    "suspicious activity reporting (SAR/STR)", "record keeping and retention",
    "training and awareness", "governance roles and responsibilities",
    "third-party reliance", "ongoing monitoring", "high-risk countries",
]
CONDUCT_TOPICS = [
    "fair lending and transparency", "communication and disclosures",
    "complaints handling and escalation", "forbearance and hard-ship measures",
    "fees and charges", "collections practices", "SME treatment standards",
    "governance and accountability", "monitoring and remediation",
]

def load_db():
    emb = OpenAIEmbeddings()
    return Chroma(
        embedding_function=emb,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION,
    )

def fetch_context(db, query: str, category: str, k: int = 6):
    # Chroma filter by metadata
    docs = db.similarity_search(query, k=k, filter={"doc_category": category})
    return docs

SYSTEM_PROMPT = """You are a compliance analyst.
Compare the company's policy text against the cited regulatory text and codes.
Return a gap analysis in this exact JSON schema:
{{
  "topic": "<short title>",
  "company_summary": "<what company policy says>",
  "benchmark_summary": "<what WWFT/Code require>",
  "gaps": ["gap 1", "gap 2", "..."],
  "recommendations": ["action 1", "action 2", "..."],
  "citations": [
    {{"type":"company","source":"<doc_name>#<page_label>","quote":"<short quote>"}},
    {{"type":"wwft","source":"<doc_name>#<page_label>","quote":"<short quote>"}},
    {{"type":"code","source":"<doc_name>#<page_label>","quote":"<short quote>"}}
  ]
}}
If information is missing, be explicit and put a gap for it. Keep quotes short (<=40 words).
"""

USER_PROMPT = """Topic: {topic}

Company Policy (retrieved excerpts):
{company_block}

WWFT (retrieved excerpts):
{wwft_block}

Code of Conduct (retrieved excerpts):
{code_block}
"""

def docs_block(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("doc_name", "unknown")
        page = d.metadata.get("page_label") or d.metadata.get("page") or "?"
        lines.append(f"[{src}#{page}] {d.page_content.strip()[:1200]}")
    return "\n\n".join(lines) if docs else "(no results)"

def analyze_topic(db, llm, topic: str):
    comp = fetch_context(db, topic, "CompanyPolicy", k=6)
    wwft = fetch_context(db, topic, "WWFT", k=6)
    code = fetch_context(db, topic, "CodeOfConduct", k=6)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])
    msgs = prompt.format_messages(
        topic=topic,
        company_block=docs_block(comp),
        wwft_block=docs_block(wwft),
        code_block=docs_block(code),
    )
    resp = llm.invoke(msgs)
    return resp.content

def write_report(results: list[str], outfile: str):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    header = f"# Gap Analysis Report\nGenerated: {ts}\n\n"
    body = ""
    for item in results:
        body += f"```json\n{item}\n```\n\n"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(header + body)
    print(f"Wrote report → {outfile}")

def main():
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env"

    db = load_db()
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    topics = WWFT_TOPICS + CONDUCT_TOPICS
    results = []
    for t in topics:
        print(f"Analyzing: {t} ...")
        results.append(analyze_topic(db, llm, t))

    out = os.path.join(BASE_DIR, "gap_report.md")
    write_report(results, out)

if __name__ == "__main__":
    main()
