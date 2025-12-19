# app.py

import os
import io
import tempfile
import datetime
from typing import List
import json
import re

import streamlit as st

# LangChain / Vector DB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Vector store (new package)
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# PDF report generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY



# ----------------------------
# UI CONFIG & CONSTANTS
# ----------------------------
st.set_page_config(page_title="Compliance RAG — WWFT / Code vs Company", layout="wide")

PRIMARY_TOPICS = [
    "risk assessment methodology",
    "Anti money laundering",
    "beneficial ownership",
    "PEPs",
    "sanctions screening",
    "suspicious activity reporting (SAR/STR)",
    "third-party reliance",
    "ongoing monitoring",
    "high-risk countries",
    "communication and disclosures",
    "complaints handling and escalation",
    "monitoring and remediation",
]

CSS = """
<style>
:root {
  --primary-color: #FFC20E;
  --bg: #FFFFFF; --bg2: #F7F8FA; --txt: #111; --muted:#666;
}
.stApp { background: var(--bg); color: var(--txt); }
.block-container { padding-top: 1rem; }
button[kind="primary"] { background: var(--primary-color) !important; color: #000 !important; }
.card {background:#fff;border:1px solid #eee;border-radius:12px;padding:16px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------

def _ensure_embeddings():
    # Priority: user-entered key (stored in session)
    key = (
        st.session_state.get("user_api")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        st.warning("⚠️ No OpenAI API key set. Please enter it in the sidebar.")
        return None

    os.environ["OPENAI_API_KEY"] = key  # ensure downstream tools see it
    return OpenAIEmbeddings()

def load_pdf_to_documents(uploaded_file, category: str) -> List[Document]:
    """Save uploaded PDF to temp and load as Documents using PyPDFLoader."""
    if uploaded_file is None:
        return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    for d in docs:
        src = uploaded_file.name
        d.metadata = {
            **(d.metadata or {}),
            "source": src,
            "doc_name": src,
            "doc_category": category,
            "page_label": str(d.metadata.get("page", 0) + 1),
        }
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, chunk_overlap=200, add_start_index=True
        )
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
    return splitter.split_documents(docs)

def build_vectorstore(company_docs: List[Document], benchmark_docs: List[Document]) -> Chroma:
    embeddings = _ensure_embeddings()
    if embeddings is None:
        st.error("Embedding setup failed — please add your OpenAI API key first.")
        return
    persist_dir = os.path.join(tempfile.gettempdir(), f"rag_chroma_{st.session_state.get('run_id','default')}")
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="compliance_rag",
    )
    if company_docs:
        vs.add_documents(company_docs)
    if benchmark_docs:
        vs.add_documents(benchmark_docs)
    return vs

def retrieve(vs: Chroma, query: str, category: str, k: int = 6) -> List[Document]:
    flt = {"doc_category": category} if category else None
    return vs.similarity_search(query, k=k, filter=flt)

# ---------- Gap Analysis Prompt (Jinja2 so braces in JSON are safe)
system_text = """
You are a senior compliance analyst.
Compare the company's policy excerpts against the benchmark excerpts and produce a precise gap analysis in JSON.
The JSON schema MUST be exactly:
{
  "topic": "<short title>",
  "company_summary": "<what company policy says>",
  "benchmark_summary": "<what benchmark requires>",
  "gaps": ["gap 1", "gap 2"],
  "recommendations": ["action 1", "action 2"],
  "citations": [
    {"type":"company","source":"<doc_name>#<page_label>","quote":"<<=40 words>"},
    {"type":"benchmark","source":"<doc_name>#<page_label>","quote":"<<=40 words>"}
  ]
}
If some info is missing on either side, list this as a gap explicitly.
Only return JSON. No extra text.
""".strip()

user_text = """
Topic: {{ topic }}

Company Policy (retrieved excerpts):
{{ company_block }}

Benchmark (retrieved excerpts):
{{ benchmark_block }}
""".strip()

SYSTEM_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_text, template_format="jinja2"),
    HumanMessagePromptTemplate.from_template(user_text, template_format="jinja2"),
])

def docs_to_block(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("doc_name", "?")
        page = d.metadata.get("page_label", "?")
        txt = (d.page_content or "").strip().replace("\n", " ")
        parts.append(f"[{src}#{page}] {txt[:1200]}")
    return "\n\n".join(parts) if parts else "(no results)"

def analyze_topics(vs: Chroma, topics: List[str], model: str = "gpt-4.1") -> List[dict]:
    llm = ChatOpenAI(model=model, temperature=0)
    results = []
    for t in topics:
        company = retrieve(vs, t, category="CompanyPolicy", k=6)
        bench = retrieve(vs, t, category="Benchmark", k=6)
        msgs = SYSTEM_PROMPT.format_messages(
            topic=t,
            company_block=docs_to_block(company),
            benchmark_block=docs_to_block(bench),
        )
        resp = llm.invoke(msgs)
        results.append({"topic": t, "json": resp.content})
    return results

# --------- PDF helpers for nicer citations ----------
def _nice_source(src: str) -> str:
    """Map 'file.pdf#41' -> 'file.pdf (p. 41)' and keep only basename."""
    try:
        base = os.path.basename((src or "").split("#")[0])
    except Exception:
        base = src or "?"
    m = re.search(r"#(\d+)", src or "")
    page = m.group(1) if m else "?"
    return f"{base} (p. {page})"

def _wrap_paragraph(text: str, base_style) -> Paragraph:
    """Wrap long cell text and clamp length for tidy tables."""
    style = ParagraphStyle(
        "wrap",
        parent=base_style,
        wordWrap="CJK",
        spaceAfter=2,
    )
    t = (text or "").strip()
    if len(t) > 600:
        t = t[:600] + " ..."
    # escape minimal HTML
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(t, style)

# ----------------------------
# PDF REPORT (human-readable)
# ----------------------------
def build_pdf_report(results: List[dict], company_name: str, benchmark_name: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )
    styles = getSampleStyleSheet()
    normal = styles["Normal"]; normal.alignment = TA_JUSTIFY
    heading = styles["Heading2"]; heading.spaceAfter = 6
    small = styles["BodyText"]

    story = []
    story.append(Paragraph("<b>Compliance Gap Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Company Policy: <b>{company_name}</b>", normal))
    story.append(Paragraph(f"Benchmark: <b>{benchmark_name}</b>", normal))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(PageBreak())

    # Index (TOC)
    story.append(Paragraph("<b>Index</b>", styles["Heading2"]))
    toc_data = [["#", "Topic"]]
    for idx, r in enumerate(results, start=1):
        toc_data.append([str(idx), r["topic"]])
    table = Table(toc_data, colWidths=[1.5 * cm, 14 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F2F6")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(PageBreak())

    # Sections
    for idx, r in enumerate(results, start=1):
        topic = r["topic"]
        story.append(Paragraph(f"{idx}. {topic.title()}", heading))
        try:
            data = json.loads(r["json"])
        except Exception:
            data = None

        if not data:
            story.append(Paragraph("⚠️ Could not parse AI output.", normal))
            story.append(PageBreak()); continue

        story.append(Paragraph("<b>Company Summary</b>", small))
        story.append(Paragraph(data.get("company_summary", ""), normal)); story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Benchmark Summary</b>", small))
        story.append(Paragraph(data.get("benchmark_summary", ""), normal)); story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Identified Gaps</b>", small))
        for g in data.get("gaps", []):
            story.append(Paragraph(f"• {g}", normal))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Recommendations</b>", small))
        for rec in data.get("recommendations", []):
            story.append(Paragraph(f"• {rec}", normal))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Citations</b>", small))
        cite_data = [[Paragraph("Type", small), Paragraph("Source", small), Paragraph("Quote", small)]]
        for c in data.get("citations", []):
            ctype = _wrap_paragraph(c.get("type", ""), small)
            csrc  = _wrap_paragraph(_nice_source(c.get("source", "")), small)
            cqt   = _wrap_paragraph(c.get("quote", ""), small)
            cite_data.append([ctype, csrc, cqt])

        # widen quote column, allow wrapping
        cite_table = Table(cite_data, colWidths=[2.5 * cm, 5.5 * cm, 7.5 * cm])
        cite_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(cite_table)
        story.append(Spacer(1, 6))
        story.append(PageBreak())

    doc.build(story)
    pdf_bytes = buf.getvalue(); buf.close()
    return pdf_bytes

# ----------------------------
# STREAMLIT PAGES
# ----------------------------
def page_ingest():
    st.title("📥 Upload & Index")
    st.caption("Upload one Benchmark PDF and one Company Policy PDF. We'll index them for retrieval.")

    col1, col2 = st.columns(2)
    with col1:
        bench_file = st.file_uploader("Benchmark PDF (e.g., WWFT / Code)", type=["pdf"], key="bench")
    with col2:
        comp_file = st.file_uploader("Company Policy PDF", type=["pdf"], key="company")

    if st.button("Index Documents", type="primary"):
        bench_docs = load_pdf_to_documents(bench_file, category="Benchmark")
        comp_docs = load_pdf_to_documents(comp_file, category="CompanyPolicy")
        all_docs = comp_docs + bench_docs
        if not all_docs:
            st.error("Please upload at least one PDF for both Benchmark and Company Policy.")
            return
        chunks = split_docs(all_docs)
        vs = build_vectorstore(comp_docs, bench_docs)
        st.session_state.vectorstore = vs
        st.session_state.company_name = comp_file.name if comp_file else "Company Policy"
        st.session_state.benchmark_name = bench_file.name if bench_file else "Benchmark"
        st.success(f"Indexed {len(chunks)} chunks.")

def page_compare():
    st.title("📊 Compare & Report")
    if 'vectorstore' not in st.session_state:
        st.info("Upload & index documents first on the 'Upload & Index' page.")
        return

    topics = st.multiselect("Select topics to analyze", options=PRIMARY_TOPICS, default=PRIMARY_TOPICS[:5])
    model = st.selectbox("Model", ["gpt-4.1", "gpt-4o-mini", "gpt-4o"], index=0)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing topics..."):
            results = analyze_topics(st.session_state.vectorstore, topics, model=model)
        st.session_state.results = results
        st.success("Analysis complete.")

    if 'results' in st.session_state and st.session_state.results:
        st.subheader("Results (JSON per topic)")
        for r in st.session_state.results:
            with st.expander(r['topic'], expanded=False):
                st.code(r['json'], language="json")

        if st.button("📄 Generate Indexed PDF"):
            pdf_bytes = build_pdf_report(
                st.session_state.results,
                company_name=st.session_state.get('company_name', 'Company Policy'),
                benchmark_name=st.session_state.get('benchmark_name', 'Benchmark')
            )
            st.download_button(
                label="Download Gap Report (PDF)",
                data=pdf_bytes,
                file_name="gap_report.pdf",
                mime="application/pdf",
            )

def page_chat():
    st.title("💬 Chat with your documents")
    if 'vectorstore' not in st.session_state:
        st.info("Upload & index documents first on the 'Upload & Index' page.")
        return

    if 'chat' not in st.session_state:
        st.session_state.chat = []

    for m in st.session_state.chat:
        with st.chat_message(m['role']):
            st.markdown(m['content'])

    q = st.chat_input("Ask a question about the uploaded PDFs…")
    if not q:
        return

    # minor normalization for common typo
    q_norm = q.replace("wwtf", "wwft").replace("WWTF", "WWFT")

    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    vs = st.session_state.vectorstore
    ctx = vs.similarity_search(q_norm or q, k=6)
    if not ctx:
        fallback = ("I couldn’t find relevant passages in the indexed documents. "
                    "Please ensure both Benchmark and Company Policy PDFs are indexed.")
        with st.chat_message("assistant"):
            st.markdown(fallback)
        st.session_state.chat.append({"role": "assistant", "content": fallback})
        return

    ctx_block = docs_to_block(ctx)

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a concise compliance assistant. Use the provided context to answer briefly (3–5 sentences). "
            "Include inline citations like [doc_name#page_label]. If the question is vague, give a brief overview."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context from documents:\n{{ ctx }}\n\nQuestion: {{ q }}\n\nAnswer:",
            template_format="jinja2"
        )
    ])

    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    resp = llm.invoke(chat_prompt.format_messages(ctx=ctx_block, q=q_norm or q))
    answer_text = re.sub(r'(\S+\.pdf)#(\d+)', r'\1 (p. \2)', resp.content)

    def _nice_src(d):
        name = os.path.basename(d.metadata.get('doc_name', '#'))
        page = d.metadata.get('page_label', '?')
        return f"{name} (p. {page})"

    citations = ", ".join({_nice_src(d) for d in ctx})
    answer = f"{answer_text}\n\n_Citations:_ {citations}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat.append({"role": "assistant", "content": answer})

# ----------------------------
# NAVIGATION
# ----------------------------
def main():
    # Streamlit prefers width='content' or 'stretch'
    st.sidebar.image("ey_logo.png", width='content')
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Navigation", ["Upload & Index", "Compare & Report", "Chat"], index=0)

        # --- API Key configuration ---
    st.sidebar.markdown("### 🔑 OpenAI API Key")
    user_api = st.sidebar.text_input(
        "Paste your OpenAI API Key here (will not be saved):",
        type="password",
        placeholder="sk-...",
    )
    if user_api:
        os.environ["OPENAI_API_KEY"] = user_api
        st.session_state["user_api"] = user_api
        st.sidebar.success("✅ Key active for this session")
    elif "user_api" in st.session_state:
        os.environ["OPENAI_API_KEY"] = st.session_state["user_api"]
    else:
        st.sidebar.warning("⚠️ No API key set — please enter it above before running analysis.")

    if page == "Upload & Index":
        page_ingest()
    elif page == "Compare & Report":
        page_compare()
    else:
        page_chat()

if __name__ == "__main__":
    if 'run_id' not in st.session_state:
        st.session_state.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    main()
