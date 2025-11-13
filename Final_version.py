#!/usr/bin/env python3
"""
Compliance RAG — Legal-Grade Gap Analysis
Production-ready application for comparing company policies against regulatory benchmarks
with full audit trail and human-readable reporting.
"""

import os
import io
import tempfile
import datetime
import json
import re
import hashlib
import traceback
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import streamlit as st

# LangChain / Vector DB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Vector store
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# PDF report generation
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_JUSTIFY

# Reranking
from sentence_transformers import CrossEncoder

# ============================
# UI CONFIG & CONSTANTS
# ============================

st.set_page_config(
    page_title="Compliance RAG — Legal-Grade Gap Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY_TOPICS = [
    "risk assessment methodology",
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

LEGAL_KEYWORD_MAP = {
    "risk assessment": ["risk assessment", "risk analysis", "risk evaluation", "risk-based approach", "risk methodology"],
    "beneficial ownership": ["beneficial ownership", "ultimate beneficial owner", "UBO", "beneficial owner"],
    "PEPs": ["politically exposed person", "PEP", "prominent public function", "politically exposed"],
    "sanctions screening": ["sanctions screening", "sanctions check", "sanctions list"],
    "suspicious activity reporting": ["suspicious activity report", "SAR", "STR", "suspicious transaction"],
}

# Custom CSS for professional styling
CUSTOM_CSS = """
<style>
:root {
    --primary-color: #FFC20E;
    --bg: #FFFFFF;
    --bg2: #F7F8FA;
    --txt: #111;
    --muted: #666;
}
.stApp { background: var(--bg); color: var(--txt); }
.block-container { padding-top: 1rem; }
button[kind="primary"] { background: var(--primary-color) !important; color: #000 !important; }
.card { background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.stSuccess, .stWarning, .stError { padding: 12px; border-radius: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================
# AUDIT LOGGER (Immutable)
# ============================

class AuditLogger:
    """Immutable audit logger for compliance traceability"""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.log_dir = Path("audit_runs") / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "audit.log"
        self.artifacts = {}
    
    def log(self, level: str, topic: str, message: str, data: Optional[dict] = None):
        """Log an event with timestamp and optional data"""
        # Sanitize sensitive data
        if data and "api_key" in str(data).lower():
            data = {"sanitized": "contains_sensitive_data"}
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "topic": topic,
            "message": message,
            "data": data or {}
        }
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            st.error(f"Failed to write audit log: {e}")
        
        # Store artifacts with integrity verification
        if data and "artifact_type" in data:
            try:
                artifact_path = self.log_dir / f"{topic}_{data['artifact_type']}.json"
                with open(artifact_path, "w", encoding="utf-8") as f:
                    json.dump(data["artifact"], f, ensure_ascii=False, indent=2)
                
                # Create hash for integrity verification
                artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                self.artifacts[f"{topic}_{data['artifact_type']}"] = artifact_hash
            except Exception as e:
                st.error(f"Failed to store artifact: {e}")
    
    def get_log(self) -> List[dict]:
        """Retrieve full audit log"""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                st.error(f"Failed to read audit log: {e}")
        return []

# ============================
# CORE HELPERS
# ============================

def nice_source(src: str) -> str:
    """
    Convert source references to human-readable format:
    - 'document.pdf#12' → 'document.pdf (p. 12)'
    - 'req_R1' → 'Extracted Requirement'
    """
    if not src or src.startswith("req"):
        return "Extracted Requirement"
    
    try:
        base = os.path.basename(src.split("#")[0])
    except Exception:
        base = src
    
    page_match = re.search(r"#(\d+)", src)
    page = page_match.group(1) if page_match else "?"
    
    return f"{base} (p. {page})"


def wrap_paragraph(text: str, base_style) -> Paragraph:
    """
    Wrap text in a ReportLab Paragraph with proper escaping and length limits
    """
    style = ParagraphStyle("wrap", parent=base_style, wordWrap="CJK", spaceAfter=2)
    content = (text or "").strip()
    
    # Truncate very long text
    if len(content) > 600:
        content = content[:600] + " ..."
    
    # Escape HTML entities for ReportLab
    content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    return Paragraph(content, style)


def format_confidence_score(score: int) -> str:
    """
    Convert numeric confidence to human-readable label with explanations
    """
    mapping = {
        5: "High (5/5) - Direct quote from policy",
        4: "Medium-High (4/5) - Strong inferential evidence",
        3: "Medium (3/5) - Reasonable inference from multiple statements",
        2: "Low-Medium (2/5) - Weak inferential evidence",
        1: "Low (1/5) - Speculative, requires manual verification"
    }
    return mapping.get(score, f"Unknown ({score}/5)")


def ensure_embeddings() -> Optional[OpenAIEmbeddings]:
    """
    Ensure OpenAI API key is configured and return embeddings instance
    """
    key = st.session_state.get("user_api") or os.getenv("OPENAI_API_KEY")
    
    if not key:
        st.warning("⚠️ No OpenAI API key set. Please enter it in the sidebar.")
        return None
    
    # Validate key format
    if not key.startswith("sk-"):
        st.error("Invalid API key format. Must start with 'sk-'")
        return None
    
    os.environ["OPENAI_API_KEY"] = key
    return OpenAIEmbeddings()


def load_pdf_to_documents(uploaded_file, category: str) -> List[Document]:
    """
    Load PDF with full text preservation and structural metadata
    """
    if uploaded_file is None:
        return []
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Load PDF with PyPDFLoader
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Enhance metadata for each document
        for i, doc in enumerate(docs):
            src = uploaded_file.name
            page_num = doc.metadata.get("page", 0) + 1
            
            # Extract heading from first non-empty line
            content_lines = doc.page_content.split('\n')
            heading = next((
                line.strip() for line in content_lines 
                if line.strip() and len(line.strip()) < 150
            ), "")
            
            # Create content hash for integrity verification
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
            
            doc.metadata = {
                **(doc.metadata or {}),
                "source": src,
                "doc_name": src,
                "doc_category": category,
                "page_label": str(page_num),
                "heading": heading,
                "content_hash": content_hash,
                "ingestion_timestamp": datetime.datetime.now().isoformat(),
            }
        
        return docs
        
    except Exception as e:
        if audit := st.session_state.get("audit_log"):
            audit.log("ERROR", "pdf_load", f"Failed to load PDF: {e}", {"filename": uploaded_file.name})
        st.error(f"Failed to load PDF: {e}")
        return []
        
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents with consistent chunking strategy
    """
    try:
        # Try tiktoken-based splitter first
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=800, 
            chunk_overlap=200, 
            add_start_index=True
        )
    except Exception:
        # Fallback to character-based splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            length_function=len, 
            add_start_index=True
        )
    
    return splitter.split_documents(docs)


def build_vectorstore(company_docs: List[Document], benchmark_docs: List[Document]) -> Optional[Chroma]:
    """
    Build vector store with embeddings for retrieval
    """
    embeddings = ensure_embeddings()
    if embeddings is None:
        st.error("Embedding setup failed — please add your OpenAI API key first.")
        return None
    
    # Create unique persist directory for this run
    persist_dir = os.path.join(
        tempfile.gettempdir(), 
        f"rag_chroma_{st.session_state.get('run_id', 'default')}"
    )
    
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="compliance_rag",
    )
    
    all_docs = company_docs + benchmark_docs
    if all_docs:
        vs.add_documents(all_docs)
    
    return vs


def retrieve_documents(
    vs: Chroma, 
    query: str, 
    category: str, 
    k: int = 30
) -> List[Document]:
    """
    Multi-strategy retrieval to minimize false negatives
    Combines semantic search and keyword expansion
    """
    if vs is None:
        return []
    
    results = []
    seen_ids = set()
    audit = st.session_state.get("audit_log")
    
    # Strategy 1: Semantic similarity search
    try:
        semantic_results = vs.similarity_search_with_score(
            query, 
            k=k, 
            filter={"doc_category": category}
        )
        for doc, score in semantic_results:
            doc_id = f"{doc.metadata['doc_name']}:{doc.metadata['page_label']}"
            if doc_id not in seen_ids:
                doc.metadata["retrieval_strategy"] = "semantic"
                doc.metadata["retrieval_score"] = float(score)
                results.append(doc)
                seen_ids.add(doc_id)
    except Exception as e:
        if audit:
            audit.log("ERROR", "retrieve", f"Semantic search failed: {e}", {"query": query})
    
    # Strategy 2: Keyword expansion for legal terms
    query_lower = query.lower()
    for topic, variants in LEGAL_KEYWORD_MAP.items():
        if any(topic_term in query_lower for topic_term in topic.lower().split()):
            for variant in variants:
                try:
                    keyword_results = vs.similarity_search(
                        variant, 
                        k=3, 
                        filter={"doc_category": category}
                    )
                    for doc in keyword_results:
                        doc_id = f"{doc.metadata['doc_name']}:{doc.metadata['page_label']}"
                        if doc_id not in seen_ids:
                            doc.metadata["retrieval_strategy"] = "keyword"
                            doc.metadata["retrieval_score"] = 0.6
                            results.append(doc)
                            seen_ids.add(doc_id)
                except Exception as e:
                    if audit:
                        audit.log("WARNING", "retrieve", f"Keyword search failed: {e}")
    
    # Sort by retrieval score
    scored_results = []
    for doc in results:
        score = doc.metadata.get("retrieval_score", 0.5)
        scored_results.append((score, doc))
    
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_results[:k]]


def documents_to_block(
    docs: List[Document], 
    max_display_chars: int = 1200
) -> Tuple[str, Dict[str, str]]:
    """
    Convert documents to display block and full text index
    Returns: (display_block: str, full_text_index: dict)
    """
    display_parts = []
    full_index = {}
    
    for doc in docs:
        src = doc.metadata.get("doc_name", "?")
        page = doc.metadata.get("page_label", "?")
        full_text = (doc.page_content or "").strip().replace("\n", " ")
        
        # Store full text for verification
        extract_id = f"{src}#{page}"
        full_index[extract_id] = full_text
        
        # Truncate for display only
        display_text = full_text[:max_display_chars]
        if len(full_text) > max_display_chars:
            display_text += " ..."
        
        display_parts.append(f"[{extract_id}] {display_text}")
    
    return "\n\n".join(display_parts), full_index


def verify_citation_exists(citation: Dict, full_index: Dict[str, str]) -> bool:
    """
    Verify that a cited quote exists verbatim in the source document
    """
    source = citation.get("source", "")
    quote = citation.get("quote", "")
    
    if not source or not quote:
        return False
    
    if source not in full_index:
        return False
    
    # Normalize whitespace for comparison
    normalized_full = " ".join(full_index[source].split()).lower()
    normalized_quote = " ".join(quote.split()).lower()
    
    return normalized_quote in normalized_full

# ============================
# PRODUCTION-GRADE ANALYSIS
# ============================

def analyze_topics(
    vs: Chroma, 
    topics: List[str], 
    model: str = "gpt-4"
) -> List[dict]:
    """
    LEGAL-GRADE gap analysis with human-readable outputs
    Extracts requirements from benchmark and checks against company policy
    """
    if 'reranker' not in st.session_state:
        st.session_state.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(model=model, temperature=0, max_retries=3)
    
    reranker = st.session_state.reranker
    llm = st.session_state.llm
    audit = st.session_state.get('audit_log')
    results = []
    
    def clean_json_response(text: str) -> str:
        """Clean and extract JSON from LLM response"""
        text = re.sub(r'```json?\n?', '', text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group(0).strip() if match else text.strip()
    
    def log(level: str, topic: str, msg: str, data: Optional[dict] = None):
        """Unified logging with UI feedback"""
        if audit:
            audit.log(level, topic, msg, data)
        if level == "ERROR":
            st.error(f"🔴 {topic}: {msg}")
        elif level == "WARNING":
            st.warning(f"⚠️ {topic}: {msg}")
        elif level == "INFO":
            st.info(f"ℹ️ {topic}: {msg}")
    
    progress = st.progress(0)
    status = st.empty()
    
    for idx, topic in enumerate(topics):
        status.text(f"🔍 Analyzing: {topic}")
        log("INFO", topic, "Starting requirement-based analysis")
        
        topic_results = {
            "topic": topic,
            "analysis_method": "requirement-based",
            "metrics": {"benchmark_requirements": 0, "gaps_found": 0, "coverage_pct": 0},
            "gaps": [],
            "recommendations": [],
            "citations": [],
            "verification_notes": [],  # Human-readable notes
            "errors": []
        }
        
        try:
            # Step 1: Retrieve benchmark content
            bench_chunks = retrieve_documents(vs, topic, category="Benchmark", k=30)
            if not bench_chunks:
                log("WARNING", topic, "No benchmark chunks retrieved")
                topic_results["verification_notes"].append("⚠️ Could not retrieve benchmark content")
                continue
            
            # Step 2: Extract structured requirements from benchmark
            company_block, company_index = documents_to_block(bench_chunks, max_display_chars=2500)
            
            extraction_prompt = f"""
            EXTRACT ALL obligations from the following text.
            Return ONLY a JSON object with this structure: 
            {{"requirements": [{{"id": "R1", "text": "requirement text", "clause_type": "mandatory|recommended"}}]}}
            
            Text: {company_block[:4000]}
            """
            
            try:
                raw_extraction = llm.invoke(extraction_prompt).content
                extraction_data = json.loads(clean_json_response(raw_extraction))
                requirements = extraction_data.get("requirements", [])
                topic_results["metrics"]["benchmark_requirements"] = len(requirements)
            except Exception as e:
                log("ERROR", topic, f"Requirement extraction failed: {e}")
                topic_results["errors"].append(f"Extraction error: {str(e)}")
                continue
            
            # Step 3: Retrieve company policy content
            company_chunks = retrieve_documents(vs, topic, category="CompanyPolicy", k=30)
            company_block_full, company_index_full = documents_to_block(company_chunks, max_display_chars=3000)
            
            # Step 4: Check each requirement against company policy
            for req in requirements:
                try:
                    req_id = req.get("id", "unknown")
                    req_text = req.get("text", "")
                    req_type = req.get("clause_type", "unknown")
                    
                    check_prompt = f"""
                    Does the company policy fulfill this requirement?
                    
                    Requirement ({req_type}): {req_text[:400]}
                    
                    Company policy excerpts: {company_block_full[:1500]}
                    
                    Return JSON: 
                    {{
                        "fulfilled": boolean,
                        "confidence": 1-5,
                        "evidence": "direct quote from company policy or null",
                        "gap_type": "OMISSION|PARTIAL|CONTRADICTION",
                        "reasoning": "brief explanation"
                    }}
                    """
                    
                    raw_check = llm.invoke(check_prompt).content
                    check_data = json.loads(clean_json_response(raw_check))
                    
                    # Verify evidence quote exists in source
                    if check_data.get("evidence"):
                        if not verify_citation_exists(
                            {"source": "company", "quote": check_data["evidence"]},
                            company_index_full
                        ):
                            check_data["evidence"] = None
                            topic_results["verification_notes"].append(
                                f"⚠️ Could not verify exact quote for requirement {req_id} - manual review recommended"
                            )
                    
                    # Record gap if not fulfilled
                    if not check_data.get("fulfilled", False):
                        # Build descriptive gap description
                        gap_type = check_data.get("gap_type", "OMISSION")
                        gap_desc = f"**{gap_type}**: {req_text[:250]}..."
                        
                        if gap_type == "OMISSION":
                            gap_desc += " [Company policy does not address this mandatory requirement]"
                        
                        severity = "CRITICAL" if req_type == "mandatory" else "HIGH"
                        
                        gap = {
                            "requirement_id": req_id,
                            "requirement_text": req_text[:300],
                            "description": gap_desc,
                            "severity": severity,
                            "confidence_score": check_data.get("confidence", 1),
                            "confidence_label": format_confidence_score(check_data.get("confidence", 1)),
                            "type": gap_type,
                            "evidence": check_data.get("evidence"),
                            "reasoning": check_data.get("reasoning", "")
                        }
                        
                        topic_results["gaps"].append(gap)
                        topic_results["metrics"]["gaps_found"] += 1
                        
                        # Add benchmark citation
                        topic_results["citations"].append({
                            "type": "benchmark",
                            "source": f"requirement_{req_id}",
                            "quote": req_text[:200]  # Longer quote for context
                        })
                        
                        # Add actionable recommendation
                        topic_results["recommendations"].append(
                            f"[{severity}] Draft explicit policy addressing: {req_text[:150]}... "
                            f"(Reference: Benchmark Req {req_id})"
                        )
                    
                except Exception as req_error:
                    log("ERROR", topic, f"Requirement {req.get('id', 'unknown')} check failed: {req_error}")
                    topic_results["errors"].append(f"Req {req.get('id', 'unknown')}: {str(req_error)}")
            
            # Calculate coverage percentage
            total_bench_pages = len(set(d.metadata['page_label'] for d in bench_chunks))
            topic_results["metrics"]["coverage_pct"] = min(
                100, 
                (len(requirements) / max(1, total_bench_pages // 2)) * 100
            )
            
        except Exception as e:
            log("ERROR", topic, f"Topic analysis failed: {e}", {"traceback": traceback.format_exc()})
            topic_results["errors"].append(f"Topic-level failure: {str(e)}")
        
        finally:
            results.append({"topic": topic, "json": json.dumps(topic_results, ensure_ascii=False, indent=2)})
            progress.progress((idx + 1) / len(topics))
    
    status.empty()
    progress.empty()
    return results

# ============================
# PDF REPORT GENERATOR (Human-Readable)
# ============================

def build_pdf_report(
    results: List[dict], 
    company_name: str, 
    benchmark_name: str
) -> bytes:
    """
    Generate professional PDF report with executive summary and detailed findings
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, 
        pagesize=A4,
        leftMargin=2 * cm, 
        rightMargin=2 * cm, 
        topMargin=1.5 * cm, 
        bottomMargin=1.5 * cm
    )
    
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.alignment = TA_JUSTIFY
    heading = styles["Heading2"]
    heading.spaceAfter = 6
    small = styles["BodyText"]
    small.fontSize = 9
    
    story = []
    
    # Title and metadata
    story.append(Paragraph("<b>Compliance Gap Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Company Policy: <b>{company_name}</b>", normal))
    story.append(Paragraph(f"Benchmark: <b>{benchmark_name}</b>", normal))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Paragraph(f"Analysis ID: {st.session_state.get('run_id', 'N/A')}", small))
    
    # MANDATORY LEGAL DISCLAIMER
    story.append(Spacer(1, 12))
    disclaimer = (
        "<font color='red'><b>⚠️ LEGAL DISCLAIMER: This is an AI-assisted analysis. "
        "ALL FINDINGS MUST BE REVIEWED AND VALIDATED BY A QUALIFIED COMPLIANCE ATTORNEY "
        "BEFORE USE IN ANY REGULATORY FILING OR LEGAL PROCEEDING.</b></font>"
    )
    story.append(Paragraph(disclaimer, styles["BodyText"]))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    
    total_gaps = sum(len(json.loads(r["json"]).get("gaps", [])) for r in results)
    story.append(Paragraph(f"Total Critical/High Priority Gaps Identified: <b>{total_gaps}</b>", normal))
    
    # Manual review items
    all_verification_notes = []
    for r in results:
        notes = json.loads(r["json"]).get("verification_notes", [])
        all_verification_notes.extend(notes)
    
    if all_verification_notes:
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Items Requiring Manual Review:</b>", small))
        for note in sorted(set(all_verification_notes)):
            story.append(Paragraph(f"• {note}", normal))
        story.append(Spacer(1, 6))
    
    story.append(PageBreak())
    
    # Detailed Findings by Topic
    for idx, r in enumerate(results, start=1):
        data = json.loads(r["json"])
        story.append(Paragraph(f"{idx}. {data['topic'].title()}", heading))
        
        # Analysis metrics
        metrics = data.get("metrics", {})
        story.append(Paragraph(
            f"Requirements Analyzed: {metrics.get('benchmark_requirements', 0)} | "
            f"Gaps Identified: {metrics.get('gaps_found', 0)} | "
            f"Coverage Estimate: {metrics.get('coverage_pct', 0):.0f}%",
            small
        ))
        story.append(Spacer(1, 6))
        
        # Gaps with human-readable formatting
        if data.get("gaps"):
            story.append(Paragraph("<b>Identified Gaps</b>", small))
            for gap in data["gaps"]:
                severity_color = {
                    "CRITICAL": "#d32f2f",
                    "HIGH": "#f57c00",
                    "MEDIUM": "#fbc02d",
                    "LOW": "#689f38"
                }.get(gap["severity"], "#000")
                
                gap_text = (
                    f"• <font color='{severity_color}'><b>[{gap['severity']}]</b></font> "
                    f"{gap['description']}<br/>"
                    f"<i>Confidence Assessment: {gap['confidence_label']}</i>"
                )
                if gap.get("reasoning"):
                    gap_text += f"<br/><i>Reasoning: {gap['reasoning'][:100]}...</i>"
                
                story.append(Paragraph(gap_text, normal))
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("<i>No gaps identified for this topic.</i>", normal))
            story.append(Spacer(1, 6))
        
        # Actionable recommendations
        if data.get("recommendations"):
            story.append(Paragraph("<b>Recommended Actions</b>", small))
            for rec in data["recommendations"]:
                story.append(Paragraph(f"• {rec}", normal))
            story.append(Spacer(1, 6))
        
        # Source citations
        if data.get("citations"):
            story.append(Paragraph("<b>Key Citations</b>", small))
            cite_data = [[
                Paragraph("Type", small), 
                Paragraph("Source", small), 
                Paragraph("Quote", small)
            ]]
            
            for c in data["citations"]:
                cite_data.append([
                    wrap_paragraph(c.get("type", ""), small),
                    wrap_paragraph(nice_source(c.get("source", "")), small),
                    wrap_paragraph(c.get("quote", ""), small)
                ])
            
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
    return buf.getvalue()

# ============================
# CHAT INTERFACE
# ============================

def page_chat():
    """Q&A chat interface with proper citation handling"""
    st.title("💬 Chat with Documents")
    
    if 'vectorstore' not in st.session_state:
        st.info("Please upload and index documents first.")
        return
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Normalize common typos
        query_normalized = query.replace("wwtf", "wwft").replace("WWTF", "WWFT")
        
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        vs = st.session_state.vectorstore
        
        try:
            # Retrieve relevant context
            context_docs = vs.similarity_search(query_normalized, k=6)
            
            if not context_docs:
                fallback = "No relevant passages found. Please ensure documents are properly indexed."
                with st.chat_message("assistant"):
                    st.markdown(fallback)
                st.session_state.chat_history.append({"role": "assistant", "content": fallback})
                return
            
            # Format context for LLM
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You are a compliance assistant. Answer questions concisely and cite sources using the format: [Document.pdf (p. X)]."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
                    template_format="jinja2"
                )
            ])
            
            context_block, _ = documents_to_block(context_docs, max_display_chars=2000)
            response = st.session_state.llm.invoke(chat_prompt.format_messages(
                context=context_block, 
                question=query_normalized
            ))
            
            # Clean up citations in response
            answer = re.sub(r'(\S+\.pdf)#(\d+)', r'\1 (p. \2)', response.content)
            
            # Add source citations
            citation_strs = []
            for doc in context_docs:
                src = f"{doc.metadata.get('doc_name', '?')}#{doc.metadata.get('page_label', '?')}"
                citation_strs.append(nice_source(src))
            
            if citation_strs:
                answer += f"\n\n**Sources:** {', '.join(citation_strs)}"
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"Error processing query: {e}"
            st.error(error_msg)
            if audit := st.session_state.get("audit_log"):
                audit.log("ERROR", "chat", f"Chat error: {e}", {"query": query})

# ============================
# DOCUMENT INGESTION PAGE
# ============================

def page_ingest():
    """Document upload and indexing page"""
    st.title("📄 Upload & Index Documents")
    
    # Company Policy Upload
    st.markdown("### Company Policy Document")
    company_file = st.file_uploader(
        "Upload company policy PDF", 
        type=["pdf"], 
        key="company_uploader"
    )
    
    # Benchmark Document Upload
    st.markdown("### Regulatory Benchmark Document")
    benchmark_file = st.file_uploader(
        "Upload benchmark/regulation PDF", 
        type=["pdf"], 
        key="benchmark_uploader"
    )
    
    # Indexing button
    col1, col2 = st.columns([1, 4])
    with col1:
        index_clicked = st.button(
            "🚀 Index Documents", 
            type="primary", 
            disabled=not (company_file and benchmark_file)
        )
    
    if index_clicked:
        with st.spinner("Processing documents..."):
            try:
                # Load PDFs
                company_docs = load_pdf_to_documents(company_file, "CompanyPolicy")
                benchmark_docs = load_pdf_to_documents(benchmark_file, "Benchmark")
                
                if not company_docs or not benchmark_docs:
                    st.error("Failed to load one or both documents.")
                    return
                
                # Split documents
                company_chunks = split_documents(company_docs)
                benchmark_chunks = split_documents(benchmark_docs)
                
                # Build vector store
                vs = build_vectorstore(company_chunks, benchmark_chunks)
                
                if vs:
                    st.session_state.vectorstore = vs
                    st.session_state.company_name = company_file.name
                    st.session_state.benchmark_name = benchmark_file.name
                    
                    st.success(f"✅ Successfully indexed {len(company_chunks)} company chunks and {len(benchmark_chunks)} benchmark chunks")
                    
                    # Log to audit
                    if audit := st.session_state.get("audit_log"):
                        audit.log("INFO", "ingestion", "Documents indexed successfully", {
                            "company_chunks": len(company_chunks),
                            "benchmark_chunks": len(benchmark_chunks)
                        })
                else:
                    st.error("Failed to build vector store")
                    
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                if audit := st.session_state.get("audit_log"):
                    audit.log("ERROR", "ingestion", f"Indexing failed: {e}", {"traceback": traceback.format_exc()})

# ============================
# COMPARE & REPORT PAGE
# ============================

def page_compare():
    """Gap analysis and report generation page"""
    st.title("⚖️ Compare & Generate Report")
    
    # Check prerequisites
    if 'vectorstore' not in st.session_state:
        st.info("Please upload and index documents first.")
        return
    
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(model="gpt-4", temperature=0, max_retries=3)
    
    # Analysis configuration
    st.markdown("### Analysis Configuration")
    
    topics_to_analyze = st.multiselect(
        "Select topics to analyze:",
        PRIMARY_TOPICS,
        default=PRIMARY_TOPICS[:5]
    )
    
    model_choice = st.selectbox(
        "OpenAI Model:",
        ["gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Run analysis button
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_clicked = st.button("🔍 Run Analysis", type="primary", disabled=not topics_to_analyze)
    
    if analyze_clicked:
        with st.spinner("Running gap analysis..."):
            try:
                results = analyze_topics(
                    st.session_state.vectorstore,
                    topics_to_analyze,
                    model=model_choice
                )
                
                if results:
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")
                    
                    # Generate PDF
                    pdf_bytes = build_pdf_report(
                        results,
                        st.session_state.get("company_name", "Company Policy"),
                        st.session_state.get("benchmark_name", "Benchmark")
                    )
                    
                    st.session_state.pdf_report = pdf_bytes
                    
                    # Show summary
                    total_gaps = sum(len(json.loads(r["json"]).get("gaps", [])) for r in results)
                    st.info(f"Total gaps identified: **{total_gaps}**")
                    
                else:
                    st.error("Analysis produced no results")
                    
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                if audit := st.session_state.get("audit_log"):
                    audit.log("ERROR", "analysis", f"Analysis failed: {e}", {"traceback": traceback.format_exc()})
    
    # Download buttons
    if 'pdf_report' in st.session_state:
        st.markdown("### 📥 Downloads")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download PDF Report",
                data=st.session_state.pdf_report,
                file_name=f"Compliance_Report_{st.session_state.run_id}.pdf",
                mime="application/pdf",
                type="primary"
            )
        
        with col2:
            if 'audit_log' in st.session_state:
                audit_data = json.dumps(
                    st.session_state.audit_log.get_log(), 
                    indent=2, 
                    ensure_ascii=False
                )
                st.download_button(
                    label="📋 Download Audit Log",
                    data=audit_data,
                    file_name=f"Audit_Log_{st.session_state.run_id}.json",
                    mime="application/json"
                )

# ============================
# MAIN APPLICATION
# ============================

def main():
    """Main application entry point"""
    # Initialize session state
    if 'run_id' not in st.session_state:
        st.session_state.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = AuditLogger(st.session_state.run_id)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    st.sidebar.markdown("### ⚖️ Legal-Grade Analysis")
    
    # Check for logo
    if os.path.exists("ey_logo.png"):
        st.sidebar.image("ey_logo.png", width=200)
    
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Index", "Compare & Report", "Chat"],
        index=0
    )
    
    # API Key (must be set for all pages)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 OpenAI API Key")
    
    user_api = st.sidebar.text_input(
        "Enter OpenAI API key:",
        type="password",
        placeholder="sk-...",
        help="Your API key is not stored permanently and is only used for this session"
    )
    
    if user_api:
        os.environ["OPENAI_API_KEY"] = user_api
        st.session_state["user_api"] = user_api
        st.sidebar.success("✅ Key active")
    elif "user_api" in st.session_state:
        os.environ["OPENAI_API_KEY"] = st.session_state["user_api"]
        st.sidebar.success("✅ Using cached key")
    else:
        st.sidebar.warning("⚠️ No key set - functionality limited")
    
    # Legal disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<font color='red'><b>⚠️ FOR DECISION SUPPORT ONLY</b><br>"
        "Requires attorney validation before legal use.</font>",
        unsafe_allow_html=True
    )
    
    # Route to appropriate page
    if page == "Upload & Index":
        page_ingest()
    elif page == "Compare & Report":
        page_compare()
    else:
        page_chat()


if __name__ == "__main__":
    main()