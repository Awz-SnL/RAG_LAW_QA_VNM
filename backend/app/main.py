"""
main.py  –  FastAPI application
────────────────────────────────
Endpoints:
  GET  /health                   – liveness probe
  GET  /info                     – system info (model, collection stats)
  GET  /documents                – list PDF files in documents folder
  POST /ingest                   – ingest all PDFs into Qdrant
  DELETE /collection             – delete the vector collection
  POST /query                    – RAG answer
  POST /query/no-rag             – pure-LLM answer (no retrieval)
  POST /compare                  – RAG vs no-RAG side-by-side
  POST /evaluate                 – batch evaluation from QA_TEST.xlsx
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from .config import get_settings
from .document_loader import load_documents_from_folder
from .embeddings import get_embedding_model
from .vector_store import VectorStore
from .rag_pipeline import RAGPipeline
from .evaluator import load_qa_test_file, evaluate_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Globals (initialised at startup) ────────────────────────────────────────

_store: VectorStore = None
_pipeline: RAGPipeline = None
_startup_error: Optional[str] = None
_llm_config_error: Optional[str] = None


def _validate_llm_settings() -> None:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "glm" and not settings.glm_api_key:
        raise ValueError("LLM_PROVIDER=glm nhưng GLM_API_KEY đang trống.")
    if provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("LLM_PROVIDER=gemini nhưng GEMINI_API_KEY đang trống.")
    if provider == "openai" and not settings.openai_api_key:
        raise ValueError("LLM_PROVIDER=openai nhưng OPENAI_API_KEY đang trống.")


def _resolve_allowed_sources(settings) -> Set[str]:
    folder = Path(settings.documents_path)
    if not folder.exists():
        return set()
    return {
        f.name
        for f in sorted(list(folder.glob("*.pdf")) + list(folder.glob("*.docx")))
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once on startup."""
    global _store, _pipeline, _startup_error, _llm_config_error
    settings = get_settings()

    try:
        try:
            _validate_llm_settings()
            _llm_config_error = None
        except ValueError as exc:
            _llm_config_error = str(exc)
            logger.warning("LLM config warning: %s", _llm_config_error)

        logger.info("Loading embedding model: %s", settings.embedding_model)
        emb_model = get_embedding_model(settings.embedding_model)

        max_retries = 5
        retry_delay = 3
        last_err = None
        logger.info("Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port)
        for attempt in range(1, max_retries + 1):
            try:
                _store = VectorStore(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    collection_name=settings.collection_name,
                    vector_dim=emb_model.dim,
                )
                _store.collection_info()
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                logger.warning("Qdrant chưa sẵn sàng (lần %d/%d): %s", attempt, max_retries, exc)
                time.sleep(retry_delay)

        if last_err:
            raise RuntimeError(f"Không thể kết nối Qdrant sau {max_retries} lần thử: {last_err}")

        _pipeline = RAGPipeline(
            settings=settings,
            embedding_model=emb_model,
            vector_store=_store,
        )
        _startup_error = None
        logger.info("RAG system ready.")
    except Exception as exc:
        _startup_error = str(exc)
        logger.exception("Startup failed: %s", exc)
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vietnamese RAG QA API",
    description="Hệ thống hỏi–đáp tiếng Việt dựa trên RAG (PhoBERT / Sentence-BERT + Qdrant)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    filter_source: Optional[str] = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Question cannot be empty.")
        return value.strip()


class EvaluateRequest(BaseModel):
    qa_test_path: Optional[str] = None   # defaults to QA_TEST.xlsx in documents parent


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    if _startup_error:
        return {"status": "error", "detail": _startup_error}
    return {"status": "ok"}


@app.get("/info", tags=["System"])
def info():
    settings = get_settings()
    emb = get_embedding_model(settings.embedding_model)
    collection = _store.collection_info() if _store else {}
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dim": emb.dim,
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.llm_model_glm
            if settings.llm_provider == "glm"
            else settings.llm_model_gemini
            if settings.llm_provider == "gemini"
            else settings.llm_model_openai
        ),
        "collection": collection,
        "top_k": settings.top_k,
        "retrieval_top_n": settings.retrieval_top_n,
        "enforce_top_n": settings.enforce_top_n,
        "startup_error": _startup_error,
        "llm_config_error": _llm_config_error,
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all PDF files currently in the documents folder."""
    settings = get_settings()
    folder = Path(settings.documents_path)
    if not folder.exists():
        return {"files": [], "count": 0}
    files = [
        {
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        }
        for f in sorted(list(folder.glob("*.pdf")) + list(folder.glob("*.docx")))
    ]
    return {"files": files, "count": len(files), "path": str(folder)}


@app.post("/ingest", tags=["Documents"])
def ingest_documents(recreate: bool = False):
    """
    Load all PDFs from the documents folder, encode them with the embedding
    model and upsert into Qdrant.

    Set `recreate=true` to wipe and rebuild the collection from scratch.
    """
    settings = get_settings()
    emb_model = get_embedding_model(settings.embedding_model)

    logger.info("Starting ingestion from %s", settings.documents_path)
    chunks, file_names = load_documents_from_folder(
        folder_path=settings.documents_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=404, detail="No PDF files found in documents folder.")

    logger.info("Loaded %d chunks from %d files. Encoding…", len(chunks), len(file_names))

    # Ensure collection exists
    _store.create_collection(recreate=recreate)

    # Batch encode
    texts = [c["text"] for c in chunks]
    vectors = emb_model.encode(texts, batch_size=32, show_progress=True)

    # Upsert
    total = _store.upsert_chunks(chunks, vectors)

    return {
        "status": "success",
        "files_processed": file_names,
        "chunks_ingested": total,
        "collection": _store.collection_info(),
    }


@app.delete("/collection", tags=["Documents"])
def delete_collection():
    """Delete the entire Qdrant collection (all vectors)."""
    _store.delete_collection()
    return {"status": "deleted", "collection": get_settings().collection_name}


@app.post("/query", tags=["QA"])
def query_rag(req: QueryRequest):
    """Answer a question using RAG (retrieval + LLM)."""
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"System startup error: {_startup_error}")
    if _llm_config_error:
        raise HTTPException(status_code=503, detail=f"LLM config error: {_llm_config_error}")
    collection = _store.collection_info()
    if not collection.get("exists") or not collection.get("count"):
        raise HTTPException(status_code=400, detail="Vector DB đang trống. Hãy gọi /ingest trước khi truy vấn.")
    settings = get_settings()
    allowed_sources = _resolve_allowed_sources(settings)
    if not allowed_sources:
        raise HTTPException(status_code=400, detail="Không có file trong documents. Hãy thêm PDF/DOCX và ingest lại.")

    return _pipeline.answer_with_rag(
        question=req.question,
        top_k=req.top_k,
        filter_source=req.filter_source,
        allowed_sources=allowed_sources,
    )


@app.post("/query/top-passages", tags=["QA"])
def query_top_passages(req: QueryRequest):
    """Retrieve top-N passages strictly from ingested documents (no LLM synthesis)."""
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"System startup error: {_startup_error}")
    collection = _store.collection_info()
    if not collection.get("exists") or not collection.get("count"):
        raise HTTPException(status_code=400, detail="Vector DB đang trống. Hãy gọi /ingest trước khi truy vấn.")
    settings = get_settings()
    allowed_sources = _resolve_allowed_sources(settings)
    if not allowed_sources:
        raise HTTPException(status_code=400, detail="Không có file trong documents. Hãy thêm PDF/DOCX và ingest lại.")

    res = _pipeline.retrieve_top_passages(
        question=req.question,
        top_n=req.top_k,
        filter_source=req.filter_source,
        allowed_sources=allowed_sources,
    )

    if res["passages_count"] == 0:
        raise HTTPException(status_code=404, detail="Không tìm thấy đoạn tài liệu phù hợp trong documents đã ingest.")
    if res["passages_count"] < res["top_n_target"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Chỉ tìm thấy {res['passages_count']} đoạn hợp lệ từ documents, "
                f"chưa đủ {res['top_n_target']} đoạn."
            ),
        )
    return res


@app.post("/query/no-rag", tags=["QA"])
def query_no_rag(req: QueryRequest):
    """Answer a question using LLM only – no retrieval (baseline for comparison)."""
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"System startup error: {_startup_error}")
    if _llm_config_error:
        raise HTTPException(status_code=503, detail=f"LLM config error: {_llm_config_error}")
    return _pipeline.answer_without_rag(question=req.question)


@app.post("/compare", tags=["QA"])
def compare_rag_vs_no_rag(req: QueryRequest):
    """Run both RAG and no-RAG for the same question and return both answers."""
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"System startup error: {_startup_error}")
    if _llm_config_error:
        raise HTTPException(status_code=503, detail=f"LLM config error: {_llm_config_error}")
    settings = get_settings()
    allowed_sources = _resolve_allowed_sources(settings)
    return _pipeline.compare(question=req.question, top_k=req.top_k, allowed_sources=allowed_sources)


@app.post("/evaluate", tags=["Evaluation"])
def evaluate(req: EvaluateRequest):
    """
    Run batch evaluation against QA_TEST.xlsx.
    Returns per-question metrics and aggregate scores (EM, F1, ROUGE-L, grounding).
    """
    settings = get_settings()

    # Resolve path
    if req.qa_test_path:
        qa_path = req.qa_test_path
    else:
        # Search in documents folder first, then parent
        docs_folder = Path(settings.documents_path)
        docs_parent = docs_folder.parent
        candidates = (
            list(docs_folder.glob("QA_TEST*.xlsx")) +
            list(docs_folder.glob("qa_test*.xlsx")) +
            list(docs_parent.glob("QA_TEST*.xlsx")) +
            list(docs_parent.glob("qa_test*.xlsx"))
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="QA_TEST.xlsx not found. Provide qa_test_path.")
        qa_path = str(candidates[0])

    logger.info("Loading QA test file: %s", qa_path)
    df = load_qa_test_file(qa_path)

    import time as _time
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        q = str(row["question"]).strip()
        ref = str(row["reference"]).strip()

        if i > 0:
            _time.sleep(7)  # Stay under Gemini free-tier rate limit (10 RPM)

        rag_res = _pipeline.answer_with_rag(q)
        _time.sleep(7)
        no_rag_res = _pipeline.answer_without_rag(q)

        results.append({
            "question": q,
            "reference": ref,
            "rag_answer": rag_res["answer"],
            "no_rag_answer": no_rag_res["answer"],
        })

    return evaluate_batch(results)
