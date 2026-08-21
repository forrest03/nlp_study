"""
RAG 问答 HTTP 服务（FastAPI）

提供两类接口：
  /query        — 标准问答，返回答案 + 引用
  /query/debug  — 教学调试接口，逐步返回每个检索阶段的中间结果
  /             — 教学可视化 Web 页面
  /health       — 健康检查

启动：
  cd src_my/code
  uvicorn serve:app --host 0.0.0.0 --port 8000

开发模式（修改代码后自动重载）：
  uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

依赖：
  pip install fastapi uvicorn
"""

import os
import importlib.util
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PIPELINE_PATH = Path(__file__).parent / "rag_pipeline.py"


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("rag_pipeline", _PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = None
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _module, pipeline
    logger.info("服务启动，初始化 RAG Pipeline...")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        logger.error("请设置环境变量 DASHSCOPE_API_KEY")
        logger.error("Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx")
        logger.error("Windows:   set DASHSCOPE_API_KEY=sk-xxx")
        raise RuntimeError("缺少 DASHSCOPE_API_KEY 环境变量")

    _module = _load_pipeline_module()
    pipeline = _module.RAGPipeline(
        use_bm25=True,
        use_rerank=False,
        use_query_rewrite=False,
    )
    logger.info("Pipeline 初始化完成，开始接受请求")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="RAG 问答服务",
    description="FAISS + BM25 混合检索 + DashScope qwen-plus，含教学调试接口",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

HTML_DIR = Path(__file__).parent / "html"
app.mount("/static", StaticFiles(directory=str(HTML_DIR)), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., examples=["2025年重点领域和行业节能改造节能量多少标准煤"])
    years: Optional[str] = Field(None, examples=["2024|2025"])


class Citation(BaseModel):
    index: int
    source: str
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class VecChunk(BaseModel):
    rank: int
    vec_score: float
    source: str
    content_preview: str


class BM25Chunk(BaseModel):
    rank: int
    bm25_score: float
    source: str
    content_preview: str


class RRFChunk(BaseModel):
    rank: int
    rrf_score: float
    vec_rank: Optional[int]
    bm25_rank: Optional[int]
    source: str
    content_preview: str


class ContextChunk(BaseModel):
    index: int
    source: str
    content: str


class DebugResponse(BaseModel):
    question: str
    vec_results: list[VecChunk]
    bm25_results: list[BM25Chunk]
    rrf_results: list[RRFChunk]
    context_chunks: list[ContextChunk]
    answer: str
    citations: list[Citation]


def _build_source(item: dict) -> str:
    s = item.get("source_file", "")
    section = item.get("section", "")
    if section:
        parts = section.split(" > ")
        s += " · " + " > ".join(parts[-2:])
    page = item.get("page_num", -1)
    if page and page != -1:
        s += f" · 第{page}页"
    return s


def _preview(text: str, n: int = 150) -> str:
    text = text.strip()
    return text[:n] + "…" if len(text) > n else text


def _filter_meta(req: QueryRequest) -> Optional[dict]:
    if req.years:
        return {"years": req.years.split("|")}
    return None


@app.get("/", summary="首页")
def index():
    return {
        "title": "RAG 问答服务",
        "endpoints": {
            "/query": "标准问答接口",
            "/query/debug": "教学调试接口",
            "/health": "健康检查",
        },
        "usage": {
            "query": {"question": "2025年重点领域和行业节能改造节能量多少标准煤", "years": "2024|2025"},
        },
    }


@app.get("/health", summary="健康检查")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    return {"status": "ok", "pipeline_ready": True}


@app.post("/query", response_model=QueryResponse, summary="标准问答")
def query(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")
    try:
        result = pipeline.query(req.question, filter_meta=_filter_meta(req), verbose=True)
    except Exception as e:
        logger.error(f"Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
    )


@app.post("/query/debug", response_model=DebugResponse, summary="教学调试：逐步返回中间结果")
def query_debug(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    TOP_K = _module.TOP_K_RETRIEVE
    TOP_K_FINAL = _module.TOP_K_RERANK
    fm = _filter_meta(req)

    try:
        vec_results = pipeline.vec_store.search(req.question, TOP_K, fm)
        vec_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(vec_results, 1)}

        bm25_results = pipeline.bm25_store.search(req.question, TOP_K) if pipeline.bm25_store else []
        bm25_rank_map = {item["chunk_id"]: rank for rank, item in enumerate(bm25_results, 1)}

        if bm25_results:
            candidates = _module.reciprocal_rank_fusion(vec_results, bm25_results)
        else:
            candidates = vec_results

        final = candidates[:TOP_K_FINAL]
        context, cits = _module.build_context(final)

        answer = _module.call_llm(req.question, context, pipeline.client)

    except Exception as e:
        logger.error(f"Debug Pipeline 异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return DebugResponse(
        question=req.question,
        vec_results=[
            VecChunk(
                rank=i + 1,
                vec_score=item.get("vec_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(vec_results[:5])
        ],
        bm25_results=[
            BM25Chunk(
                rank=i + 1,
                bm25_score=item.get("bm25_score", 0.0),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(bm25_results[:5])
        ],
        rrf_results=[
            RRFChunk(
                rank=i + 1,
                rrf_score=item.get("rrf_score", 0.0),
                vec_rank=vec_rank_map.get(item["chunk_id"]),
                bm25_rank=bm25_rank_map.get(item["chunk_id"]),
                source=_build_source(item),
                content_preview=_preview(item["content"]),
            ) for i, item in enumerate(candidates[:5])
        ],
        context_chunks=[
            ContextChunk(
                index=i + 1,
                source=_build_source(item),
                content=item.get("parent_content") or item["content"],
            ) for i, item in enumerate(final)
        ],
        answer=answer,
        citations=[Citation(**c) for c in cits],
    )