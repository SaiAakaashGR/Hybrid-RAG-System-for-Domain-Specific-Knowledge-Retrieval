#run server: uv run uvicorn main:app
#run ingest: npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
#run qdrant: docker run -d --name qdrantRagDB -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
#stop docker: docker stop qdrantRagDB

import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult
from reranker import Reranker
from retrieval_pipeline import RetrievalPipeline
from query_engine import QueryEngine




load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context)->RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


    def _upsert(chunks_and_src:RAGChunkAndSrc)->RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks)) ]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
# async def rag_query_pdf_ai(ctx: inngest.Context):
#     def _search(question: str, top_k: int = 5) -> RAGSearchResult:

#         pipeline = RetrievalPipeline()

#         contexts, sources, mode = pipeline.retrieve(
#             question,
#             top_k
#         )

#         logging.info(f"Retrieval mode used: {mode}")

#         return RAGSearchResult(
#             contexts=contexts,
#             sources=sources
#         )


#     question = ctx.event.data["question"]
#     top_k = int(ctx.event.data.get("top_k", 5))
#     #found=await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

#     engine = QueryEngine()

#     contexts, sources = await engine.retrieve_contexts(
#         ctx,
#         question,
#         top_k
#     )

#     context_block = "\n\n".join(f"- {c}" for c in contexts)


#     context_block = "\n\n".join(f"- {c}" for c in found.contexts)
#     user_content = (
#         "Use the following context to answer the question.\n\n"
#         f"Context:\n{context_block}\n\n"
#         f"Question: {question}\n"
#         "Answer concisely using the context above"
#     )
#     adapter = ai.openai.Adapter(
#         auth_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-5-nano"
#     )

#     res = await ctx.step.ai.infer(
#         "llm-answer",
#         adapter=adapter,
#         body={
#             "max_completion_tokens": 1024,
#             "messages": [
#                 {"role": "system", "content": "You answer questions using only the provided context"},
#                 {"role": "user", "content": user_content},
                
#             ]
#         }
#     )

#     answer = res["choices"][0]["message"]["content"].strip()
#     return {"answer": answer, "sources":found.sources, "num_contexts": len(found.contexts)}

async def rag_query_pdf_ai(ctx: inngest.Context):

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    engine = QueryEngine()

    contexts, sources = await engine.retrieve_contexts(
        ctx,
        question,
        top_k
    )

    logging.info(f"Retrieved {len(contexts)} contexts")

    context_block = "\n\n".join(f"- {c}" for c in contexts)

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above"
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-5-nano"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_completion_tokens": 1024,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context"
                },
                {"role": "user", "content": user_content},
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": sources,
        "num_contexts": len(contexts)
    }


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])