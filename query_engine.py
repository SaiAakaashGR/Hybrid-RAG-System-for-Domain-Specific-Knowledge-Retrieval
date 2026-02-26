import os
from inngest.experimental import ai
from retrieval_pipeline import RetrievalPipeline
from rag_trace import RAGTrace

class QueryEngine:
    def __init__(self, source_id):
        self.pipeline = RetrievalPipeline(source_id)
        self.trace = RAGTrace()
        self.adapter = ai.openai.Adapter(
            auth_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-5-nano"
        )

    async def rewrite_query(self, ctx, question: str):

        prompt = f"""
Rewrite the user question into a precise search query
optimized for document retrieval.

User question:
{question}

Return ONLY the rewritten query.
"""

        res = await ctx.step.ai.infer(
            "rewrite-query",
            adapter=self.adapter,
            body={
                "messages": [
                    {"role": "system", "content": "You optimize search queries."},
                    {"role": "user", "content": prompt},
                ]
            },
        )

        return res["choices"][0]["message"]["content"].strip()

    async def needs_second_hop(self, ctx, question, contexts):

        context_preview = "\n".join(contexts[:3])

        prompt = f"""
Question: {question}

Current evidence:
{context_preview}

Is this enough information to answer confidently?
Reply ONLY with YES or NO.
"""

        res = await ctx.step.ai.infer(
            "multi-hop-check",
            adapter=self.adapter,
            body={
                "messages": [
                    {"role": "system", "content": "You judge evidence sufficiency."},
                    {"role": "user", "content": prompt},
                ]
            },
        )

        answer = res["choices"][0]["message"]["content"].strip().upper()
        
        
        return answer.startswith("NO")
    
    def rerank(self, question, contexts):

        scored = []

        for c in contexts:
            score = len(set(question.lower().split())
                        & set(c.lower().split()))
            scored.append((score, c))

        scored.sort(reverse=True)
        reranked = [c for _, c in scored]

        self.trace.log(
            "Reranking",
            {"before": len(contexts), "after": len(reranked)}
        )

        return reranked

    async def retrieve_contexts(self, ctx, question:str, top_k:int):
        
        #original query
        self.trace.log("Original Query", {"query": question})

        # rewrite
        rewritten = await self.rewrite_query(ctx, question)
        self.trace.log("Rewritten Query", {"rewritten": rewritten})

        contexts, sources, stats = self.pipeline.retrieve(
            rewritten, top_k
        )

        self.trace.log("Hybrid Retrieval", stats)
        contexts = self.rerank(question, contexts)

        # multi-hop decision
        do_second = await self.needs_second_hop(
            ctx, question, contexts
        )
        self.trace.log("Multi-hop Decision", {"decision": do_second})

        if do_second:
            followup_query = f"{question} detailed explanation"
            more_contexts, _, _ = self.pipeline.retrieve(
                followup_query,
                top_k
            )

            # merge unique contexts
            contexts = list(dict.fromkeys(contexts + more_contexts))[:top_k]

        return contexts, sources, self.trace.export()
