import os
from inngest.experimental import ai
from retrieval_pipeline import RetrievalPipeline


class QueryEngine:

    def __init__(self):
        self.pipeline = RetrievalPipeline()

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
                    {"role": "system", "content": "You improve search queries."},
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

    async def retrieve_contexts(self, ctx, question, top_k):

        # rewrite
        rewritten = await self.rewrite_query(ctx, question)

        contexts, sources, mode = self.pipeline.retrieve(
            rewritten, top_k
        )

        # multi-hop decision
        do_second = await self.needs_second_hop(
            ctx, question, contexts
        )

        if do_second:
            followup_query = f"{question} detailed explanation"
            more_contexts, _, _ = self.pipeline.retrieve(
                followup_query,
                top_k
            )

            # merge unique contexts
            contexts = list(dict.fromkeys(contexts + more_contexts))[:top_k]

        return contexts, sources
