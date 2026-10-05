"""
Rerank Node

Third node in the policy agent workflow. Consumes the RRF-fused candidate
pool from hybrid_retrieval_node (state["retrieval_results"]) and re-scores
every chunk JOINTLY with the query using a cross-encoder. Unlike the
bi-encoder used in retrieval (query and chunk embedded separately —
"same topic" signal), the cross-encoder attends across query+chunk text
together — an "answers the question" signal.

Keeps rrf_score/source_ranks/hit_count intact on each chunk for debugging;
adds rerank_score. Output is the score-floored top FINAL_TOP_K.
"""

import logging
import math
from typing import Any, Dict, List

from sentence_transformers import CrossEncoder
import re

_HEADER_RE = re.compile(r"^(Policy:.*\n)?(Policy ID:.*\n)?(Section:.*\n)?(Topic:.*\n)?")
from app.agents.brain_agents.policy_agent.state import PolicyAgentState

logger = logging.getLogger(__name__)

# Module-level singleton — same pattern as hybrid_retrieval.py.
# Model load is expensive; never do it per-call.
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,  # hard truncate defensively; your chunks are 45-286 tokens
    device="mps",  # Apple Silicon GPU; fall back to "cpu" if it misbehaves
)

FINAL_TOP_K = 7
SCORE_FLOOR = (
    0.35  # sigmoid space, 0..1 — TUNE after observing distributions (see below)
)


def _rerank_text(chunk: Dict[str, Any]) -> str:
    # Header carries citation identity (kept in metadata); the cross-encoder
    # only needs section content — the identical header inflates all scores.
    return _HEADER_RE.sub("", (chunk.get("text") or ""), count=1).strip()


def rerank_node(state: PolicyAgentState) -> Dict[str, Any]:
    candidates = state.get("retrieval_results") or []
    if not candidates:
        return {"retrieval_results": [], "error": None}

    # Rerank against the retrieval-shaped query, not the raw conversational one
    query = state.get("enhanced_query") or state.get("user_query", "")

    try:
        pairs = [(query, _rerank_text(c)) for c in candidates]

        logits = reranker.predict(pairs, batch_size=16, show_progress_bar=False)

        # raw logits are unbounded → sigmoid → interpretable 0..1
        for chunk, logit in zip(candidates, logits):
            logit = float(logit)
            chunk["rerank_logit"] = logit
            chunk["rerank_score"] = 1.0 / (1.0 + math.exp(-logit))

        reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

        kept = [c for c in reranked if c["rerank_score"] >= SCORE_FLOOR][:FINAL_TOP_K]

        # Never silently hand back an empty context that had candidates —
        # mark low confidence so the generation node can say "not found in policy"
        low_confidence = len(kept) == 0

        logger.info(
            "rerank_node: %d candidates -> %d kept (floor=%.2f), "
            "top=%.3f bottom_kept=%.3f",
            len(candidates),
            len(kept),
            SCORE_FLOOR,
            reranked[0]["rerank_score"],
            kept[-1]["rerank_score"] if kept else -1,
        )
        print("\n========== RERANK DEBUG ==========")
        print("QUERY:", query)

        for chunk, logit in zip(candidates, logits):
            print("\n-----------------------------")
            print("RAW LOGIT:", float(logit))
            print(
                "SIGMOID:",
                1 / (1 + math.exp(-float(logit)))
            )
            print("RRF:", chunk.get("rrf_score"))
            print("HITS:", chunk.get("hit_count"))
            print("TEXT:", chunk.get("text", "")[:500])

        print("=================================\n")
        return {
            "retrieval_results": kept,
            "low_confidence": low_confidence,
            "error": None,
        }

    except Exception as e:
        logger.exception("rerank_node failed")
        return {"error": f"Rerank failed: {str(e)}"}
