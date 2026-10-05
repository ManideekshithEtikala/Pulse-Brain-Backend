"""
Hybrid Retrieval Node

Second node in the policy agent's LangGraph workflow. Consumes the plan from
query_analysis_node (enhanced_query, search_queries, metadata_filters) and
returns state["retrieval_results"]: the final, fused top-K chunks.

What changed vs. the original script this was built from:
  - metadata_filters is now ACTUALLY applied. The original script hardcoded
    namespace="attendance-leave-policy" and never touched metadata_filters at
    all -- the entire grounding/validation pipeline from query_analysis_node
    never reached retrieval.
  - Namespace resolution: your ingestion puts each document in its own
    Pinecone namespace named after its policy_id. If metadata_filters pins
    down policy_id (user named the policy explicitly), we search only that
    namespace. Otherwise we search every currently-known namespace (from
    known_metadata.py) and merge results by score, since we don't yet know
    which document is relevant -- this is exactly the case your "leave on
    25th September" query hits, and why an empty metadata_filters there
    doesn't mean "no filtering", it means "search everything, let ranking
    decide".
  - BM25 is not namespace-partitioned (one flat index over everything), so
    the full metadata_filters (including policy_id) is applied client-side
    after an oversampled BM25 pull, then re-ranked contiguously before RRF.
  - RRF is generalized from a hardcoded (bm25, vector) pair to an arbitrary
    number of ranked lists, so search_queries (the 2-4 alternate phrasings
    from query_analysis_node) actually get used, not just enhanced_query.
  - Pinecone client, embedding model, and BM25 retriever are built ONCE at
    module load, not per-call -- this runs as a repeatedly-invoked graph
    node, not a one-shot script, so re-loading a SentenceTransformer model
    on every user query would add multi-second latency to every request.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


from RAG_ingestion_policies.bm25_index import BM25Retriever
from app.core.config import settings
from app.agents.brain_agents.policy_agent.state import PolicyAgentState
from app.agents.brain_agents.policy_agent.nodes.known_metadatavalues import KNOWN_METADATA_VALUES
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

PINECONE_INDEX_NAME = settings.PINECONE_POLICY_INDEX_NAME

VECTOR_TOP_K = 15
BM25_TOP_K = 15
BM25_OVERSAMPLE_FACTOR = 5  # widen the BM25 pull when a metadata filter is active
RRF_K = 60
RERANK_CANDIDATES = 20
MAX_QUERY_VARIANTS = 4  # enhanced_query + up to 2 of the alternate search_queries


# ---------------------------------------------------------
# Module-level singletons -- built once, reused across every call
# ---------------------------------------------------------

PINECONE_API_KEY = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

bm25_retriever = BM25Retriever()


# ---------------------------------------------------------
# Namespace resolution
# ---------------------------------------------------------


def resolve_namespaces(metadata_filters: Dict[str, Any]) -> List[str]:
    """
    Decide which Pinecone namespace(s) to search this request.
      - policy_id present in metadata_filters -> that one namespace only.
      - otherwise -> every currently-ingested namespace (from known_metadata.py),
        since we don't know which document the query is actually about.
    """
    policy_id = metadata_filters.get("policy_id")
    if policy_id:
        return [policy_id]
    return list(KNOWN_METADATA_VALUES.get("policy_id", []))


# ---------------------------------------------------------
# Vector search
# ---------------------------------------------------------


def build_pinecone_filter(metadata_filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert our plain {field: value} filters into Pinecone's filter syntax.
    policy_id is excluded here since it's already enforced via namespace
    selection -- no need to also filter on it inside the namespace.
    """
    fields = {k: v for k, v in metadata_filters.items() if k != "policy_id"}
    if not fields:
        return None
    return {key: {"$eq": value} for key, value in fields.items()}


def vector_search(
    query_vector: List[float],
    namespaces: List[str],
    metadata_filters: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Query one or more namespaces with the same filter and merge the results.
    Since every namespace shares the same embedding model/metric, raw scores
    are directly comparable across namespaces -- sort once, then assign fresh
    contiguous ranks over the merged set.
    """
    pinecone_filter = build_pinecone_filter(metadata_filters)

    all_matches: List[Dict[str, Any]] = []

    for ns in namespaces:
        response = index.query(
            namespace=ns,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter,
        )
        for match in response["matches"]:
            all_matches.append(
                {
                    "id": match["id"],
                    "vector_score": match["score"],
                    "text": match["metadata"].get("chunk_text", ""),
                    "metadata": match["metadata"],
                    "namespace": ns,
                }
            )

    all_matches.sort(key=lambda m: m["vector_score"], reverse=True)
    top_matches = all_matches[:top_k]

    for rank, m in enumerate(top_matches, start=1):
        m["rank"] = rank

    return top_matches


# ---------------------------------------------------------
# BM25 search
# ---------------------------------------------------------


def bm25_search_filtered(
    query_text: str, metadata_filters: Dict[str, Any], top_k: int
) -> List[Dict[str, Any]]:
    """
    BM25Retriever is a single flat index across every ingested document (not
    namespace-partitioned like Pinecone), so the FULL metadata_filters
    (including policy_id) must be applied here -- Pinecone got policy_id for
    free via namespace selection, BM25 doesn't.

    We don't assume BM25Retriever supports a native filter param -- instead we
    oversample candidates and filter/re-rank client-side, which works
    regardless of what it supports internally. If BM25Retriever DOES support
    native filtering, push metadata_filters into bm25_retriever.search()
    directly instead for efficiency.
    """
    oversample_k = top_k * BM25_OVERSAMPLE_FACTOR if metadata_filters else top_k
    raw_results = bm25_retriever.search(query_text, top_k=oversample_k)

    if metadata_filters:
        raw_results = [
            r
            for r in raw_results
            if all(
                r.get("metadata", {}).get(k) == v for k, v in metadata_filters.items()
            )
        ]

    trimmed = raw_results[:top_k]

    # Re-rank contiguously after filtering so RRF ranks reflect the filtered set,
    # not the original pre-filter positions.
    for rank, r in enumerate(trimmed, start=1):
        r["rank"] = rank

    return trimmed


# ---------------------------------------------------------
# Reciprocal Rank Fusion (generalized to N ranked lists)
# ---------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Dict[str, Any]]], k: int = 60
) -> List[Dict[str, Any]]:
    """
    Fuse an arbitrary number of ranked lists (one per query-variant x
    retrieval-method combination) instead of a hardcoded (bm25, vector) pair.
    Each list contributes 1/(k+rank) to a chunk's total score; a chunk that
    shows up near the top across multiple query phrasings AND both retrieval
    methods naturally floats to the top.
    """
    chunks: Dict[str, Dict[str, Any]] = {}

    for source_label, results in ranked_lists.items():
        for result in results:
            chunk_id = result["id"]
            rank = result["rank"]
            contribution = 1 / (k + rank)

            if chunk_id not in chunks:
                chunks[chunk_id] = {
                    "id": chunk_id,
                    "text": result["text"],
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "source_ranks": {},  # e.g. {"vector_q1": 2, "bm25_q2": 5}
                    "hit_count": 0,
                }

            chunks[chunk_id]["rrf_score"] += contribution
            chunks[chunk_id]["source_ranks"][source_label] = rank
            chunks[chunk_id]["hit_count"] += 1

    return sorted(chunks.values(), key=lambda c: c["rrf_score"], reverse=True)


# ---------------------------------------------------------
# Node
# ---------------------------------------------------------


def hybrid_retrieval_node(state: PolicyAgentState) -> Dict[str, Any]:
    """
    LangGraph node. Reads state["enhanced_query"], state["search_queries"],
    and state["metadata_filters"] (all produced by query_analysis_node);
    returns a partial PolicyAgentState update with retrieval_results.
    """

    enhanced_query = state.get("enhanced_query")
    if not enhanced_query:
        return {
            "error": "hybrid_retrieval_node: state['enhanced_query'] is missing "
            "-- run query_analysis_node first."
        }

    search_queries = state.get("search_queries") or []
    metadata_filters = state.get("metadata_filters") or {}
    raw_query = state.get("user_query") or enhanced_query
    query_variants = [raw_query, enhanced_query] + [
        q for q in search_queries if q not in (raw_query, enhanced_query)
    ]
    query_variants = query_variants[:MAX_QUERY_VARIANTS]   # maybe bump to 4

    try:
        namespaces = resolve_namespaces(metadata_filters)

        if not namespaces:
            logger.warning(
                "hybrid_retrieval_node: no known namespaces to search "
                "(has anything been ingested into known_metadata.py yet?)"
            )
            return {"retrieval_results": [], "error": None}

        ranked_lists: Dict[str, List[Dict[str, Any]]] = {}

        for i, query_text in enumerate(query_variants, start=1):
            query_vector = embedding_model.encode(query_text).tolist()

            ranked_lists[f"vector_q{i}"] = vector_search(
                query_vector=query_vector,
                namespaces=namespaces,
                metadata_filters=metadata_filters,
                top_k=VECTOR_TOP_K,
            )

            ranked_lists[f"bm25_q{i}"] = bm25_search_filtered(
                query_text=query_text,
                metadata_filters=metadata_filters,
                top_k=BM25_TOP_K,
            )

        hybrid_results = reciprocal_rank_fusion(ranked_lists, k=RRF_K)
        final_results = hybrid_results[:RERANK_CANDIDATES]

        logger.info(
            "hybrid_retrieval_node: %d query variant(s) x %d namespace(s) -> "
            "%d unique chunks -> top %d returned",
            len(query_variants),
            len(namespaces),
            len(hybrid_results),
            len(final_results),
        )
        print("####################.  hybird retrevials #################")
        print({"retrieval_results": final_results[:1], "error": None})
        print("#"*20)
        return {"retrieval_results": final_results, "error": None}

    except Exception as e:
        logger.exception("hybrid_retrieval_node failed for query: %r", enhanced_query)
        return {"error": f"Hybrid retrieval failed: {str(e)}"}
