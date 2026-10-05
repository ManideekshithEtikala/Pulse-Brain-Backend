"""
Retrieval Refinement Node (Loop Node)

Runs ONLY when validation failed and refinement_would_help was True.
Diagnoses WHY the evidence was insufficient (from validation.issues +
missing_information) and produces a corrected retrieval plan: revised
enhanced_query, revised search_queries, revised metadata_filters.

The revised plan is written back into the SAME state fields the original
plan used (enhanced_query / search_queries / metadata_filters), so the
existing hybrid_retrieval -> rerank -> policy_answer nodes run unchanged
on the second cycle. Filters are re-validated against known ingested
values with the exact same validator used by query_analysis_node — the
LLM never gets to invent values on the retry either.
"""

import logging
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agents.brain_agents.policy_agent.state import PolicyAgentState
from app.agents.brain_agents.policy_agent.nodes.answer_validator import (
    MAX_REFINEMENT_ATTEMPTS,
)
from app.agents.brain_agents.policy_agent.nodes.query_analysis_node import (
    format_known_values_block,
    get_known_metadata_values,
    validate_metadata_filters,
    get_trusted_context_filters,
)

logger = logging.getLogger(__name__)


class RefinedPlan(BaseModel):
    revised_enhanced_query: str = Field(
        description="Corrected primary retrieval query targeting the gaps."
    )
    revised_search_queries: List[str] = Field(
        description="2-4 alternative phrasings targeting the missing evidence."
    )
    revised_metadata_filters: Dict[str, Any] = Field(
        description=(
            "Adjusted filters. Relax filters that were too narrow; add a "
            "version filter ONLY to pin the latest version when mixed "
            "versions caused contradictions. Omit fields that should stay broad."
        )
    )
    strategy_note: str = Field(
        description="One sentence: what was wrong and how this plan fixes it."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_output_tokens=2048,
    google_api_key=settings.GEMINI_API_KEY,
)
structured_llm = llm.with_structured_output(RefinedPlan)

REFINEMENT_PROMPT = """
You are the Retrieval Refinement Agent. A previous retrieval attempt for the
question below produced evidence that FAILED validation. Diagnose the
retrieval problem and output a CORRECTED retrieval plan.

EMPLOYEE QUESTION:
{user_query}

PREVIOUS PLAN:
- enhanced_query: {prev_enhanced}
- search_queries: {prev_search}
- metadata_filters: {prev_filters}

VALIDATION ISSUES FOUND (these are the diagnosis — read carefully):
{issues}

Missing information that the evidence lacked:
{missing}

Known metadata values in the index (use EXACT strings; omit a field rather
than guess):
{known_values_block}

Rules:
1. Rewrite the queries to target the SPECIFIC gaps in the issues — not a
   paraphrase of the old queries. Include the policy vocabulary an HR
   document would actually use (e.g. if the gap is about leave
   carry-forward, search "carry forward of leave balance encashment").
2. If previous filters were too narrow (evidence missing whole topics),
   relax or drop them. If the issue was mixed policy versions, pin the
   latest known version — and only from the known values above.
3. Never invent metadata values. Omit unknown fields.
4. 2-4 search queries, each a genuinely different formulation.
"""


def retrieval_refinement_node(state: PolicyAgentState) -> Dict[str, Any]:
    attempts = state.get("refinement_attempts", 0)
    validation = state.get("validation") or {}

    # Hard loop guard — the router should prevent this, but never trust it.
    if attempts >= MAX_REFINEMENT_ATTEMPTS:
        logger.warning("refinement: attempt cap reached — skipping")
        return {"error": None}

    known_values = get_known_metadata_values()

    try:
        result = structured_llm.invoke(
            REFINEMENT_PROMPT.format(
                user_query=state.get("user_query", ""),
                prev_enhanced=state.get("enhanced_query", ""),
                prev_search=state.get("search_queries", []),
                prev_filters=state.get("metadata_filters", {}),
                issues="\n".join(f"- {i}" for i in validation.get("issues", []))
                or "- (unspecified)",
                missing=validation.get("missing_information") or "(none stated)",
                known_values_block=format_known_values_block(known_values),
            )
        )

        validated_filters, dropped = validate_metadata_filters(
            result.revised_metadata_filters, known_values
        )
        # Trusted context re-applied on the retry: same precedence as the
        # original plan (explicit validated mention overrides context).
        metadata_filters = {
            **get_trusted_context_filters(state.get("user_context") or {}),
            **validated_filters,
        }

        log_entry = {
            "attempt": attempts + 1,
            "strategy_note": result.strategy_note,
            "dropped_filters": dropped,
            "issues_addressed": validation.get("issues", []),
        }
        history = list(state.get("refinement_log") or [])
        history.append(log_entry)

        logger.info(
            "retrieval_refinement_node: attempt %d — %s",
            attempts + 1,
            result.strategy_note,
        )
        print(
            {
                "enhanced_query": result.revised_enhanced_query.strip(),
                "search_queries": [
                    q.strip() for q in result.revised_search_queries if q and q.strip()
                ][:4],
                "metadata_filters": metadata_filters,
                "refinement_attempts": attempts + 1,
                "refinement_log": history,
                "error": None,
            }
        )

        return {
            "enhanced_query": result.revised_enhanced_query.strip(),
            "search_queries": [
                q.strip() for q in result.revised_search_queries if q and q.strip()
            ][:4],
            "metadata_filters": metadata_filters,
            "refinement_attempts": attempts + 1,
            "refinement_log": history,
            "error": None,
        }

    except Exception as e:
        logger.exception("retrieval_refinement_node failed")
        return {"error": f"Retrieval refinement failed: {str(e)}"}
