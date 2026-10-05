"""
PulseBrainAgent
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.brain_agents.Neo4j_agent.state import BrainAgentState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin Brain Agent"])


def _load_neo4j_agent():
    """Load the Neo4j graph only when its endpoint is used.

    Neo4j and Gemini are optional for the liveness endpoint. Importing their
    graph at module load time made the whole FastAPI application fail before
    it could serve health checks when configuration was incomplete.
    """
    from app.agents.brain_agents.Neo4j_agent.graph import neo4j_agent

    return neo4j_agent


def _load_policy_agent_graph():
    """Load the policy graph only when a policy endpoint is used."""
    from app.agents.brain_agents.policy_agent.graph import policy_agent_graph

    return policy_agent_graph


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Neo4jQueryRequest(BaseModel):
    user_query: str


class Neo4jAgentResponse(BaseModel):
    user_query: str
    cypher_query: str = ""
    error: Optional[str] = None
    node_results: Optional[List[Dict[str, Any]]] = None
    iteration_count: int = 0
    next_action: str = ""
    final_result: Optional[str] = None


class PolicyAgentRequest(BaseModel):
    user_query: str


class CitationOut(BaseModel):
    marker: Optional[str] = None
    document_title: Optional[str] = None
    version: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None


class EvidenceOut(BaseModel):
    rerank_score: Optional[float] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None


class ValidationOut(BaseModel):
    overall: Optional[str] = None
    issues: Optional[List[str]] = None
    refinement_would_help: Optional[bool] = None


class PolicyAgentResponse(BaseModel):
    user_query: str
    intent: Optional[str] = None
    enhanced_query: Optional[str] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    answer_confidence: Optional[Any] = None
    low_confidence: Optional[bool] = None
    citations: Optional[List[CitationOut]] = None
    validation: Optional[ValidationOut] = None
    refinement_attempts: int = 0
    error: Optional[str] = None
    evidence_count: int = 0
    evidence: List[EvidenceOut] = []


class TraceStep(BaseModel):
    node: str
    update: Dict[str, Any]


class PolicyAgentDebugResponse(BaseModel):
    user_query: str
    nodes_executed: List[str] = []
    execution_trace: List[TraceStep] = []
    final_answer: Optional[str] = None
    final_state: Dict[str, Any] = {}
    pipeline_healthy: bool = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _summarize_update(update: Dict[str, Any]) -> Dict[str, Any]:
    """Trim bulky fields so the per-node trace stays readable."""
    summary: Dict[str, Any] = {}
    for key, value in update.items():
        if key == "retrieval_results" and isinstance(value, list):
            summary[key] = {
                "count": len(value),
                "chunks": [
                    {
                        "rerank_score": c.get("rerank_score"),
                        "section_number": (c.get("metadata") or {}).get(
                            "section_number"
                        ),
                        "section_title": (c.get("metadata") or {}).get("section_title"),
                    }
                    for c in value
                ],
            }
        elif key == "refinement_log" and isinstance(value, list):
            summary[key] = [e.get("strategy_note") for e in value]
        else:
            summary[key] = value
    return summary


def _sanitize(state: Dict[str, Any]) -> Dict[str, Any]:
    """Make the whole state JSON-safe for the response."""
    safe: Dict[str, Any] = {}
    for key, value in state.items():
        try:
            json.dumps(value)
            safe[key] = value
        except (TypeError, ValueError):
            safe[key] = str(value)
    return safe


# --------------------------------------------------------------------------
# Neo4j agent (unchanged)
# --------------------------------------------------------------------------


@router.post("/generate-cypher-query", response_model=Neo4jAgentResponse)
async def get_neo4j_data(input_data: Neo4jQueryRequest):
    try:
        agent = _load_neo4j_agent()
    except Exception as e:
        logger.exception("Neo4j agent is not configured")
        raise HTTPException(
            status_code=503,
            detail=f"Neo4j agent is unavailable. Check backend/.env: {e}",
        ) from e

    try:
        result = await agent.ainvoke(
            {
                "user_query": input_data.user_query,
                "cypher_query": "",
                "error": None,
                "node_results": None,
                "iteration_count": 0,
                "next_action": "",
                "final_result": "",
            }
        )
        return {
            "user_query": result["user_query"],
            "cypher_query": result["cypher_query"],
            "error": result.get("error"),
            "node_results": result.get("node_results"),
            "iteration_count": result.get("iteration_count"),
            "next_action": result.get("next_action"),
            "final_result": result.get("final_result"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Policy agent — normal run (final merged state)
# --------------------------------------------------------------------------


@router.post("/policy_information", response_model=PolicyAgentResponse)
async def get_ans_policy(input_data: PolicyAgentRequest):
    """
    Runs the full pipeline:
    query_analysis -> hybrid_retrieval -> rerank -> policy_answer
    -> answer_validator (-> retrieval_refinement loop if needed) -> END
    """
    try:
        policy_agent_graph = _load_policy_agent_graph()
    except Exception as e:
        logger.exception("Policy agent is not configured")
        raise HTTPException(
            status_code=503,
            detail=f"Policy agent is unavailable. Check backend/.env: {e}",
        ) from e

    try:
        # Use the PRE-COMPILED instance. Do NOT call build_policy_agent_graph()
        # here — it's a factory and would recompile the graph per request.
        result = await policy_agent_graph.ainvoke({"user_query": input_data.user_query})

        validation = result.get("validation") or {}
        chunks = result.get("retrieval_results") or []

        return PolicyAgentResponse(
            user_query=result.get("user_query", input_data.user_query),
            intent=result.get("intent"),
            enhanced_query=result.get("enhanced_query"),
            metadata_filters=result.get("metadata_filters"),
            answer=result.get("answer"),
            answer_confidence=result.get("answer_confidence"),
            low_confidence=result.get("low_confidence"),
            citations=result.get("citations") or [],
            validation=(
                ValidationOut(
                    overall=validation.get("overall"),
                    issues=validation.get("issues"),
                    refinement_would_help=validation.get("refinement_would_help"),
                )
                if validation
                else None
            ),
            refinement_attempts=result.get("refinement_attempts", 0),
            error=result.get("error"),
            evidence_count=len(chunks),
            evidence=[
                EvidenceOut(
                    rerank_score=c.get("rerank_score"),
                    section_number=(c.get("metadata") or {}).get("section_number"),
                    section_title=(c.get("metadata") or {}).get("section_title"),
                )
                for c in chunks
            ],
        )
    except Exception as e:
        logger.exception("Policy agent run failed")
        raise HTTPException(status_code=500, detail=f"Policy agent failed: {e}")


# --------------------------------------------------------------------------
# Policy agent — debug run (node-by-node trace, mirrors graph.py's __main__)
# --------------------------------------------------------------------------


@router.post("/policy_information/debug", response_model=PolicyAgentDebugResponse)
async def debug_policy_agent(input_data: PolicyAgentRequest):
    """
    Same graph, but streams `stream_mode="updates"` so the response contains
    one entry per node executed. Use this to verify the wiring matches the
    pipeline docstring in graph.py.
    """
    nodes_executed: List[str] = []
    trace: List[Dict[str, Any]] = []
    final_state: Dict[str, Any] = {"user_query": input_data.user_query}

    try:
        policy_agent_graph = _load_policy_agent_graph()
        async for step in policy_agent_graph.astream(
            {"user_query": input_data.user_query},
            stream_mode="updates",
        ):
            for node_name, update in step.items():
                if node_name == "__end__" or not update:
                    continue
                nodes_executed.append(node_name)
                trace.append({"node": node_name, "update": _summarize_update(update)})
                final_state.update(update)

        validation = final_state.get("validation") or {}
        pipeline_healthy = (
            bool(final_state.get("answer"))
            and not final_state.get("error")
            and validation.get("overall") == "pass"
        )

        return PolicyAgentDebugResponse(
            user_query=input_data.user_query,
            nodes_executed=nodes_executed,
            execution_trace=trace,
            final_answer=final_state.get("answer"),
            final_state=_sanitize(final_state),
            pipeline_healthy=pipeline_healthy,
        )
    except Exception as e:
        logger.exception("Policy agent debug run failed")
        raise HTTPException(status_code=500, detail=f"Policy agent debug failed: {e}")
