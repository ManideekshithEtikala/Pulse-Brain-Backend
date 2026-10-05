"""
Policy Answer Node (Generation)

Final node in the policy agent workflow. Consumes the reranked, score-floored
chunks in state["retrieval_results"] and produces the user-facing answer.

Design decisions:
  - DETERMINISTIC FALLBACK: if rerank returned nothing usable (empty results
    or low_confidence=True), we do NOT call the LLM at all. Generating an
    answer from an empty context is the #1 hallucination invite; the fallback
    is a fixed template that honestly says "not found" and routes to HR.
  - CITATIONS BY MARKER, VALIDATED BY CODE: the LLM writes inline [1][2]
    markers referring to the numbered evidence chunks. We extract the markers
    it actually used and build the structured citations list from chunk
    metadata ourselves. Invalid (out-of-range) markers are stripped + logged;
    an answer with zero citations is logged as a quality warning.
  - SELF-REPORTED CONFIDENCE: the model labels the answer
    fully_answered / partially_answered / not_found_in_evidence. Combined
    with citation coverage, this gives the API layer its confidence signal.
  - TODAY'S DATE is injected so the model can anchor relative statements
    about leave dates (e.g. "25 Sep 2025 is a Thursday"), while complex
    date arithmetic is explicitly discouraged.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agents.brain_agents.policy_agent.state import PolicyAgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Structured LLM output
# ---------------------------------------------------------


class PolicyAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Final user-facing answer in plain professional language, "
            "with inline [n] citation markers referring to the numbered "
            "policy evidence chunks."
        )
    )
    answer_confidence: Literal[
        "fully_answered", "partially_answered", "not_found_in_evidence"
    ] = Field(
        description=(
            "fully_answered: evidence covers the question. "
            "partially_answered: evidence covers part of it. "
            "not_found_in_evidence: evidence does not address the question."
        )
    )
    missing_info: Optional[str] = Field(
        default=None,
        description=(
            "If partially_answered or not_found_in_evidence: the specific "
            "information the policy evidence did not contain."
        ),
    )


# ---------------------------------------------------------
# LLM (module-level — built once, reused across every call)
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.1,  # lower than planning: factual fidelity > creativity
    max_output_tokens=2048,
    google_api_key=settings.GEMINI_API_KEY,
)

structured_llm = llm.with_structured_output(PolicyAnswer)


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

POLICY_ANSWER_PROMPT = """
You are the Policy Answer Agent of an enterprise HR policy assistant.
Employees ask questions about company policies. You answer using ONLY the
policy evidence provided below. Today's date is {today}.

EMPLOYEE QUESTION:
{user_query}

RESPONSE CONTEXT:
- Detected intent: {intent}
- Policy topic: {policy_topic}
{user_context_block}

POLICY EVIDENCE (the only source of truth, numbered for citation):
{evidence_block}

RESPONSE RULES:

Grounding (most important):
1. Use ONLY the evidence above. Never add rules, numbers, limits, or
   conditions from your own knowledge — if it is not in the evidence,
   it does not exist.
2. Attach a citation marker [n] to every policy claim, referring to the
   evidence chunk it came from. Place the marker immediately after the
   sentence or bullet it supports, e.g. "...30 minutes per month [1]."
3. Quote every number, duration, limit, threshold, and deadline EXACTLY
   as written in the evidence. Never round, estimate, or compute
   entitlements the evidence does not state.
4. If the evidence answers the question only partially, answer the part
   it covers and then plainly state what the policies do not specify
   (also fill missing_info). Never paper over a gap.
5. If two evidence chunks contradict each other, present both statements
   with their citations and advise confirming with HR — never silently
   pick one.
6. A leave/date the employee mentions is NOT related to the effective_date
   metadata — that field describes when the policy document itself took
   effect, not the employee's plans.

Style:
7. Write in plain, professional language any employee understands.
   Briefly translate policy jargon (e.g. "LOP" -> "loss of pay"), keeping
   the official term in brackets where useful.
8. Structure: start with a one-sentence direct answer. Then a short
   "What this means for you" section or a bullet list of the key
   conditions/steps. If the question is procedural (applying for leave,
   claiming something), give the steps in order: what to do, whom to
   inform, by when, and through which portal.
9. Keep it tight: about 120 words for simple lookups, up to 250 for
   procedural answers. No preamble ("Based on the provided..."), no
   closing filler, and never mention chunks, evidence, retrieval, or
   this system — just answer naturally as the policy assistant.
10. If the intent indicates the employee wants to DO something (take
    leave, apply, claim), end with the concrete next action.

Honesty:
11. If the evidence does not address the question at all, set
    answer_confidence to "not_found_in_evidence", reply briefly and
    politely that the policy documents do not appear to cover it, and
    recommend contacting HR.
"""

NOT_FOUND_TEMPLATE = (
    "I couldn't find anything in the current HR policy documents that answers "
    'your question: "{query}".\n\n'
    "This usually means either the policy doesn't cover it, or it's worded "
    "differently than the question. Please check with your Location HR / "
    "Corporate HR team — they can confirm the exact rule or point you to the "
    "right document."
)


# ---------------------------------------------------------
# Evidence assembly + citation post-processing
# ---------------------------------------------------------


def build_evidence_block(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        md = chunk.get("metadata", {})
        header = (
            f"[{i}] Source: {md.get('document_title', 'Unknown policy')} "
            f"(Version {md.get('version', '?')}) — "
            f"Section {md.get('section_number', '?')} "
            f"{md.get('section_title', '')}".rstrip()
        )
        lines.append(header)
        lines.append((chunk.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines)


def format_user_context_block(user_context: Dict[str, Any]) -> str:
    if not user_context:
        return ""
    parts = [
        f"- {field} = {user_context[field]}"
        for field in ("department", "country", "role", "employee_type")
        if user_context.get(field)
    ]
    if not parts:
        return ""
    return (
        "Trusted employee context from the HR system (authoritative; use it "
        "to tailor wording only — never as a source of policy content):\n"
        + "\n".join(parts)
    )


CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def clean_and_extract_citations(
    answer_text: str, n_chunks: int
) -> tuple[str, List[int], List[int]]:
    """
    Strip out-of-range [n] markers, return the cleaned answer plus the
    cited chunk numbers in order of first appearance.
    """
    used_order: List[int] = []
    invalid: List[int] = []

    def _repl(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_chunks:
            if n not in used_order:
                used_order.append(n)
            return match.group(0)
        invalid.append(n)
        return ""

    cleaned = CITATION_RE.sub(_repl, answer_text)
    return cleaned, used_order, invalid


def build_citations_list(
    chunks: List[Dict[str, Any]], used_order: List[int]
) -> List[Dict[str, Any]]:
    citations = []
    for n in used_order:
        chunk = chunks[n - 1]
        md = chunk.get("metadata", {})
        citations.append(
            {
                "marker": f"[{n}]",
                "policy_id": md.get("policy_id"),
                "document_title": md.get("document_title"),
                "document_code": md.get("document_code"),
                "version": md.get("version"),
                "section_number": md.get("section_number"),
                "section_title": md.get("section_title"),
                "breadcrumb": md.get("breadcrumb"),
                "effective_date": md.get("effective_date"),
                "chunk_id": md.get("chunk_id", chunk.get("id")),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
    return citations


# ---------------------------------------------------------
# Node
# ---------------------------------------------------------


def policy_answer_node(state: PolicyAgentState) -> Dict[str, Any]:
    retrieval_results = state.get("retrieval_results") or []
    user_query = state.get("user_query", "")

    # ---- Deterministic fallback: NEVER generate from an empty context ----
    if state.get("low_confidence") or not retrieval_results:
        logger.warning(
            "policy_answer_node: no usable evidence for query %r — "
            "returning deterministic not-found answer (no LLM call)",
            user_query,
        )
        return {
            "answer": NOT_FOUND_TEMPLATE.format(query=user_query),
            "citations": [],
            "answer_confidence": "not_found_in_evidence",
            "error": None,
        }

    try:
        prompt = POLICY_ANSWER_PROMPT.format(
            today=datetime.now().strftime("%A, %d %B %Y"),
            user_query=user_query,
            intent=state.get("intent", "unknown"),
            policy_topic=state.get("policy_topic") or "(not identified)",
            user_context_block=format_user_context_block(
                state.get("user_context") or {}
            ),
            evidence_block=build_evidence_block(retrieval_results),
        )

        result = structured_llm.invoke(prompt)

        cleaned_answer, used_order, invalid = clean_and_extract_citations(
            result.answer, len(retrieval_results)
        )
        if invalid:
            logger.warning(
                "policy_answer_node: stripped out-of-range citation markers %s",
                invalid,
            )
        if not used_order:
            logger.warning(
                "policy_answer_node: answer contains no citation markers "
                "— treating as low citation confidence for query %r",
                user_query,
            )

        logger.info(
            "policy_answer_node: confidence=%s, cited %d of %d chunks",
            result.answer_confidence,
            len(used_order),
            len(retrieval_results),
        )
        print("############### final answer #############")
        print(
            {
                "answer": cleaned_answer,
                "citations": build_citations_list(retrieval_results, used_order),
                "answer_confidence": result.answer_confidence,
                "error": None,
            }
        )
        print("#"*20)
        return {
            "answer": cleaned_answer,
            "citations": build_citations_list(retrieval_results, used_order),
            "answer_confidence": result.answer_confidence,
            "error": None,
        }

    except Exception as e:
        logger.exception("policy_answer_node failed for query: %r", user_query)
        return {"error": f"Answer generation failed: {str(e)}"}
