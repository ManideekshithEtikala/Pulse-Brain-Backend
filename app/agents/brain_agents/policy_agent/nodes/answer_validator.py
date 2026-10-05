"""
Answer Validator Node

Fourth node. Independently audits the generated answer against the evidence
it claims to be based on. Five checks: grounded, relevant, complete,
contradiction-free, current-policy. Produces a structured verdict that the
graph router uses to either END, or route into the retrieval refinement loop.

Design decisions:
  - Short-circuits WITHOUT an LLM call when the answer came from the
    deterministic not-found fallback (low_confidence / no citations) --
    there is nothing generated to validate.
  - Code-level pre-checks before the LLM: empty citations => auto-fail
    groundedness; multiple versions of the same policy_id among cited
    chunks => contradiction/current-policy risk flagged deterministically.
  - The verdict includes `refinement_would_help`, so the graph only spends
    a refinement cycle on failures retrieval can actually fix
    (relevance / completeness / contradiction), never on hallucination
    or stale-source failures.
  - FINAL GATE: if validation ultimately fails and no (further) refinement
    is possible, this node ships a safe answer: not-found template on
    groundedness failure, downgraded + caveated answer otherwise.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agents.brain_agents.policy_agent.state import PolicyAgentState
from app.agents.brain_agents.policy_agent.nodes.policy_agent import (
    build_evidence_block,
    NOT_FOUND_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Refined retrieval gets one more shot at a good answer. 1 keeps the worst
# case at 2 full generate+validate cycles (~acceptable latency/cost).
MAX_REFINEMENT_ATTEMPTS = 1


class ValidationVerdict(BaseModel):

    is_grounded: bool = Field(
        description=(
            "True if EVERY factual claim in the answer (numbers, limits, "
            "durations, conditions, deadlines) is supported by the cited "
            "evidence chunk. False if anything is invented, embellished, "
            "or cited to a chunk that does not contain it."
        )
    )
    is_relevant: bool = Field(
        description="True if the answer addresses what the user actually asked."
    )
    is_complete: bool = Field(
        description=(
            "True if the main aspects of the question are covered. NOTE: "
            "if the answer honestly declares that the policies do not "
            "specify some part (and that part is genuinely absent from the "
            "evidence), this still counts as complete -- transparent gaps "
            "are acceptable; silent ones are not."
        )
    )
    no_unresolved_contradiction: bool = Field(
        description=(
            "True unless evidence chunks disagree AND the answer silently "
            "picked one side. Presenting both versions with citations and "
            "advising HR counts as resolved."
        )
    )
    uses_current_policy: bool = Field(
        description=(
            "True if the answer relies only on the most recent policy "
            "version present in the evidence (compare version/effective_date "
            "in the evidence headers). If evidence contains several versions "
            "of the same policy, the answer must use the latest."
        )
    )
    refinement_would_help: bool = Field(
        description=(
            "True only if a DIFFERENT retrieval (new query phrasings or "
            "adjusted filters) could plausibly fix the issues -- e.g. wrong "
            "or incomplete evidence, mixed versions. False for groundedness "
            "failures (hallucination is a generation problem, not a "
            "retrieval problem) and for stale-source problems (ingestion)."
        )
    )
    overall: Literal["pass", "fail"]
    issues: List[str] = Field(
        description=(
            "Empty if pass. Otherwise specific, actionable problems, e.g. "
            "'answer states 15 days grace period but evidence [1] says 30 "
            "minutes', 'question also asked about carry-forward, evidence "
            "does not cover it', 'evidence mixes V.01 and V.02'."
        )
    )
    missing_information: Optional[str] = Field(
        default=None,
        description="What the evidence would need to contain to answer fully.",
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,  # auditing = maximum strictness
    max_output_tokens=2048,
    google_api_key=settings.GEMINI_API_KEY,
)
structured_llm = llm.with_structured_output(ValidationVerdict)


VALIDATOR_PROMPT = """
You are the Answer Validator for an enterprise HR policy assistant.
An earlier agent answered an employee's question using numbered policy
evidence. Your job is to AUDIT that answer against the evidence. You are
adversarial and precise: your reputation is ruined by letting one invented
number through, and equally by rejecting an honest, careful answer.

TODAY: {today}

EMPLOYEE QUESTION:
{user_query}

POLICY EVIDENCE (numbered; headers show policy, version, section):
{evidence_block}

ANSWER UNDER AUDIT:
{answer}

Declared answer confidence: {answer_confidence}

Audit each dimension:

1. GROUNDED — check claim by claim. Every quantity, condition, and
   requirement in the answer must exist, with the same value, in the cited
   evidence chunk. Common failures: rounding ("7 days" -> "about a week"),
   imported general HR knowledge, dropped qualifiers ("only once", "minimum
   1 year of service"), or a citation attached to a claim the chunk
   doesn't contain.

2. RELEVANT — does the answer answer THE question asked, not an adjacent
   one?

3. COMPLETE — are the main aspects covered? An honestly declared gap
   ("the policy does not specify X") is acceptable completeness. A silent
   gap is not.

4. CONTRADICTION — if two chunks conflict (often different versions of a
   policy), the answer must present both and advise HR, not silently
   choose.

5. CURRENT — compare version/effective_date in the evidence headers. If
   several versions of the same policy appear, the answer must rely on the
   latest.

Then set:
- overall: "pass" only if grounded AND relevant AND complete AND
  no_unresolved_contradiction AND uses_current_policy.
- issues: specific and quotable (cite chunk numbers). These drive the
  retrieval refinement loop, so write them as retrieval instructions:
  what evidence is missing or wrong, not style complaints.
- refinement_would_help per the field description.
"""


def _code_prechecks(
    answer_text: str, chunks: List[Dict[str, Any]]
) -> Optional[ValidationVerdict]:
    """
    Deterministic checks that need no LLM. Returns a fail verdict if one
    triggers, else None.
    """
    used_markers = [
        int(m) for m in __import__("re").findall(r"\[(\d{1,2})\]", answer_text)
    ]
    if not used_markers:
        return ValidationVerdict(
            is_grounded=False,
            is_relevant=False,
            is_complete=False,
            no_unresolved_contradiction=True,
            uses_current_policy=True,
            refinement_would_help=False,  # generation problem, not retrieval
            overall="fail",
            issues=["Answer contains no citation markers at all."],
        )

    # Same policy cited at multiple versions -> contradiction risk
    versions_by_policy: Dict[str, set] = {}
    for n in used_markers:
        if 1 <= n <= len(chunks):
            md = chunks[n - 1].get("metadata", {})
            versions_by_policy.setdefault(md.get("policy_id", "?"), set()).add(
                md.get("version", "?")
            )
    mixed = {p: v for p, v in versions_by_policy.items() if len(v) > 1}
    if mixed:
        # Not an auto-fail (the answer may have resolved it) — but recorded
        logger.warning("validator: mixed policy versions cited: %s", mixed)
    return None


def _finalize_failed_answer(
    state: PolicyAgentState, verdict: ValidationVerdict
) -> Dict[str, Any]:
    """
    Final gate: no (further) refinement will run. Ship something safe.
    """
    attempts = state.get("refinement_attempts", 0)
    exhausted = attempts >= MAX_REFINEMENT_ATTEMPTS

    if not exhausted and verdict.refinement_would_help:
        # A refinement cycle will run — leave the current answer untouched.
        return {}

    if not verdict.is_grounded:
        logger.error(
            "validator FINAL GATE: ungrounded answer detected after %d "
            "attempt(s) — replacing with deterministic not-found answer",
            attempts,
        )
        return {
            "answer": NOT_FOUND_TEMPLATE.format(query=state.get("user_query", "")),
            "citations": [],
            "answer_confidence": "not_found_in_evidence",
        }

    # Grounded but imperfect (incomplete / irrelevant parts / versions):
    # keep it, downgrade honestly, add one transparent caveat line.
    answer = state.get("answer") or ""
    if not answer.endswith(
        "For anything not covered above, please confirm with your HR team."
    ):
        answer += (
            "\n\nFor anything not covered above, please confirm with your HR team."
        )
    logger.warning(
        "validator FINAL GATE: shipping degraded answer after %d attempt(s); "
        "issues: %s",
        attempts,
        verdict.issues,
    )
    return {
        "answer": answer,
        "answer_confidence": "partially_answered",
    }


def answer_validator_node(state: PolicyAgentState) -> Dict[str, Any]:
    answer = state.get("answer")
    chunks = state.get("retrieval_results") or []

    # The deterministic not-found fallback produced this — nothing to audit.
    if state.get("low_confidence") or not chunks or not answer:
        return {
            "validation": {
                "overall": "pass",
                "note": "deterministic fallback answer — skipped audit",
            },
            "error": None,
        }

    precheck_fail = _code_prechecks(answer, chunks)
    if precheck_fail is not None:
        verdict = precheck_fail
    else:
        try:
            prompt = VALIDATOR_PROMPT.format(
                today=datetime.now().strftime("%A, %d %B %Y"),
                user_query=state.get("user_query", ""),
                evidence_block=build_evidence_block(chunks),
                answer=answer,
                answer_confidence=state.get("answer_confidence", "unknown"),
            )
            verdict = structured_llm.invoke(prompt)
        except Exception as e:
            logger.exception("answer_validator_node failed")
            # Fail OPEN-but-safe: treat as validation failure with no retry,
            # final gate decides. (Never crash the graph post-generation.)
            verdict = ValidationVerdict(
                is_grounded=False,
                is_relevant=False,
                is_complete=False,
                no_unresolved_contradiction=False,
                uses_current_policy=False,
                refinement_would_help=False,
                overall="fail",
                issues=[f"Validator itself errored: {e}"],
            )

    update: Dict[str, Any] = {
        "validation": verdict.model_dump(),
        "error": None,
    }
    update.update(_finalize_failed_answer(state, verdict))

    logger.info(
        "answer_validator_node: overall=%s issues=%s refine=%s attempts=%d",
        verdict.overall,
        verdict.issues,
        verdict.refinement_would_help,
        state.get("refinement_attempts", 0),
    )
    print("#"*20)
    print(update)
    return update
