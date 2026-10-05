"""
Query Analysis Node (Retrieval Planner)

Single node in the policy agent's LangGraph workflow. Reads `user_query` (and,
optionally, `user_context`) from PolicyAgentState and returns a partial state
update with the retrieval plan: intent, policy_topic, enhanced_query,
search_queries, metadata_filters.

Metadata filter sourcing, in priority order:
  1. Trusted application context (state["user_context"]) -- e.g. the logged-in
     user's department/country from your auth/session layer. These bypass the
     LLM entirely and are applied directly, since they're facts your app
     already knows, not something to infer from text.
  2. LLM-inferred filters, but ONLY when the user explicitly names something
     in the query (see rules 2/3 in the prompt) -- and only when the value is
     grounded against known_metadata.py (real ingested values), never guessed.
  An explicit query mention overrides the context default for the same field.
  If you need context to be non-overridable (e.g. department is used for
  access control, not just search convenience), swap the merge order in
  query_enhancer_node so context always wins.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agents.brain_agents.policy_agent.state import PolicyAgentState

load_dotenv()

logger = logging.getLogger(__name__)
# NOTE: no logging.basicConfig() here -- this is a library/node module that
# gets imported into your app, not run standalone anymore. basicConfig()
# should be called once, by your app's entry point, not by every module that
# happens to log something -- calling it here could clobber logging config
# set up elsewhere in your FastAPI app.

# ---------------------------------------------------------
# Metadata configuration
# ---------------------------------------------------------

ALLOWED_METADATA_FIELDS = {
    "document_type",
    "document_title",
    "country",
    "department",
    "policy_id",
    "scope",
    "chunk_type",
    "section_title",
    "section_number",
    "parent_section",
    "version",
    "effective_date",
}

# Fields we actively GROUND against real ingested values (categorical,
# low-cardinality, and the kind of thing a user might name explicitly, e.g.
# "the India HR policy" or "the Attendance and Leave Policy"). Structural
# fields (section_number, parent_section, version, effective_date, scope,
# chunk_type, section_title) are intentionally NOT grounded here: they're
# rarely something a user states in free text, and per rules 2/3 below, most
# of these should only ever be set from trusted app context, not parsed out
# of the query.
GROUNDED_FIELDS = [
    "document_type",
    "document_title",
    "country",
    "department",
    "policy_id",
]


# ---------------------------------------------------------
# Structured LLM output
# ---------------------------------------------------------


class RetrievalPlan(BaseModel):
    intent: str = Field(description="User intent, for example policy_lookup")
    policy_topic: Optional[str] = Field(
        default=None, description="Policy topic relevant to the user query"
    )
    enhanced_query: str = Field(description="Primary semantic retrieval query")
    search_queries: List[str] = Field(description="2-4 alternative retrieval queries")
    metadata_filters: Dict[str, Any] = Field(
        description="Candidate metadata filters for retrieval"
    )


# ---------------------------------------------------------
# LLM (module-level -- built once, reused across every call)
# ---------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    max_output_tokens=8192,
    google_api_key=settings.GEMINI_API_KEY,
)

structured_llm = llm.with_structured_output(RetrievalPlan)


# ---------------------------------------------------------
# Known-value grounding
# ---------------------------------------------------------


def get_known_metadata_values(vector_store=None) -> Dict[str, List[str]]:
    """
    Pull the ACTUAL distinct metadata values currently ingested, so the LLM is
    grounded to real values instead of inventing plausible-sounding ones.

    Currently reads from known_metadata.py, updated by hand on each new
    document ingestion (see that file).

    TODO: once ingestion is automated and a vector store client exists, swap
    the body of this function for a live "distinct values" query -- nothing
    else in this file needs to change when you do that.
    """
    from app.agents.brain_agents.policy_agent.nodes.known_metadatavalues import KNOWN_METADATA_VALUES

    return KNOWN_METADATA_VALUES


def format_known_values_block(known_values: Dict[str, List[str]]) -> str:
    lines = []
    for field in GROUNDED_FIELDS:
        values = known_values.get(field) or []
        if values:
            lines.append(f"- {field}: {', '.join(values)}")
        else:
            lines.append(
                f"- {field}: (no values ingested yet -- do not set this field)"
            )
    return "\n".join(lines)


def get_trusted_context_filters(user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metadata filters supplied directly by the application (e.g. the logged-in
    user's department/country), not inferred by the LLM. These are
    authoritative -- no grounding check needed, since they come from your own
    trusted system rather than a guess. Only fields in GROUNDED_FIELDS are
    considered here, to keep this in sync with what the LLM is allowed to set.
    """
    return {
        field: user_context[field]
        for field in GROUNDED_FIELDS
        if user_context.get(field)
    }


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

RETRIEVAL_PLANNER_PROMPT = """
You are the Retrieval Planning Agent for an enterprise HR Policy RAG system.

Your responsibility is ONLY to create a retrieval plan.
Do NOT answer the user's question.

Analyze the user query and produce:

1. intent
2. policy_topic
3. enhanced_query
4. 2-4 alternative search_queries
5. metadata_filters

Available metadata fields:

- chunk_type
- country
- department
- document_title
- document_type
- effective_date
- parent_section
- policy_id
- scope
- section_number
- section_title
- version

Known values currently in the index (use these EXACT strings, never invent new ones):
{known_values_block}

If the user's query doesn't clearly map to one of the known values above for a
given field, OMIT that field entirely rather than guessing a plausible-sounding one.

Rules:

1. Do not invent metadata values. Only use values from the "Known values" list above.

2. Only generate a metadata filter when:
   - the user explicitly provides the information, OR
   - it is supplied through trusted application context.

3. Do not infer country, department, role, or other
   user attributes from the wording of the query. Only set these
   fields if the user names them explicitly (e.g. "the India HR policy").

4. Do not use these fields as semantic filters:
   - chunk_id
   - chunk_index
   - page_start
   - page_end
   - token_estimate
   - chunk_text

5. If the user is asking about an HR policy,
   document_type may be "HR_POLICY".

6. Do not over-filter.

7. Prefer broad enough filters to avoid eliminating
   relevant documents.

8. Generate exactly 2-4 search queries.

9. search_queries must cover DIFFERENT ASPECTS of the information need,
   never paraphrases of each other. Cover facets like: relevant leave
   types and eligibility, the application procedure and portal,
   advance-notice/deadline rules, and anything date-related the user
   mentioned. Paraphrase lists ("leave policy" / "time off policy") are
   a failure.

10. enhanced_query should be the strongest primary
    semantic retrieval query.

11. Do not answer the user's question.

12. A date mentioned as when the user wants leave/time off is NOT the
    same as effective_date (which describes when the policy DOCUMENT
    itself took effect, not when the user wants time off). Do not set
    effective_date from a leave date mentioned in the query.

USER QUERY:

{user_query}
"""


# ---------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------


def validate_metadata_filters(
    filters: Dict[str, Any], known_values: Dict[str, List[str]]
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    validated: Dict[str, Any] = {}
    dropped: Dict[str, str] = {}

    for key, value in filters.items():

        if key not in ALLOWED_METADATA_FIELDS:
            dropped[key] = f"{value!r} -> rejected (not an allowed metadata field)"
            continue

        if value is None or (isinstance(value, str) and not value.strip()):
            continue  # empty values are just omitted, nothing to log

        if key in known_values and value not in known_values[key]:
            dropped[key] = (
                f"{value!r} -> rejected (not in known ingested values: {known_values[key]})"
            )
            continue

        validated[key] = value

    if dropped:
        logger.warning("Dropped invalid/hallucinated metadata filters: %s", dropped)

    return validated, dropped


# ---------------------------------------------------------
# Query enhancement node
# ---------------------------------------------------------


def query_enhancer_node(state: PolicyAgentState) -> Dict[str, Any]:
    """
    LangGraph node. Reads state["user_query"] and, if present,
    state["user_context"]; returns a partial PolicyAgentState update.
    """

    user_query = state["user_query"]
    user_context = state.get("user_context") or {}

    known_values = get_known_metadata_values()

    prompt = RETRIEVAL_PLANNER_PROMPT.format(
        known_values_block=format_known_values_block(known_values),
        user_query=user_query,
    )

    try:
        result = structured_llm.invoke(prompt)

        llm_filters, dropped_filters = validate_metadata_filters(
            result.metadata_filters, known_values
        )

        # Trusted context is the default; an explicit query mention (already
        # validated above) overrides it for the same field. See module
        # docstring if you need the opposite precedence (context always wins).
        trusted_filters = get_trusted_context_filters(user_context)
        metadata_filters = {**trusted_filters, **llm_filters}

        if dropped_filters:
            logger.info(
                "query_analysis_node: dropped filters for query %r: %s",
                user_query,
                dropped_filters,
            )

        search_queries = [q.strip() for q in result.search_queries if q and q.strip()][
            :4
        ]
        print("#####################   Query Enhancer ########################")
        print(
            {
                "intent": result.intent,
                "policy_topic": result.policy_topic,
                "enhanced_query": result.enhanced_query.strip(),
                "search_queries": search_queries,
                "metadata_filters": metadata_filters,
                "error": None,
            }
        )
        print("#"*20)
        return {
            "intent": result.intent,
            "policy_topic": result.policy_topic,
            "enhanced_query": result.enhanced_query.strip(),
            "search_queries": search_queries,
            "metadata_filters": metadata_filters,
            "error": None,
        }

    except Exception as e:
        logger.exception("query_analysis_node failed for query: %r", user_query)
        return {"error": f"Retrieval planning failed: {str(e)}"}
