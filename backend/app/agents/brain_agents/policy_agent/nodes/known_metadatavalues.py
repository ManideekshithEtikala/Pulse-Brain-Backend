"""
Registry of known metadata values currently ingested into the vector store.

This is manually maintained WHILE YOU'RE INGESTING ONE DOCUMENT AT A TIME.
Every time you ingest a new document, add its values here -- APPEND to the
lists, don't replace them. Each field is a list because multiple documents
can share a value (e.g. two policies can both be document_type "HR_POLICY").

Once you have a real ingestion pipeline and a vector store client wired up,
delete this file and replace get_known_metadata_values() in your node with a
live query against the vector store instead (e.g. a "distinct values per
field" query). Until then, this file IS your source of truth for grounding.
"""

KNOWN_METADATA_VALUES = {
    "document_type": ["HR_POLICY"],
    "document_title": [
        "Attendance and Leave Policy",
        "Probation and Confirmation Policy",
        "Code of Conduct Policy",
        "Pulse Seperation Policy",
        "Employee Onboarding Policy",
        "Pulse Posh Policy",
        "Pulse Recruitment Policy",
        "Pulse Training Policy",
    ],
    "country": ["India"],
    "department": ["Human Resources"],
    "policy_id": [
        "attendance-leave-policy",
        "probation-confirmation-policy",
        "code_of_conduct",
        "employee_onboarding_policy",
        "pulse_seperation_policy",
        "pulse_posh_policy",
        "pulse_recruitment_policy",
        "pulse_training_dev_policy",
    ],
}
