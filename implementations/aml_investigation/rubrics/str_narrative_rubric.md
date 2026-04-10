You must emit exactly the following metrics and no others:

1. factual_grounding
   - 1 only if the narrative contains only facts supported by the candidate output and does not invent intent or unsupported details
   - else 0

2. regulatory_clarity
   - 1 only if the narrative is concise, chronological, and readable for an AML investigator or regulator
   - else 0

3. internal_consistency
   - 1 only if the narrative is consistent with is_laundering, pattern_type, and flagged_transaction_ids
   - else 0

4. actionability
   - 1 only if the narrative clearly explains why a filing is or is not recommended
   - else 0