# Planted Flaws — NimbusWorks Technologies Knowledge Base

This knowledge base contains 12 HR policy documents for a fictional company, **NimbusWorks Technologies Pvt. Ltd.** The following flaws were planted deliberately to test the Research Desk agent's grounding, conflict-handling, and abstention behavior.

## 1. Gaps
Questions with **no answer anywhere** in the knowledge base:
1. **Paternity/maternity leave duration** — not mentioned in any document (leave_policy_v1.md, leave_policy_v2.md, or benefits_overview.md)
2. **Total number of employees at the company** — not mentioned in any document
3. **Notice period during probation extension** — probation_policy.md covers standard probation termination (15 days) but does not address notice period if probation itself is extended

## 2. Contradictions
At least 3 places where two documents disagree:
1. **Sick leave days** — `leave_policy_v1.md` (Jan 2023) says 10 days; `leave_policy_v2.md` (Mar 2024) says 12 days
2. **Hotel accommodation limit** — `expense_reimbursement_v1.md` (2022) says ₹3,000/night; `expense_reimbursement_v2.md` (2024) says ₹4,500/night
3. **Probation period length** — `probation_policy.md` (Mar 2023) says 3 months; `onboarding_checklist.md` (May 2023) says 6 months. Note: these are two independently-dated documents that were never reconciled — this is an intentional real-world-style contradiction, not resolved by recency alone (onboarding_checklist.md is dated *after* probation_policy.md, so a naive "newest wins" heuristic would pick the wrong answer here — the correct system behavior is to surface the conflict, not silently pick the newer doc).

## 3. Stale Data
Documents dated older than others, where the newer one should win:
1. `leave_policy_v1.md` (Jan 2023) superseded by `leave_policy_v2.md` (Mar 2024) — sick leave provision
2. `expense_reimbursement_v1.md` (2022) superseded by `expense_reimbursement_v2.md` (2024) — hotel/conveyance/meal limits
3. `benefits_overview.md` (Jul 2023) references "SecureLife Insurance Co." as the current provider, but `code_of_conduct.md` (Nov 2022, technically an *older* doc) contains a forward-looking note claiming the insurance provider was migrated "as of late 2023" — a subtle inconsistency where an older document references a change that postdates it. This is meant to test whether the agent naively trusts doc age vs. actually reasoning about content dates.

## 4. Near-miss Content
Text that looks relevant to a question but actually answers something different:
1. **`wfh_policy.md`** discusses WFH *eligibility criteria* for existing office-based employees (keyword: "remote"), but a question about which **cities/zones support full-remote hiring** should instead be answered by **`remote_work_zones.md`**. A naive retriever will likely surface wfh_policy.md due to keyword overlap ("remote") even though it doesn't answer the zone question.
2. **`code_of_conduct.md`** mentions a "notice period" that is waived in cases of *misconduct termination*, but a general question like "what is the notice period?" should be answered by **`exit_offboarding_policy.md`** (standard resignation = 60 days). These are both "notice period" mentions but refer to different scenarios entirely.

## Document Index
| File | Date | Role |
|---|---|---|
| leave_policy_v1.md | Jan 2023 | Superseded |
| leave_policy_v2.md | Mar 2024 | Current |
| wfh_policy.md | Jun 2023 | Near-miss source |
| expense_reimbursement_v1.md | Apr 2022 | Superseded |
| expense_reimbursement_v2.md | Feb 2024 | Current |
| onboarding_checklist.md | May 2023 | Contradiction source |
| benefits_overview.md | Jul 2023 | Stale reference source |
| probation_policy.md | Mar 2023 | Contradiction source |
| remote_work_zones.md | Jan 2024 | Near-miss correct-answer source |
| exit_offboarding_policy.md | Aug 2023 | Near-miss correct-answer source |
| performance_review_policy.md | Apr 2023 | Clean (no planted flaw) |
| code_of_conduct.md | Nov 2022 | Near-miss + stale-reference source |