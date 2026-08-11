# Planted Flaws Registry

This file documents every intentional flaw planted in the knowledge base,
for grading traceability and for writing eval questions with known ground truth.

## 1. Stale Data
| Doc (outdated) | Doc (current) | What changed | Correct behavior |
|---|---|---|---|
| leave_policy_v1.md | leave_policy_v2.md | [e.g., sick leave went from 10 → 12 days] | System should cite v2, ignore v1 as source of truth, or explicitly note the version conflict if asked |
| expense_reimbursement_v1.md | expense_reimbursement_v2.md | [e.g., hotel cap ₹X → ₹Y] | Same — cite current version |

## 2. Contradictions
| Doc A | Doc B | Topic | Nature of contradiction |
|---|---|---|---|
| probation_policy.md | onboarding_checklist.md | Probation period length | [e.g., probation_policy.md says 6 months, onboarding_checklist.md says 90 days] |

> System should surface this conflict to the user, not silently pick one side.

## 3. Gaps (unanswerable)
| Topic | Which doc(s) were checked | Why it's missing |
|---|---|---|
| [e.g., Paternity leave] | leave_policy_v2.md, benefits_overview.md | Never mentioned anywhere in KB |
| [topic 2] | ... | ... |
| [topic 3] | ... | ... |

## 4. Near-miss content
| Doc | What it looks like it answers | What it actually doesn't cover | Risk |
|---|---|---|---|
| remote_work_zones.md | Looks like it answers "can I work from another country" | Only covers domestic remote zones, not international | High risk of confident hallucination if Researcher over-matches on keyword similarity |

<!-- ## Cross-reference to eval/questions.json
| Question ID | Flaw type tested |
|---|---|
| Q_ | stale data |
| Q_ | contradiction |
| Q_ | gap |
| Q_ | near-miss | -->