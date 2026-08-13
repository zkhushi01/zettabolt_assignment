"""
questions.py
The 15-question eval set: 6 single-doc, 4 multi-hop, 3 unanswerable,
2 contradictory. Every question and reference_answer is grounded directly in
docs/planted_flaws.md and docs/README.md's flaw registry -- not invented --
so "correct" has an actual ground truth to check against, and every category
maps to one of the KB's deliberately planted failure modes:

- single_doc / multi_hop: baseline correctness, and (for a couple of the
  multi-hop ones) resistance to the near-miss traps documented in the
  registry (e.g. "notice period" meaning two different things in
  exit_offboarding_policy.md vs code_of_conduct.md).
- unanswerable: the registry's 3 real gaps -- nothing in the KB answers
  these, so the only correct behavior is an honest refusal, never a guess.
- contradictory: the registry's documented cross-doc contradictions, where
  correct behavior is surfacing both values/sources, not silently picking
  one (including probation length, which the registry specifically notes is
  NOT resolved by "newest doc wins" -- onboarding_checklist.md is dated
  after probation_policy.md but that doesn't make its number the right one).
"""

QUESTIONS = [
    # ---- single-doc (6) ----
    {
        "id": "Q01",
        "category": "single_doc",
        "question": "What is the annual casual leave entitlement?",
        "reference_answer": "Employees are entitled to 8 days of casual leave per calendar year (leave_policy_v2.md). This figure is unchanged between the old and current leave policy.",
    },
    {
        "id": "Q02",
        "category": "single_doc",
        "question": "How many public holidays are employees entitled to per year?",
        "reference_answer": "Employees are entitled to 10 public holidays per year, per the calendar published annually by HR (leave_policy_v2.md).",
    },
    {
        "id": "Q03",
        "category": "single_doc",
        "question": "How often are performance reviews conducted, and what are the two review periods called?",
        "reference_answer": "Performance reviews are conducted twice a year: an H1 Review covering January-June (conducted in July) and an H2 Review covering July-December (conducted in January) (performance_review_policy.md).",
    },
    {
        "id": "Q04",
        "category": "single_doc",
        "question": "How many days per week can an approved employee work from home?",
        "reference_answer": "Approved employees may work from home up to 2 days per week (wfh_policy.md). Full-time WFH requires additional sign-off and is only granted in exceptional circumstances.",
    },
    {
        "id": "Q05",
        "category": "single_doc",
        "question": "What is the earned leave accrual rate and the maximum carry-forward limit?",
        "reference_answer": "Employees accrue 1.5 days of earned leave per month of service, capped at 18 days per year, and may carry forward up to a maximum of 30 days (leave_policy_v2.md).",
    },
    {
        "id": "Q06",
        "category": "single_doc",
        "question": "What documents must a new hire submit before their start date?",
        "reference_answer": "Before joining, a new hire must submit ID proof, address proof, and educational certificates (onboarding_checklist.md, Pre-Joining section).",
    },
    # ---- multi-hop (4) ----
    {
        "id": "Q07",
        "category": "multi_hop",
        "question": "What is the earned leave accrual rate per month, and what is the standard notice period for voluntary resignation?",
        "reference_answer": "Earned leave accrues at 1.5 days per month of service (leave_policy_v2.md). The standard notice period for voluntary resignation is 60 days from the date of resignation (exit_offboarding_policy.md). These are two independent facts from two different documents.",
    },
    {
        "id": "Q08",
        "category": "multi_hop",
        "question": "What is the standard notice period for resignation, and can that notice period be waived for termination due to misconduct?",
        "reference_answer": "The standard notice period for voluntary resignation is 60 days (exit_offboarding_policy.md). Separately, for termination due to proven misconduct, the notice period requirement may be waived entirely at the company's discretion (code_of_conduct.md) -- this is a different scenario from standard resignation, not a modification of the 60-day figure.",
    },
    {
        "id": "Q09",
        "category": "multi_hop",
        "question": "What are the eligibility criteria for working from home, and which cities are approved for full-remote hiring?",
        "reference_answer": "WFH eligibility (for existing office-based employees) requires: 6+ months of continuous employment, a 'Meets Expectations' rating or higher in the most recent review, a role designated remote-eligible, and no more than 1 attendance violation in the preceding 12 months (wfh_policy.md). Full-remote HIRING (a separate, different policy) is currently approved only for Pune, Bengaluru, Hyderabad, Jaipur, Chandigarh, Kochi, and Tier-2 cities with a partnered coworking space -- currently Indore, Coimbatore, Nagpur (remote_work_zones.md).",
    },
    {
        "id": "Q10",
        "category": "multi_hop",
        "question": "What is the casual leave entitlement per year, and how often are performance reviews conducted?",
        "reference_answer": "Casual leave entitlement is 8 days per calendar year (leave_policy_v2.md). Performance reviews are conducted twice a year (performance_review_policy.md). Two independent facts from two different documents.",
    },
    # ---- unanswerable (3) -- real gaps per docs/README.md ----
    {
        "id": "Q11",
        "category": "unanswerable",
        "question": "What is the duration of paternity or maternity leave?",
        "reference_answer": "Not covered anywhere in the knowledge base (checked leave_policy_v1.md, leave_policy_v2.md, benefits_overview.md). The system must say it does not know rather than invent a number.",
    },
    {
        "id": "Q12",
        "category": "unanswerable",
        "question": "How many total employees does NimbusWorks currently have?",
        "reference_answer": "Not covered anywhere in the knowledge base. The system must say it does not know rather than guess a headcount.",
    },
    {
        "id": "Q13",
        "category": "unanswerable",
        "question": "If my probation period is extended, what notice period applies if I resign during that extension?",
        "reference_answer": "Not explicitly addressed. probation_policy.md only covers termination notice (15 days) during the STANDARD probation period; it does not say what applies during an extension. The system must not assume the standard 15-day figure carries over -- it should say this specific case is not covered.",
    },
    # ---- contradictory (2) -- documented cross-doc contradictions ----
    {
        "id": "Q14",
        "category": "contradictory",
        "question": "How many days of paid sick leave am I entitled to per year?",
        "reference_answer": "Contradiction: leave_policy_v1.md (superseded, effective Jan 2023) says 10 days; leave_policy_v2.md (current, effective Mar 2024) says 12 days. The system should surface both values and both sources rather than silently stating only one.",
    },
    {
        "id": "Q15",
        "category": "contradictory",
        "question": "What is the standard probationary period length for new hires?",
        "reference_answer": "Contradiction: probation_policy.md says 3 months; onboarding_checklist.md says 6 months. Note: onboarding_checklist.md is dated AFTER probation_policy.md, but 'newest document wins' is NOT the correct resolution here -- these are genuinely unreconciled, independently-dated documents. The system should surface both values rather than picking the newer document's number as if it were authoritative.",
    },
]
