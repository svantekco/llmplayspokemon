# Execution planning

For significant refactors, architecture changes, or multi-phase work, use `.plans/PLANS.md` as the authoritative execution plan.

Rules:
- Do not begin major implementation without first reading `.plans/PLANS.md`.
- Treat `.plans/PLANS.md` as the source of truth for architecture, invariants, milestone order, and acceptance criteria.
- If a milestone has a corresponding file in `.plans/phases/`, read that file before implementing the milestone.
- Use files in `.plans/tasks/` only as subordinate implementation specs; they must not contain critical context that is missing from `.plans/PLANS.md`.
- Update `.plans/PLANS.md` as a living document when decisions change, discoveries are made, or milestones are split.
- Keep progress, decision log, and surprises/discoveries current.
- Implement one milestone at a time unless the plan explicitly says phases may overlap.