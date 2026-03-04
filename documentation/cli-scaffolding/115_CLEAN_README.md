### **Issue: Refactor README.md for External Stakeholder Clarity (The "Scream Sheet" Update)**

**Description:**
The current `README.md` is technically exhaustive but functions more as a technical manual than a product overview. To support the upcoming LinkedIn launch and "External Contact" phase, the documentation needs to be refocused on the value proposition (Risk Novelty & Delta Analysis) while offloading implementation details to supporting files.

**Objectives:**

1. **Lead with Value:** Replace the technical definition with a "Scream Sheet" header that explains *why* SigmaK exists (solving "Risk Drift").
2. **Visual Proof:** Add a section for a "Sample Output" or "Analysis Table" to show the semantic delta scoring in action.
3. **Modularize Technical Debt:** Move deep-dive sections (REST API, Docker Performance, Redis Memory) into `/docs/ARCHITECTURE.md`.
4. **Persona-Based Quick Start:** Create distinct paths for "Analyst" (Running reports) vs. "Engineer" (System setup).

**Technical Constraints:**

* Maintain the CI/CD badge.
* Keep the CLI usage section but condense the global flags.
* Must explicitly mention the "Human in the Loop" alpha-testing phase to encourage feedback.

**Definition of Done:**

* [ ] `README.md` reduced by 50-60% in length.
* [ ] `docs/ARCHITECTURE.md` created with original performance/maintenance/API specs.
* [ ] A "Why SigmaK?" comparison table is present in the main README.
* [ ] All `mypy` and `pytest` instructions remain discoverable but secondary.
