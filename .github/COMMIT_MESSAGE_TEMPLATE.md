# Commit Message Template

Subject (first line):
- Brief summary in the imperative mood (e.g., "Add X", "Fix Y", "Update Z").
- Limit subject to 50 characters or fewer.

<blank line>

Body: up to 6 bullet points explaining reasons/intent (each prefixed with a single hyphen)
- Keep bullets concise; focus on why things changed, not how.
- Do not include more than 6 bullets.

Example:
Add file-based logging for report generation

- Add file handler writing to output/log with timestamped filename
- Persist extracted Item 1A snippets per filing for triage
- Avoid duplicate handlers when script re-runs
- Ensure logs use UTF-8 encoding

Notes:
- Save this file as `.github/COMMIT_MESSAGE_TEMPLATE.md` in the repository root.
- When you ask me to produce a commit message, say "use repo commit template" and I will read and follow this template.
- (Optional) To enable a local git commit template for your machine, save the same content to `.gitmessage` and run:

```bash
git config --local commit.template .gitmessage
```
