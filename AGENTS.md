# Course workspace guidance

This workspace contains Machine Learning coursework for Dr. Ali Katanforosh, MSc Computer Science, Shahid Beheshti University (SBU).

- Read the root README and the relevant assignment README and original PDF before implementing homework.
- Preserve original course materials. Keep textbook PDFs under `references/textbooks/` and assignment briefs in their respective assignment folders.
- The seven instructor-suggested dataset links are in `data/README.md`. Homework 1 uses the daily UCI Bike Sharing dataset; preserve this choice and its recorded provenance for homework 2.
- Homework 2 extends homework 1 with the same dataset and reusable preprocessing/basis code. Keep splits fixed across its methods and lambda values.
- Follow the briefs: implement basis functions, losses, gradients, and learning procedures ourselves. High-level solvers are permitted for linear systems, inverses, and least squares; avoid high-level ML model implementations.
- Keep raw data unchanged, fit preprocessing only on training data, record seeds and settings, and never fabricate dataset provenance or experimental results.
- Do not create an environment or install packages until implementation requires them.
- Use uv for package and environment management. Dependencies are declared in the root `pyproject.toml`, with exact versions in `uv.lock`. Use `uv sync --locked` and `uv run --locked` for reproducibility.

## Current documentation with Context7

Use Context7 MCP for questions about a library, framework, SDK, API, CLI tool, or cloud service, including syntax, configuration, migration, setup, and library-specific debugging. Start with `resolve-library-id` using the library name and question unless an exact `/org/project` ID is supplied. Select the most relevant reputable match, using version-specific IDs when appropriate. Then call `query-docs` with the selected ID and a full question scoped to one concept; query separate concepts separately unless their interaction is the subject. Prefer these docs over web search.

Context7 is not required for general programming concepts, business logic debugging, code review, refactoring, or scripts written from scratch that do not require library documentation.
