# Machine Learning · SBU

**M.Sc. Computer Science · Shahid Beheshti University**  
Coursework by **[Amirhossein Mahmoudi](https://github.com/ammahmoudi)** · Instructor: **Dr. Ali Katanforosh**

[![Checks](https://github.com/ammahmoudi/machine-learning-sbu/actions/workflows/checks.yml/badge.svg)](https://github.com/ammahmoudi/machine-learning-sbu/actions/workflows/checks.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Environment](https://img.shields.io/badge/managed_with-uv-DE5FE9)


Assignments, notebooks, mathematical derivations, and experiments from my graduate Machine Learning course.

## Coursework

| Assignment | Topics | Status |
| --- | --- | --- |
| [01 · Linear regression](assignments/hw01-linear-regression/README.md) | Basis functions, least squares, evaluation, hyperparameter search | Complete |
| [02 · Regularized regression](assignments/hw02-regularized-linear-regression/README.md) | Ridge, Lasso, GD, SGD, sparsity | Planned |

Each assignment has its own README with the problem description, notebook, implementation, results, and execution instructions.

## Course resources

- [Dataset catalog](data/README.md): the instructor's suggested datasets and recorded data provenance.
- [Course notes](notes/README.md): lecture notes, derivations, and study summaries.
- [References](references/README.md): textbook bibliography.

## Working environment

Python 3.13, with dependencies and environments managed by [uv](https://docs.astral.sh/uv/). The shared environment is defined in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). Follow each assignment's README to install dependencies, obtain its data, and run the experiments.

Textbook and instructor PDFs, raw downloads, and local submission bundles are kept outside version control. Dataset citations and attribution are documented with the relevant assignment and in the dataset catalog.
