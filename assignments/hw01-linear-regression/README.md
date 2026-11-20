# Homework 1 - Linear Regression

Status: complete and verified. Dataset: UCI Bike Sharing, daily observations.

Source of requirements: the instructor-provided *Linear Regression - Programming Assignment - Part 1* PDF, retained locally and excluded from the public repository. The original brief takes precedence over this checklist.

## Implementation checklist

- [x] Select one dataset from the [suggested catalog](../../data/README.md).
- [x] Define the features and regression target, and document preprocessing.
- [x] Choose a basis family controlled by at least one hyperparameter, such as polynomial degree or RBF width/number of centers.
- [x] Choose offline least squares or online SGD (sequential or mini-batch).
- [x] Implement separate functions for basis evaluation and sum of squared errors (SSE).
- [x] Implement the design matrix for offline learning or the error gradient for online learning.
- [x] Split approximately 80% training / 20% testing and record the split.
- [x] Compute training and test RMSE separately.
- [x] Vary the basis hyperparameter and plot both training and test RMSE against it.

## Submission checklist

- [x] Introduce the dataset, show its first rows, and explain every selected feature and the target.
- [x] Describe who collected the data, how and why, and which subjects or objects it represents; cite the original sources.
- [x] Explain the mathematical formulation and any derivations developed for the assignment.
- [x] Include all source code and package installation/execution instructions.
- [x] Present experiment tables/plots, analysis, and conclusions.

The brief permits a single organized Jupyter Notebook containing the deliverables, or a publicly accessible cloud link to the complete deliverables.

## Implementation constraints

The brief strongly encourages avoiding high-level ML libraries and deep learning frameworks. Library functions are explicitly permitted for solving linear systems, matrix inversion, and least-squares problems. Basis functions, error calculations, gradients, and the learning procedure must be implemented by the student. Basic vector and matrix operations are permitted.

Keep the dataset, preprocessing decisions, and basis implementation reusable for [homework 2](../hw02-regularized-linear-regression/README.md).


## Deliverables and execution

- [Executed Jupyter Notebook](assignment1.ipynb): dataset description, equations, model code, experiments, tables, six plots, and a concise discussion. It reads the original daily CSV from the shared data folder.
- Pandas loads the CSV and displays the data preview and results tables. Basis functions, category encoding, scaling, SSE, RMSE, and the regression implementation remain manual.
- Local submission ZIP (excluded from version control): report, source code, original brief, data/provenance, saved results, and uv project/lock files. Extract it and follow `START_HERE.txt`.
- [Complete report](results/report.html): dataset preview and provenance, feature definitions, mathematical derivation, experimental tables/plots, analysis, full regression implementation, and run instructions. The HTML embeds its plots and can be read offline.
- [Regression implementation](regression.py): manually implemented basis functions, design matrix, preprocessing, SSE, and RMSE; offline training through the permitted NumPy least-squares solver.
- [Download script](download_data.py): verifies the exact UCI archive and preserves original files.
- [Mathematical and data checks](test_regression.py): nine tests covering the basis, metrics, synthetic weight recovery, scaling, splits, leakage exclusions, and least-squares stationarity.
- [Metrics](results/metrics.csv), [test predictions](results/test_predictions.csv), [split assignments](results/split.csv), [experiment metadata](results/metadata.json), and [saved model](results/model.npz).
- [RBF illustration](results/rbf_intuition.png): how width and M affect Gaussian basis functions. [Actual RBF features](results/rbf_features.png): 30 test days by 256 centers.
- [Width RMSE plot](results/rmse_vs_width.png), [center-count RMSE plot](results/rmse_vs_centers.png), and [width-study prediction plot](results/test_predictions.png).
- [Full-grid metrics](results/hyperparameter_grid.csv), [train/validation/test heatmaps](results/hyperparameter_grid.png), [final selected grid model](results/grid_model.npz), and [its test predictions](results/grid_test_predictions.csv).
- [Center-count metrics](results/center_count_metrics.csv) and [center-count selected model](results/center_count_model.npz). The original saved model is retained for the width study.

Use uv from the workspace root (Python 3.13 is selected by the workspace version file):

```powershell
uv sync --locked
uv run --locked python assignments/hw01-linear-regression/download_data.py
uv run --locked python assignments/hw01-linear-regression/regression.py
uv run --locked python -m unittest discover -s assignments/hw01-linear-regression -p test_regression.py
```

Dependencies are declared in [pyproject.toml](../../pyproject.toml), with exact resolved versions and hashes in [uv.lock](../../uv.lock). uv manages the local environment; no activation command is needed. No high-level ML framework is installed or used. After downloading dependencies and data, the experiment can also run with `uv run --locked --offline`.

### Using the notebook

Run `uv sync --locked --group notebook`, open [assignment1.ipynb](assignment1.ipynb) in VS Code, select the workspace `.venv` Python interpreter as the kernel, and choose **Run All**. The notebook already contains executed outputs, so it can also be read without running it. Its 29 code cells were successfully executed and reproduce the original results. Keep the project folders together so the notebook can read the CSV; there is no encoded dataset block.

To regenerate and execute the notebook programmatically:

```powershell
uv run --locked --group notebook python assignments/hw01-linear-regression/build_notebook.py
uv run --locked --group notebook python assignments/hw01-linear-regression/execute_notebook.py
```

The notebook separates CSV loading, data checks, numerical inputs, category encoding, target selection, basis functions, splitting, scaling, center selection, training, and evaluation. Columns are renamed immediately after loading (for example, `temp` to `temperature` and `cnt` to `total_rentals`), and the readable names are used throughout the notebook. Section headings keep the steps separate; explanatory notes focus on modeling choices and mathematics. The combined `load_data` wrapper remains available to the command-line script.

The [builder](build_notebook.py) copies core functions and experiment logic from the verified script and uses normal CSV loading. Building clears existing outputs; the [executor](execute_notebook.py) runs every cell, validates the notebook, and saves the executed outputs and plots.

## Measured outcome

- Fixed split: 584 training days / 147 test days; seeds and exact row assignments are saved.
- Width study: 64 training-selected centers plus an intercept; nine widths from 0.5 to 12.
- Center-count study: M = 8, 16, 32, 64, 128, 256 at validation-selected width 2.0, using nested center sets and the same splits/scaling. All systems remain overdetermined.
- Width-study validation selected width 2.0 at M = 64; train RMSE **620.31**, test RMSE **697.09 rentals/day**.
- Center-count validation selected M = 128 at width 2.0: train RMSE **514.28**, validation RMSE **718.93**, test RMSE **699.28**. At M = 256, training RMSE falls to 400.00 while test RMSE rises to 754.98.
- The first two studies are sequential sweeps. A third study evaluates the complete **6 x 9 grid (54 pairs)** and selects both parameters jointly using validation.
- **Final grid choice: M = 256, width = 1.5.** Training RMSE **429.42**, validation RMSE **700.78**, test RMSE **731.94**. The grid lowers validation RMSE but does not improve test RMSE over the sequential choice on this split.
- [metadata.json](results/metadata.json) records the original width study at top level, the sequential count study under `center_count_study`, and the full grid under `grid_search`. `final_model` identifies [grid_model.npz](results/grid_model.npz).
- Constant training-mean baseline test RMSE: **1910.15**.
- The sweep's lowest test RMSE is 679.55 at width 3.0; it is reported descriptively and not used to change the validation-selected model.
- All nine tests pass. The random split evaluates interpolation within 2011-2012 and does not establish future forecasting accuracy.
