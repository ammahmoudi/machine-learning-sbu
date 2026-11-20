"""Homework 1: Gaussian RBF regression implemented without an ML framework."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import random
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATA = ROOT / "data/raw/bike-sharing/day.csv"
WIDTHS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)
CENTER_COUNTS = (8, 16, 32, 64, 128, 256)
COLUMN_NAMES = {
    "instant": "record_id",
    "dteday": "date",
    "season": "season",
    "yr": "year",
    "mnth": "month",
    "holiday": "is_holiday",
    "weekday": "day_of_week",
    "workingday": "is_working_day",
    "weathersit": "weather_condition",
    "temp": "temperature",
    "atemp": "feels_like_temperature",
    "hum": "humidity",
    "windspeed": "wind_speed",
    "casual": "casual_rentals",
    "registered": "registered_rentals",
    "cnt": "total_rentals",
}
CONTINUOUS = ("temperature", "humidity", "wind_speed")
BINARY = ("year", "is_holiday", "is_working_day")
CATEGORIES = {"season": (1, 2, 3, 4), "day_of_week": tuple(range(7)), "weather_condition": (1, 2, 3, 4)}
ARCHIVE_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"


def shuffled_indices(n, seed):
    """A deterministic split using the standard library, independent of ML tools."""
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return np.array(indices, dtype=int)


def read_dataset(path):
    """Read the original CSV into a DataFrame without changing its values."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def rename_columns(data):
    """Use descriptive column names without changing the original data values."""
    return data.rename(columns=COLUMN_NAMES)


def validate_dataset(data):
    """Check for missing values, invalid category codes, and inconsistent counts."""
    if data.empty:
        raise ValueError("The dataset is empty")
    for column in data.columns:
        if any(not str(value).strip() for value in data[column]):
            raise ValueError(f"Missing values in {column}")
    for name in BINARY:
        if any(float(value) not in (0.0, 1.0) for value in data[name]):
            raise ValueError(f"Unexpected binary value in {name}")
    for name, levels in CATEGORIES.items():
        if any(float(value) not in levels for value in data[name]):
            raise ValueError(f"Unexpected category in {name}")
    # The two rental components must add up to the prediction target.
    for casual, registered, total in zip(data["casual_rentals"], data["registered_rentals"], data["total_rentals"]):
        if int(casual) + int(registered) != int(total):
            raise ValueError("Unexpected target-component mismatch")


def numeric_features(data):
    """Return continuous/binary inputs as a numeric matrix and their column names.

    Each row is one day. These six columns are selected explicitly so neither
    the target nor its component counts can become model inputs.
    """
    names = list(CONTINUOUS + BINARY)
    matrix = np.empty((len(data), len(names)), dtype=float)
    for j, name in enumerate(names):
        matrix[:, j] = [float(value) for value in data[name]]
    return matrix, names


def encode_categories(data):
    """Manually create one indicator column for each possible category.

    Example: season 2 becomes [0, 1, 0, 0]. Return the indicator matrix and
    labels such as 'season=2', in the order defined by CATEGORIES.
    """
    names = []
    columns = []
    for name, levels in CATEGORIES.items():
        for level in levels:
            # A value is 1 only when the day belongs to this category.
            columns.append([float(int(value) == level) for value in data[name]])
            names.append(f"{name}={level}")
    return np.array(columns, dtype=float).T, names


def load_data(path):
    """Combine the data-preparation steps for the command-line experiment.

    The notebook shows each of these steps in a separate cell instead.
    Return original records, the input matrix, target counts, and input names.
    """
    original = read_dataset(path)
    data = rename_columns(original)
    validate_dataset(data)
    numeric, numeric_names = numeric_features(data)
    categorical, category_names = encode_categories(data)
    x = np.column_stack((numeric, categorical))
    y = np.array([float(value) for value in data["total_rentals"]])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite data")
    return original.to_dict(orient="records"), x, y, numeric_names + category_names


def fit_scaler(x):
    """Compute population means/scales of continuous inputs using training rows only."""
    means = np.zeros(x.shape[1])
    scales = np.ones(x.shape[1])
    for j in range(len(CONTINUOUS)):
        means[j] = sum(float(value) for value in x[:, j]) / len(x)
        variance = sum((float(value) - means[j]) ** 2 for value in x[:, j]) / len(x)
        scales[j] = math.sqrt(variance) if variance > 0 else 1.0
    return means, scales


def transform(x, means, scales):
    """Apply previously fitted statistics; never recompute them on test data."""
    return (x - means) / scales


def gaussian_basis(x, centers, width):
    """phi_j(x) = exp(-||x-c_j||^2 / (2 sigma^2)); no library basis builder."""
    if not math.isfinite(width) or width <= 0:
        raise ValueError("RBF width must be finite and positive")
    result = np.empty((len(x), len(centers)))
    for j, center in enumerate(centers):
        squared_distance = np.zeros(len(x))
        for k in range(x.shape[1]):
            difference = x[:, k] - center[k]
            squared_distance += difference * difference
        result[:, j] = np.exp(-squared_distance / (2.0 * width * width))
    return result


def design_matrix(x, centers, width):
    """Construct [1, phi_1, ..., phi_M] explicitly, including an intercept."""
    matrix = np.ones((len(x), len(centers) + 1))
    matrix[:, 1:] = gaussian_basis(x, centers, width)
    return matrix


def sse(target, prediction):
    """Sum squared differences between actual counts and model predictions."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 1 or prediction.shape != target.shape or len(target) == 0:
        raise ValueError("SSE requires matching nonempty one-dimensional vectors")
    residual = target - prediction
    return float(residual @ residual)


def rmse(target, prediction):
    """Convert mean squared error back to the target's units: rentals/day."""
    return math.sqrt(sse(target, prediction) / len(target))


def fit_least_squares(matrix, target):
    """Least-squares solvers are explicitly permitted by the assignment brief."""
    weights, _, rank, singular_values = np.linalg.lstsq(matrix, target, rcond=None)
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] else float("inf")
    return weights, int(rank), condition


def choose_centers(n, count=64, seed=44):
    """Choose center row positions from fitting data without using targets."""
    return shuffled_indices(n, seed)[:min(count, n)]


def fit_rbf_model(inputs, targets, count, width):
    """Fit an RBF model using a reproducible prefix of fitting-data centers."""
    if count < 1 or count + 1 >= len(inputs):
        raise ValueError("Use a positive center count with fewer weights than fitting rows")
    centers = inputs[choose_centers(len(inputs), count=count)]
    matrix = design_matrix(inputs, centers, width)
    weights, rank, condition = fit_least_squares(matrix, targets)
    return centers, weights, rank, condition


def plot_hyperparameter_grid(results, selected_count, selected_width):
    """Show train, validation, and test RMSE on the same color scale."""
    import matplotlib.pyplot as plt

    metrics = [("train_rmse", "Training"), ("validation_rmse", "Inner validation"),
               ("test_rmse", "Test")]
    maximum = max(row[key] for row in results for key, _ in metrics)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    for ax, (key, title) in zip(axes, metrics):
        values = np.array([row[key] for row in results]).reshape(len(CENTER_COUNTS), len(WIDTHS))
        heatmap = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(len(WIDTHS)), labels=[f"{width:g}" for width in WIDTHS])
        ax.set_yticks(range(len(CENTER_COUNTS)), labels=[str(count) for count in CENTER_COUNTS])
        ax.set(xlabel="RBF width", ylabel="Number of centers M", title=title)
        for i in range(len(CENTER_COUNTS)):
            for j in range(len(WIDTHS)):
                ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > maximum * 0.55 else "black")
        ax.scatter(WIDTHS.index(selected_width), CENTER_COUNTS.index(selected_count),
                   marker="s", s=550, facecolors="none", edgecolors="#d35400", linewidths=2)
    fig.colorbar(heatmap, ax=axes, label="RMSE (rentals/day)", shrink=0.85)
    fig.suptitle("Full hyperparameter grid - orange boxes mark the validation-selected pair")
    return fig


def write_csv(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join(
        f"<th>{html.escape(str(value))}</th>" for value in headers
    ) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f"<td>{html.escape(str(value))}</td>" for value in row) + '</tr>'
        for row in rows
    ) + '</tbody></table></div>'


def build_report(rows, results, metadata, output):
    chosen = next(row for row in results if row["width"] == metadata["selected_width"])
    small, large = results[0], results[-1]
    preview = table(["Original column"] + [f"Record {i+1}" for i in range(5)],
                    [[name] + [row[name] for row in rows[:5]] for name in rows[0]])
    schema = [
        ("instant", "Sequential record identifier", "Excluded: identifier"),
        ("dteday", "Calendar date", "Excluded: date; calendar features used instead"),
        ("season", "Season code, 1-4", "Input: four indicators"),
        ("yr", "2011=0, 2012=1", "Input: binary"),
        ("mnth", "Month, 1-12", "Excluded: keep a compact seasonal representation"),
        ("holiday", "Holiday indicator", "Input: binary"),
        ("weekday", "Weekday code, 0-6", "Input: seven indicators"),
        ("workingday", "Neither weekend nor holiday", "Input: binary"),
        ("weathersit", "1 clear; 2 mist/cloud; 3 light precipitation; 4 severe precipitation", "Input: four indicators"),
        ("temp", "Source-normalized temperature", "Input: standardized on training data"),
        ("atemp", "Source-normalized apparent temperature", "Excluded: overlaps temperature information"),
        ("hum", "Relative humidity divided by 100", "Input: standardized on training data"),
        ("windspeed", "Wind speed divided by 67", "Input: standardized on training data"),
        ("casual", "Casual-user rental count", "Excluded: direct target component"),
        ("registered", "Registered-user rental count", "Excluded: direct target component"),
        ("cnt", "Total daily rental count", "Target, in rentals/day"),
    ]
    result_table = table(
        ["Width", "Train RMSE", "Test RMSE", "Inner validation RMSE", "Rank", "Condition number"],
        [[r["width"], f'{r["train_rmse"]:.2f}', f'{r["test_rmse"]:.2f}',
          f'{r["validation_rmse"]:.2f}', r["rank"], f'{r["condition_number"]:.3g}'] for r in results])
    figures = ''.join(f'<figure><img alt="{alt}" src="data:image/png;base64,{base64.b64encode((output / filename).read_bytes()).decode()}"><figcaption>{alt}</figcaption></figure>'
                      for filename, alt in [("rmse_vs_width.png", "Training, test, and inner validation RMSE versus RBF width."),
                                            ("test_predictions.png", "Observed and predicted test rental counts, using the validation-selected width.")])
    count_results = metadata["center_count_study"]["results"]
    count_selected = metadata["center_count_study"]["selected_count"]
    count_best = next(r for r in count_results if r["center_count"] == count_selected)
    count_table = table(["Centers M", "Train RMSE", "Test RMSE", "Validation RMSE"],
                        [[r["center_count"], f'{r["train_rmse"]:.2f}', f'{r["test_rmse"]:.2f}',
                          f'{r["validation_rmse"]:.2f}'] for r in count_results])
    count_plot = base64.b64encode((output / "rmse_vs_centers.png").read_bytes()).decode()
    count_section = f'''<h3>Effect of the number of centers</h3>
<p>After the width study at M = 64, fix width at its validation-selected value {metadata['selected_width']:g} and vary M over {list(CENTER_COUNTS)}. Keep the splits, scaling, and center-selection seed fixed. Each larger center set extends the same shuffled prefix, giving nested basis sets at this fixed width. All systems remain overdetermined, including the inner-fitting systems.</p>
{count_table}<figure><img alt="RMSE versus number of centers" src="data:image/png;base64,{count_plot}"></figure>
<p>Training-only validation selects <strong>M = {count_selected}</strong> at width {metadata['selected_width']:g}. Refit train/test RMSE are <strong>{count_best['train_rmse']:.2f}/{count_best['test_rmse']:.2f}</strong> rentals/day. Increasing M adds flexibility and should not increase the minimum training SSE for these nested bases, apart from numerical effects. Validation and test errors can increase as complexity grows. This sequential study explores both hyperparameters; width is not re-tuned for every M.</p>'''
    grid = metadata["grid_search"]
    grid_chosen = next(r for r in grid["results"] if r["center_count"] == grid["selected_count"]
                       and r["width"] == grid["selected_width"])
    grid_plot = base64.b64encode((output / "hyperparameter_grid.png").read_bytes()).decode()
    grid_table = table(["M / width"] + list(WIDTHS), [
        [count] + [f'{r["validation_rmse"]:.2f}' for r in grid["results"] if r["center_count"] == count]
        for count in CENTER_COUNTS])
    grid_section = f'''<h3>Full grid: jointly vary center count and width</h3>
<p>Evaluate all {len(CENTER_COUNTS)} × {len(WIDTHS)} = {len(grid['results'])} combinations using the same fitting, validation, and test rows. Select the pair solely by inner validation RMSE, before calculating the full grid of test errors. Center sets share the same shuffled prefixes for fair comparisons.</p>
<h4>Validation RMSE grid</h4>{grid_table}
<figure><img alt="Training, validation, and test RMSE heatmaps for all hyperparameter pairs" src="data:image/png;base64,{grid_plot}"></figure>
<p>The joint validation choice is <strong>M = {grid['selected_count']}, width = {grid['selected_width']:g}</strong>. Validation RMSE is {grid_chosen['validation_rmse']:.2f}; refitting on all outer training rows gives training RMSE <strong>{grid_chosen['train_rmse']:.2f}</strong> and test RMSE <strong>{grid_chosen['test_rmse']:.2f}</strong> rentals/day. These settings are the final selection from the tested grid. All test-grid values are descriptive and do not change the validation choice. The earlier one-dimensional sweeps remain useful for explaining individual effects; this grid also captures interactions between width and model size.</p>'''
    gain = 100 * (1 - chosen["test_rmse"] / metadata["baseline_test_rmse"])
    lowest_test = min(results, key=lambda row: row["test_rmse"])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Homework 1 - Linear Regression</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#182b42;background:#f4f6fa;margin:0}}main{{max-width:1060px;margin:32px auto;background:white;padding:40px;border-radius:12px}}h1,h2{{line-height:1.25}}h2{{margin-top:36px;color:#123d69}}.subtitle{{color:#52657a}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #dbe2ea;text-align:left}}th{{background:#edf3fa}}.table-wrap{{overflow-x:auto}}img{{width:100%;height:auto}}figure{{margin:26px 0}}figcaption{{color:#52657a;font-size:13px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f8;padding:16px;font-size:13px}}code{{font-family:Consolas,monospace}}.equation{{background:#edf3fa;padding:14px;font-family:Cambria,serif}}@media print{{body{{background:white}}main{{margin:0;padding:0}}h2{{break-after:avoid}}figure{{break-inside:avoid}}details{{display:none}}}}@media(max-width:650px){{main{{padding:18px;margin:0}}}}</style>
<main><h1>Linear Regression with Gaussian Basis Functions</h1>
<p class="subtitle">Machine Learning - Homework 1<br>Dr. Ali Katanforosh · MSc Computer Science · Shahid Beheshti University<br>Dataset: UCI Bike Sharing, daily observations</p>
<h2>1. Problem and dataset</h2>
<p>Predict total daily bike rentals from weather and calendar variables. We selected the daily file because its 731 observations allow inexpensive, reproducible basis-function experiments. Each row represents one day from 2011-01-01 through 2012-12-31; there are 16 original columns and {metadata['missing_values']} empty fields. The hourly file is not used.</p>
<p>Capital Bikeshare produced the underlying Washington, D.C. rental logs. Hadi Fanaee-T, affiliated with LIAAD/University of Porto, prepared the research dataset by aggregating rental activity and adding weather from Freemeteo and calendar information. The resource supports demand regression and event detection. It is operational observational data, rather than a randomized experiment. <a href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset">UCI dataset record</a>; the archived Readme.txt documents the collection process.</p>
<p>The current UCI page lists CC BY 4.0. Credit: Fanaee-T, H. (2013), <i>Bike Sharing</i>, UCI Machine Learning Repository, <a href="https://doi.org/10.24432/C5W894">doi:10.24432/C5W894</a>. The supplied README also requests citation of Fanaee-T and Gama (2013), <i>Event labeling combining ensemble detectors and background knowledge</i>, <a href="https://doi.org/10.1007/s13748-013-0040-3">doi:10.1007/s13748-013-0040-3</a>.</p>
<h3>First five rows of the original data</h3><p>Shown transposed for readability; each record column is one original data row.</p>{preview}
<h3>Feature and target meanings</h3>{table(['Column', 'Meaning', 'Use'], schema)}
<p>Source caveat: the downloaded README and current UCI page disagree about season labels and temperature normalization. We retain the supplied numeric season codes as categories and temperature in source units, without assuming a Celsius conversion or resolving the labels by guesswork. Weather category 4 has no daily observations but is retained in the fixed encoding. The target identity cnt = casual + registered is checked for every row; both components are excluded.</p>
<h2>2. Preprocessing and experimental protocol</h2>
<p>Nine original predictors become {len(metadata['feature_names'])} numeric inputs: three standardized continuous variables, three unchanged binary variables, and fixed one-hot encodings for season, weekday, and weather. No imputation, clipping, target transformation, or outlier removal is performed. We compute the continuous means and population standard deviations on the training partition only; constant columns receive scale 1.</p>
<p>A seeded shuffle (seed 42) produces {metadata['train_count']} training days and {metadata['test_count']} test days. The training partition is divided once more using seed 43 into {metadata['inner_fit_count']} fitting days and {metadata['validation_count']} validation days. This inner fitting subset supplies its own scaler and centers. Width is selected at M = 64 by inner validation RMSE, followed by center-count selection at that fixed width. Each candidate is then fitted on the full outer training set for the assignment's training/test comparison. Test errors do not change the selected width. The same shuffled indices, center count, and center-selection seed are used throughout each sweep.</p>
<p>This random split measures interpolation across the observed two-year period. Nearby days can be correlated, and both years appear in training and testing, so these results do not establish future forecasting accuracy. A chronological or blocked evaluation would be needed for that claim.</p>
<h2>3. Mathematical formulation and derivation</h2>
<p>For a continuous input component k, z<sub>ik</sub> = (x<sub>ik</sub> - μ<sub>k</sub>)/s<sub>k</sub>, where μ<sub>k</sub> = Σx<sub>ik</sub>/n and s<sub>k</sub>² = Σ(x<sub>ik</sub>-μ<sub>k</sub>)²/n, computed on the fitting data. Binary and categorical indicators remain unchanged.</p>
<p>For the width study, we choose M = 64 centers by sampling training inputs without replacement using seed 44, independently of the targets. At a fixed width σ the centers are constants. Our model is nonlinear in inputs but linear in the learned weights:</p>
<div class="equation">φ<sub>j</sub>(z) = exp(-Σ<sub>k</sub>(z<sub>k</sub> - c<sub>jk</sub>)² / (2σ²))<br>Φ<sub>i0</sub> = 1; Φ<sub>ij</sub> = φ<sub>j</sub>(z<sub>i</sub>)<br>ŷ = Φw; SSE(w) = (y - Φw)<sup>T</sup>(y - Φw); RMSE = √(SSE/n)</div>
<p>Expanding SSE gives y<sup>T</sup>y - 2w<sup>T</sup>Φ<sup>T</sup>y + w<sup>T</sup>Φ<sup>T</sup>Φw. Differentiating with respect to w yields ∇SSE = 2Φ<sup>T</sup>(Φw-y). Setting this to zero yields the normal equations Φ<sup>T</sup>Φw = Φ<sup>T</sup>y. We use the explicitly permitted least-squares solver directly on Φ rather than forming an inverse or squaring the condition number through Φ<sup>T</sup>Φ. It returns a minimum-norm solution when needed, with numerical rank determined by its documented floating-point cutoff. No explicit regularization is used in this assignment.</p>
<p>The outer design matrix has shape {metadata['train_count']} × 65 and is overdetermined. The first study varies width σ and the second varies the number of centers M; candidate values are fixed in code. Model size, centers, and split are held fixed, so changes across the curves are attributable to width under this protocol.</p>
<h2>4. Experimental results</h2><h3>Effect of width at M = 64</h3>{result_table}{figures}
<p>The validation-selected width is <strong>{chosen['width']:g}</strong>. After refitting on all outer training rows, training RMSE is <strong>{chosen['train_rmse']:.2f}</strong> and test RMSE is <strong>{chosen['test_rmse']:.2f} rentals/day</strong>. Predicting the outer training mean for every test day gives RMSE {metadata['baseline_test_rmse']:.2f}, so the selected model reduces test RMSE by {gain:.1f}% relative to that baseline. Baseline training RMSE is {metadata['baseline_train_rmse']:.2f}.</p>
<p>The lowest test RMSE in the descriptive sweep is {lowest_test['test_rmse']:.2f} at width {lowest_test['width']:g}. We do not switch to that width: choosing a model from test performance would reuse the test set for tuning. The validation curve uses fewer fitting rows and different training-only centers/scaling, so its numerical level is not directly comparable to the outer training curve.</p>
{count_section}
{grid_section}
<h2>5. Analysis and conclusions</h2>
<p>At the narrowest tested width ({small['width']:g}), training/test RMSE are {small['train_rmse']:.2f}/{small['test_rmse']:.2f}; at the widest ({large['width']:g}) they are {large['train_rmse']:.2f}/{large['test_rmse']:.2f}. Narrow RBFs respond locally and may leave many non-center observations poorly represented. Broad RBF columns become increasingly similar, which can worsen conditioning and produce large cancelling coefficients. The measured condition numbers in the table distinguish numerical sensitivity from an assertion about statistical overfitting. Changing width does not create a nested sequence of models, so training RMSE need not decrease monotonically.</p>
<p>The chosen model uses weather and calendar information to improve upon a constant prediction on this fixed split. The numerical results describe this dataset, basis family, center sample, width grid, and split only. They do not establish causal effects or statistical significance. A single validation split introduces selection variability; additional training-only cross-validation and chronological evaluation would strengthen a deployment study. Least squares can also produce negative counts; predictions are left unclipped so reported RMSE matches the fitted objective. The next assignment can reuse these inputs, split, and RBF implementation while increasing the center count to investigate overfitting and regularization.</p>
<h2>6. Source code and execution</h2>
<p>All basis evaluation, design-matrix construction, splitting, preprocessing, SSE, RMSE, and experiment orchestration are implemented in regression.py. NumPy supplies arrays, elementary arithmetic/exponentials, and the permitted least-squares solver. Pandas handles CSV loading and notebook table display; Matplotlib is used only for plotting. No scikit-learn, PyTorch, TensorFlow, library RBF builder, scaler, metric, or regression estimator is used.</p>
<p>Install uv, then run these commands from the workspace root in PowerShell:</p><pre>uv sync --locked
uv run --locked python assignments/hw01-linear-regression/download_data.py
uv run --locked python assignments/hw01-linear-regression/regression.py
uv run --locked python -m unittest discover -s assignments/hw01-linear-regression -p test_regression.py</pre>
<p>The download script validates the archived source hash and preserves the original files. After the first download, experiments run offline. Paths are resolved relative to the script, so the experiment command also works from other working directories. Optional arguments: --data PATH and --output PATH. The generated report embeds its plots and full implementation and can be opened offline or printed to PDF. Companion CSV files contain metrics and test predictions; metadata.json and model.npz preserve the exact split, centers, scaler, and selected weights.</p>
<p>Recorded runtime: Python {html.escape(metadata['python_version'])}, NumPy {html.escape(metadata['numpy_version'])}, Matplotlib {html.escape(metadata['matplotlib_version'])}. The workspace pyproject.toml declares dependencies and uv.lock records exact package versions. Dataset download date: 2026-11-20. Source archive SHA-256: <code>{ARCHIVE_SHA256}</code>. day.csv SHA-256: <code>{metadata['data_sha256']}</code>.</p>
<details><summary>Complete regression.py source</summary><pre>{html.escape(Path(__file__).read_text(encoding='utf-8'))}</pre></details>
<h2>7. References</h2><ul>
<li><a href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset">UCI Bike Sharing dataset</a> and the original archive's Readme.txt (preserved with the data).</li>
<li><a href="https://doi.org/10.1007/s13748-013-0040-3">Fanaee-T and Gama, 2013</a>, original associated publication.</li>
<li>Bishop, C. M. (2006), <i>Pattern Recognition and Machine Learning</i>, Chapter 3, Linear Models for Regression (course textbook).</li>
<li><a href="https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html">NumPy least-squares solver documentation</a>.</li>
<li>The instructor's supplied <i>Linear Regression - Programming Assignment - Part 1</i> brief.</li></ul>
</main></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")


def run(data_path=DEFAULT_DATA, output=HERE / "results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, x, y, feature_names = load_data(data_path)
    order = shuffled_indices(len(y), 42)
    cut = int(0.8 * len(y))
    train, test = order[:cut], order[cut:]
    inner_order = shuffled_indices(len(train), 43)
    inner_cut = int(0.8 * len(train))
    inner_fit, validation = train[inner_order[:inner_cut]], train[inner_order[inner_cut:]]
    means, scales = fit_scaler(x[train])
    z_train, z_test = transform(x[train], means, scales), transform(x[test], means, scales)
    center_positions = choose_centers(len(train))
    centers = z_train[center_positions]
    inner_means, inner_scales = fit_scaler(x[inner_fit])
    z_fit = transform(x[inner_fit], inner_means, inner_scales)
    z_validation = transform(x[validation], inner_means, inner_scales)
    inner_centers = z_fit[choose_centers(len(inner_fit))]

    # Width selection is finished before computing any test errors.
    validation_scores = []
    for width in WIDTHS:
        weights, _, _ = fit_least_squares(design_matrix(z_fit, inner_centers, width), y[inner_fit])
        score = rmse(y[validation], design_matrix(z_validation, inner_centers, width) @ weights)
        validation_scores.append(score)
    selected_width = WIDTHS[min(range(len(WIDTHS)), key=lambda i: validation_scores[i])]
    results, selected_weights, selected_prediction = [], None, None
    for width, validation_score in zip(WIDTHS, validation_scores):
        matrix = design_matrix(z_train, centers, width)
        weights, rank, condition = fit_least_squares(matrix, y[train])
        train_prediction = matrix @ weights
        test_prediction = design_matrix(z_test, centers, width) @ weights
        results.append(dict(width=width, train_rmse=rmse(y[train], train_prediction),
                            test_rmse=rmse(y[test], test_prediction), validation_rmse=validation_score,
                            rank=rank, condition_number=condition))
        if width == selected_width:
            selected_weights, selected_prediction = weights, test_prediction

    train_mean = sum(float(value) for value in y[train]) / len(train)
    metadata = dict(selected_width=selected_width, centers_count=len(centers), seed_split=42,
                    seed_validation=43, seed_centers=44, train_count=len(train), test_count=len(test),
                    inner_fit_count=len(inner_fit), validation_count=len(validation),
                    feature_names=feature_names, train_indices=train.tolist(), test_indices=test.tolist(),
                    inner_fit_indices=inner_fit.tolist(), validation_indices=validation.tolist(),
                    center_indices=train[center_positions].tolist(), scaler_means=means.tolist(),
                    scaler_scales=scales.tolist(), missing_values=sum(not v.strip() for r in rows for v in r.values()),
                    baseline_train_rmse=rmse(y[train], np.full(len(train), train_mean)),
                    baseline_test_rmse=rmse(y[test], np.full(len(test), train_mean)),
                    python_version=platform.python_version(), numpy_version=np.__version__,
                    matplotlib_version=matplotlib.__version__,
                    data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest())
    # Study the number of centers at the validation-selected width.
    count_validation_scores = []
    for count in CENTER_COUNTS:
        count_centers, count_weights, _, _ = fit_rbf_model(z_fit, y[inner_fit], count, selected_width)
        prediction = design_matrix(z_validation, count_centers, selected_width) @ count_weights
        count_validation_scores.append(rmse(y[validation], prediction))
    selected_count = CENTER_COUNTS[min(range(len(CENTER_COUNTS)), key=lambda i: count_validation_scores[i])]

    count_results = []
    for count, validation_score in zip(CENTER_COUNTS, count_validation_scores):
        count_centers, count_weights, rank, condition = fit_rbf_model(z_train, y[train], count, selected_width)
        count_train_prediction = design_matrix(z_train, count_centers, selected_width) @ count_weights
        count_test_prediction = design_matrix(z_test, count_centers, selected_width) @ count_weights
        count_results.append(dict(center_count=count, width=selected_width,
                                  train_rmse=rmse(y[train], count_train_prediction),
                                  test_rmse=rmse(y[test], count_test_prediction),
                                  validation_rmse=validation_score, rank=rank, condition_number=condition))
    # Save center-count study separately from the original width study.
    metadata["center_count_study"] = dict(counts=list(CENTER_COUNTS), fixed_width=selected_width,
                                         selected_count=selected_count, results=count_results)
    write_csv(output / "center_count_metrics.csv", count_results)
    count_centers, count_weights, _, _ = fit_rbf_model(z_train, y[train], selected_count, selected_width)
    np.savez(output / "center_count_model.npz", weights=count_weights, centers=count_centers,
             means=means, scales=scales, width=selected_width, train_indices=train, test_indices=test)
    # Joint hyperparameter search: select the pair before evaluating test errors.
    grid_validation_results = []
    for count in CENTER_COUNTS:
        for width in WIDTHS:
            grid_centers, grid_weights, _, _ = fit_rbf_model(z_fit, y[inner_fit], count, width)
            prediction = design_matrix(z_validation, grid_centers, width) @ grid_weights
            grid_validation_results.append(dict(center_count=count, width=width,
                                                validation_rmse=rmse(y[validation], prediction)))
    grid_best_validation = min(grid_validation_results, key=lambda row: row["validation_rmse"])
    grid_selected_count = grid_best_validation["center_count"]
    grid_selected_width = grid_best_validation["width"]

    grid_results = []
    for setting in grid_validation_results:
        count, width = setting["center_count"], setting["width"]
        grid_centers, grid_weights, rank, condition = fit_rbf_model(z_train, y[train], count, width)
        grid_train_prediction = design_matrix(z_train, grid_centers, width) @ grid_weights
        grid_test_prediction = design_matrix(z_test, grid_centers, width) @ grid_weights
        grid_results.append(dict(**setting, train_rmse=rmse(y[train], grid_train_prediction),
                                 test_rmse=rmse(y[test], grid_test_prediction),
                                 rank=rank, condition_number=condition))
        if count == grid_selected_count and width == grid_selected_width:
            grid_selected_centers = grid_centers
            grid_selected_weights = grid_weights
            grid_selected_prediction = grid_test_prediction
    # Save full-grid results and the jointly selected model.
    metadata["grid_search"] = dict(counts=list(CENTER_COUNTS), widths=list(WIDTHS),
                                  selected_count=grid_selected_count, selected_width=grid_selected_width,
                                  results=grid_results)
    metadata["final_model"] = dict(filename="grid_model.npz", center_count=grid_selected_count,
                                   width=grid_selected_width, selection="minimum inner validation RMSE over full grid")
    write_csv(output / "hyperparameter_grid.csv", grid_results)
    np.savez(output / "grid_model.npz", weights=grid_selected_weights, centers=grid_selected_centers,
             means=means, scales=scales, width=grid_selected_width, train_indices=train, test_indices=test)
    write_csv(output / "grid_test_predictions.csv", [dict(row_index=int(i), date=rows[i]["dteday"],
              actual=float(y[i]), predicted=float(predicted)) for i, predicted in zip(test, grid_selected_prediction)])
    write_csv(output / "metrics.csv", results)
    write_csv(output / "test_predictions.csv", [dict(row_index=int(i), instant=rows[i]["instant"],
              date=rows[i]["dteday"], actual=float(y[i]), predicted=float(predicted),
              residual=float(y[i] - predicted)) for i, predicted in zip(test, selected_prediction)])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez(output / "model.npz", weights=selected_weights, centers=centers, means=means,
             scales=scales, width=selected_width, train_indices=train, test_indices=test)
    split_records = []
    train_set, inner_set = set(train.tolist()), set(inner_fit.tolist())
    for i, row in enumerate(rows):
        split_records.append(dict(row_index=i, instant=row["instant"], date=row["dteday"],
                                  outer_split="train" if i in train_set else "test",
                                  inner_split=("fit" if i in inner_set else "validation") if i in train_set else "unused"))
    write_csv(output / "split.csv", split_records)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for key, label, color in [("train_rmse", f"Training ({len(train)} days)", "#12648a"),
                              ("test_rmse", f"Test ({len(test)} days)", "#cb5229"),
                              ("validation_rmse", f"Inner validation ({len(validation)} days)", "#66852c")]:
        ax.plot(WIDTHS, [r[key] for r in results], "o-", label=label, color=color)
    ax.axvline(selected_width, color="#666666", linestyle=":", label="Validation-selected width")
    ax.set(xscale="log", xlabel="Gaussian RBF width (sigma, logarithmic scale)",
           ylabel="RMSE (rentals/day)", title="Effect of RBF width - fixed centers and split")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(output / "rmse_vs_width.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    ax.scatter(y[test], selected_prediction, alpha=0.7, s=24, color="#12648a")
    low, high = min(0, float(min(selected_prediction))), max(float(max(y[test])), float(max(selected_prediction)))
    ax.plot([low, high], [low, high], "--", color="#666666", label="Perfect prediction")
    ax.set(xlabel="Observed rentals/day", ylabel="Predicted rentals/day", title=f"Held-out predictions - width {selected_width:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(output / "test_predictions.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for key, label, color in [("train_rmse", "Training", "#12648a"),
                              ("test_rmse", "Test", "#cb5229"),
                              ("validation_rmse", "Inner validation", "#66852c")]:
        ax.plot(CENTER_COUNTS, [r[key] for r in count_results], "o-", label=label, color=color)
    ax.axvline(selected_count, linestyle=":", color="#666666", label="Validation-selected M")
    ax.set(xscale="log", xlabel="Number of RBF centers M", ylabel="RMSE (rentals/day)",
           title=f"Effect of center count - fixed width {selected_width:g}")
    ax.set_xticks(CENTER_COUNTS, labels=[str(count) for count in CENTER_COUNTS])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(output / "rmse_vs_centers.png", dpi=170)
    plt.close(fig)
    fig = plot_hyperparameter_grid(grid_results, grid_selected_count, grid_selected_width)
    fig.savefig(output / "hyperparameter_grid.png", dpi=170)
    plt.close(fig)
    build_report(rows, results, metadata, output)
    print(json.dumps({"selected_width": selected_width, "results": results,
                      "final_model": metadata["final_model"],
                      "baseline_test_rmse": metadata["baseline_test_rmse"], "report": str(output / "report.html")}, indent=2))
    return results, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    arguments = parser.parse_args()
    run(arguments.data, arguments.output)
