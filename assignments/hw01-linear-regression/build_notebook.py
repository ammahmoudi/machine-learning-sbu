"""Build the concise homework notebook from the verified implementation."""

import ast
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    source = (HERE / "regression.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in tree.body if isinstance(node, ast.FunctionDef)}
    run = functions["run"]
    # Reuse the experiment body exactly, displaying each stage in a separate cell.
    validation = textwrap.dedent(run[run.index('    # Width selection'):run.index('    results, selected_weights')])
    experiment = textwrap.dedent(run[run.index('    results, selected_weights'):run.index('    train_mean =')])
    count_study = textwrap.dedent(run[run.index('    # Study the number of centers'):run.index('    # Save center-count study')])
    grid_validation = textwrap.dedent(run[run.index('    # Joint hyperparameter search'):run.index('    grid_results = []')])
    grid_evaluation = textwrap.dedent(run[run.index('    grid_results = []'):run.index('    # Save full-grid results')])
    cells = []

    def add(kind, content, collapsed=False):
        cell = dict(cell_type=kind, id=f"hw01-{len(cells)+1:02}",
                    metadata={}, source=textwrap.dedent(content).strip() + "\n")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if collapsed:
                cell["metadata"]["jupyter"] = {"source_hidden": True}
        cells.append(cell)

    md = lambda content: add("markdown", content)
    code = lambda content: add("code", content)

    md(r"""
    # Homework 1: Linear Regression

    **Amirhossein Mahmoudi**

    **Machine Learning · Dr. Ali Katanforosh**  
    **MSc Computer Science · Shahid Beheshti University**

    Predict daily bike rentals using Gaussian radial basis functions (RBFs) and
    offline least squares. Study how both RBF width and number of centers affect
    training and test RMSE.

    **Run:** from the workspace root, run `uv sync --locked --group notebook`.
    Open this notebook, select the workspace `.venv` Python kernel, and choose **Run All**.
    Keep the extracted project folders together so the notebook can read the CSV.
    """)
    code("""
    import math
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Markdown, display

    %matplotlib inline

    data_path = Path("data/raw/bike-sharing/day.csv")
    if not data_path.exists():
        data_path = Path("../../data/raw/bike-sharing/day.csv")
    """)
    md("""
    ## 1. Dataset description and provenance

    Each observation is one day in Washington, D.C., from 2011-01-01 through
    2012-12-31. The daily file contains **731 observations and 16 original columns**.
    The target is a count of rentals rather than unique cyclists.

    Capital Bikeshare produced the underlying operational trip logs. Hadi Fanaee-T
    (LIAAD/University of Porto) prepared the research dataset by aggregating rental
    activity and adding weather information from Freemeteo and calendar information.
    The dataset supports demand prediction and event detection; it is observational
    data, not a randomized experiment. The archive's original `Readme.txt` documents
    this preparation. UCI currently lists the dataset under **CC BY 4.0**.

    - [UCI dataset record](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
    - Fanaee-T, H. (2013), *Bike Sharing*, [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894).
    - Fanaee-T and Gama (2013), *Event labeling combining ensemble detectors and
      background knowledge*, [doi:10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).

    Downloaded on **2026-11-20** from the original UCI archive:
    <https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip>.
    """)
    md("""
    ### 1.1 Load the CSV and rename columns

    """)
    code(functions["read_dataset"] + "\n\ndf = read_dataset(data_path)")
    column_names = next(ast.get_source_segment(source, node) for node in tree.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == "COLUMN_NAMES"
                            for target in node.targets))
    code(column_names + "\n\n" + functions["rename_columns"]
         + "\n\ndf = rename_columns(df)\ndisplay(df.head())")
    md("""
    ### Feature dictionary and preprocessing decisions

    | Column | Meaning | Treatment |
    |---|---|---|
    | `record_id` | Sequential record identifier | Excluded |
    | `date` | Calendar date | Excluded; calendar indicators used instead |
    | `season` | Season code, 1-4 | Four one-hot indicators |
    | `year` | 0 = 2011, 1 = 2012 | Binary input |
    | `month` | Month, 1-12 | Excluded to keep the seasonal representation compact |
    | `is_holiday` | Holiday flag | Binary input |
    | `day_of_week` | Weekday code, 0-6 | Seven one-hot indicators |
    | `is_working_day` | Neither weekend nor holiday | Binary input |
    | `weather_condition` | 1 clear; 2 mist/cloud; 3 light precipitation; 4 severe precipitation | Four one-hot indicators |
    | `temperature` | Source-normalized temperature | Standardize on fitting data |
    | `feels_like_temperature` | Source-normalized apparent temperature | Excluded because it overlaps temperature information |
    | `humidity` | Relative humidity divided by 100 | Standardize on fitting data |
    | `wind_speed` | Wind speed divided by 67 | Standardize on fitting data |
    | `casual_rentals` | Casual-user rental count | Excluded: part of the target |
    | `registered_rentals` | Registered-user rental count | Excluded: part of the target |
    | `total_rentals` | Total daily rentals | Regression target, rentals/day |

    The code checks `total_rentals = casual_rentals + registered_rentals` for every observation. Including
    either component as a predictor would introduce target leakage.

    Use temperature values as provided and treat season codes as categories.
    Weather category 4 has no daily observations but stays in the fixed schema.

    Three continuous variables, three binary variables, and 15 category indicators
    produce **21 numeric inputs**. Indicators have a fixed source-defined category set.
    No imputation, outlier removal, target transformation, or prediction clipping is used.
    """)
    md("""
    ### 1.2 Choose the input columns

    Exclude `total_rentals`, `casual_rentals`, and `registered_rentals` from the inputs to prevent target leakage.
    """)
    constants = '\n'.join(ast.get_source_segment(source, node) for node in tree.body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id in
                              {'CONTINUOUS', 'BINARY', 'CATEGORIES'} for target in node.targets))
    code(constants)
    md("""
    ### 1.3 Check the data

    Check missing values, category codes, and the identity `total_rentals = casual_rentals + registered_rentals`.
    """)
    code(functions["validate_dataset"] + "\n\nvalidate_dataset(df)\nprint('Data checks passed.')")
    md("""
    ### 1.4 Prepare the numerical inputs

    """)
    code(functions["numeric_features"] + "\n\nnumeric, numeric_names = numeric_features(df)\n"
         + "display(pd.DataFrame(numeric, columns=numeric_names).head())")
    md("""
    ### 1.5 Encode the categorical inputs

    Use one-hot encoding to avoid imposing an ordinal relationship on category codes.
    """)
    code(functions["encode_categories"] + "\n\ncategorical, category_names = encode_categories(df)\n"
         + "print('Categorical input columns:', category_names)")
    md("""
    ### 1.6 Assemble X and the target y

    """)
    code("""
    x = np.column_stack((numeric, categorical))
    feature_names = numeric_names + category_names
    y = np.array([float(value) for value in df["total_rentals"]])
    print(f"Input matrix X: {x.shape}; target vector y: {y.shape}")
    """)
    md(r"""
    ## 2. Mathematical formulation

    For each continuous input coordinate, use fitting-data population statistics:

    $$\mu_k=\frac{1}{n}\sum_{i=1}^{n}x_{ik},\qquad
    s_k=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ik}-\mu_k)^2},\qquad
    z_{ik}=\frac{x_{ik}-\mu_k}{s_k}.$$

    Use scale 1 if a coordinate is constant. Binary and categorical indicators are
    left unchanged. Validation/test rows never contribute to these statistics.

    Select $M$ center inputs $c_j$ from fitting data without replacement, with seed
    44 and without consulting the targets. Start with $M=64$ for the width study,
    then vary $M$ in a second study and evaluate all pairs in a full grid. For width $\sigma>0$:

    $$\phi_j(z)=\exp\!\left(-\frac{\sum_k(z_k-c_{jk})^2}{2\sigma^2}\right),
    \quad \Phi_{i0}=1,\quad \Phi_{ij}=\phi_j(z_i).$$

    The nonlinear feature transformation gives a model **linear in its weights**:

    $$\hat y=\Phi w,\qquad
    \operatorname{SSE}(w)=(y-\Phi w)^T(y-\Phi w),\qquad
    \operatorname{RMSE}=\sqrt{\operatorname{SSE}/n}.$$

    ### Derivation of the offline solution

    Expand the squared error and differentiate:

    $$\operatorname{SSE}=y^Ty-2w^T\Phi^Ty+w^T\Phi^T\Phi w,$$
    $$\nabla_w\operatorname{SSE}=2\Phi^T(\Phi w-y).$$

    Setting the gradient to zero yields the normal equations:

    $$\Phi^T\Phi w=\Phi^Ty.$$

    We solve least squares directly on $\Phi$ with `np.linalg.lstsq`, an operation
    explicitly permitted in the assignment. We avoid forming a matrix inverse or
    squaring the condition number via $\Phi^T\Phi$. For rank-deficient inputs the
    solver returns a minimum-norm solution with its documented numerical cutoff.
    No explicit regularization is used in homework 1.

    The outer training matrix has **584 rows and M + 1 columns** (including the
    intercept). All tested systems are overdetermined. Basis
    computation, design-matrix construction, SSE, RMSE, preprocessing, and split
    logic below are implemented ourselves. NumPy provides arrays, elementary
    arithmetic/exponentials, and the permitted solver. Pandas handles CSV loading
    and table display; Matplotlib plots results.
    """)
    md("""
    ### 2.1 Gaussian basis functions

    Width controls the locality of each Gaussian response and is the hyperparameter studied.
    """)
    code(functions["gaussian_basis"])
    md("""
    #### Visual intuition: width and M

    Top row: change width around the same center. Bottom row: increase the number
    of centers at fixed width. Each curve is one Gaussian basis function.
    This is a one-dimensional illustration; the actual model uses 21 input dimensions.
    """)
    code("""
    illustration_x = np.linspace(-3, 3, 500).reshape(-1, 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, width in zip(axes[0], (0.3, 1.0)):
        response = gaussian_basis(illustration_x, np.array([[0.0]]), width)
        ax.plot(illustration_x[:, 0], response[:, 0], color="#12648a", linewidth=2)
        ax.axvline(0, color="#888888", linestyle=":")
        ax.set_title(f"Same center, width = {width:g}")
    for ax, count in zip(axes[1], (3, 9)):
        illustration_centers = np.linspace(-2, 2, count).reshape(-1, 1)
        responses = gaussian_basis(illustration_x, illustration_centers, width=0.4)
        for j in range(count):
            ax.plot(illustration_x[:, 0], responses[:, j], linewidth=1.6)
        ax.set_title(f"M = {count} centers, same width = 0.4")
    for ax in axes.flat:
        ax.set(xlabel="Illustrative input value", ylabel="RBF output", ylim=(-0.02, 1.08))
        ax.grid(alpha=0.2)
    fig.suptitle("Width changes each curve's spread; M changes how many curves we use")
    plt.show()
    """)
    md("""
    ### 2.2 Design matrix

    """)
    code(functions["design_matrix"])
    md("""
    ### 2.3 Error functions

    """)
    code(functions["sse"] + "\n\n" + functions["rmse"])
    md("""
    ### 2.4 Fit the weights

    """)
    code(functions["fit_least_squares"])
    md("""
    ## 3. Fixed split, preprocessing, and center selection

    Seed 42 creates **584 training / 147 test observations** (approximately 80/20).
    Seed 43 divides only the training partition into **467 inner fitting / 117
    validation observations**. The inner fitting partition has its own scaler and
    centers. We choose the width from validation RMSE before looking at test errors,
    then refit each candidate on the full outer training partition.

    The split, 64 centers, and center-selection seed are fixed within each width
    sweep. The random split measures interpolation within the observed two years;
    correlated nearby days and the inclusion of both years in training mean this
    is **not a future forecasting evaluation**. Such a claim needs a chronological
    or blocked evaluation.
    """)
    md("""
    ### 3.1 Create the outer training/test split

    """)
    code(functions["shuffled_indices"] + "\n\n" + textwrap.dedent("""
    order = shuffled_indices(len(y), seed=42)
    cut = int(0.8 * len(y))
    train, test = order[:cut], order[cut:]
    print(f"Training days: {len(train)}; test days: {len(test)}")
    """))
    md("""
    ### 3.2 Reserve training-only validation data

    Select width on training-only validation data to avoid tuning on test errors.
    """)
    code("""
    inner_order = shuffled_indices(len(train), seed=43)
    inner_cut = int(0.8 * len(train))
    inner_fit = train[inner_order[:inner_cut]]
    validation = train[inner_order[inner_cut:]]
    print(f"Inner fitting days: {len(inner_fit)}; validation days: {len(validation)}")
    """)
    md("""
    ### 3.3 Standardize using fitting-data statistics

    Fit scaling statistics on training rows only. Standardize continuous inputs;
    leave binary and category indicators unchanged.
    """)
    code(functions["fit_scaler"] + "\n\n" + functions["transform"])
    code("""
    # Outer models: learn scaling from all outer training days.
    means, scales = fit_scaler(x[train])
    z_train = transform(x[train], means, scales)
    z_test = transform(x[test], means, scales)

    # Validation models: learn scaling from inner fitting days only.
    inner_means, inner_scales = fit_scaler(x[inner_fit])
    z_fit = transform(x[inner_fit], inner_means, inner_scales)
    z_validation = transform(x[validation], inner_means, inner_scales)
    """)
    md("""
    ### 3.4 Select RBF centers

    Sample 64 centers from fitting data and hold them fixed across the width sweep.
    """)
    code(functions["choose_centers"] + "\n\n" + textwrap.dedent("""
    center_positions = choose_centers(len(train))
    centers = z_train[center_positions]
    inner_centers = z_fit[choose_centers(len(inner_fit))]
    print(f"Centers: {len(centers)}; design-matrix columns including intercept: {len(centers) + 1}")
    """))
    md("""
    ## 4. Hyperparameter study

    First hold M = 64 fixed and test widths 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, and 12.
    First select a width using the training-only validation partition. Then fit on
    all outer training examples and calculate **training and test RMSE separately**
    at every width. Then vary the number of centers at fixed width, followed by
    a full grid evaluating all 54 pairs.
    """)
    code("WIDTHS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)\n\n" + validation
         + "\nprint('Validation-selected width:', selected_width)")
    md("""
    ### 4.1 Refit and evaluate each width

    """)
    code(experiment + '''
results_df = pd.DataFrame(results)
display(results_df[["width", "train_rmse", "test_rmse", "validation_rmse"]].round(2))
''')
    md("""
    ### 4.2 Compare the RMSE curves

    """)
    code("""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for key, label, color in [("train_rmse", f"Training ({len(train)} days)", "#12648a"),
                              ("test_rmse", f"Test ({len(test)} days)", "#cb5229"),
                              ("validation_rmse", f"Inner validation ({len(validation)} days)", "#66852c")]:
        ax.plot(WIDTHS, [r[key] for r in results], "o-", label=label, color=color)
    ax.axvline(selected_width, linestyle=":", color="#666666", label="Validation-selected width")
    ax.set(xscale="log", xlabel="Gaussian RBF width (sigma, logarithmic scale)",
           ylabel="RMSE (rentals/day)", title="Effect of RBF width - fixed centers and split")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    plt.show()
    """)
    md("""
    ### 4.3 Width-study model and constant baseline

    Use the training-mean predictor as a baseline for interpreting the model's RMSE.
    """)
    code("""
    train_mean = sum(float(value) for value in y[train]) / len(train)
    baseline_train = rmse(y[train], np.full(len(train), train_mean))
    baseline_test = rmse(y[test], np.full(len(test), train_mean))
    chosen = next(r for r in results if r["width"] == selected_width)
    lowest_test = min(results, key=lambda r: r["test_rmse"])
    improvement = 100 * (1 - chosen["test_rmse"] / baseline_test)
    display(Markdown(
        f"**Validation-selected width: {selected_width:g}**  \\n"
        f"Training RMSE: **{chosen['train_rmse']:.2f}**; test RMSE: **{chosen['test_rmse']:.2f} rentals/day**.  \\n"
        f"Training-mean baseline: training RMSE {baseline_train:.2f}, test RMSE {baseline_test:.2f}.  \\n"
        f"Test RMSE reduction relative to this baseline: **{improvement:.1f}%**.  \\n"
        f"Descriptive minimum test RMSE: {lowest_test['test_rmse']:.2f} at width {lowest_test['width']:g}; "
        "we retain the validation-selected width."
    ))
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    ax.scatter(y[test], selected_prediction, alpha=0.7, s=24, color="#12648a")
    low = min(0, float(min(selected_prediction)))
    high = max(float(max(y[test])), float(max(selected_prediction)))
    ax.plot([low, high], [low, high], "--", color="#666666", label="Perfect prediction")
    ax.set(xlabel="Observed rentals/day", ylabel="Predicted rentals/day",
           title=f"Held-out predictions - width {selected_width:g}")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
    """)
    md("""
    ### 4.4 Study the number of centers

    Fix width at the validation-selected value and test **M = 8, 16, 32, 64, 128, 256**.
    Use the same split and scaling. Each larger center set extends the same shuffled
    prefix, so the basis sets are nested at this fixed width. Training SSE should
    therefore not increase, apart from numerical effects. All fitting systems still
    have more observations than coefficients.
    """)
    code(functions["fit_rbf_model"])
    code("CENTER_COUNTS = (8, 16, 32, 64, 128, 256)\n\n" + count_study + "\n" + textwrap.dedent("""
    count_results_df = pd.DataFrame(count_results)
    display(count_results_df[["center_count", "train_rmse", "test_rmse", "validation_rmse"]].round(2))
    count_chosen = next(r for r in count_results if r["center_count"] == selected_count)
    display(Markdown(
        f"**Validation-selected M: {selected_count}, width: {selected_width:g}.** "
        f"Training RMSE: **{count_chosen['train_rmse']:.2f}**; "
        f"test RMSE: **{count_chosen['test_rmse']:.2f} rentals/day**."
    ))
    """))
    md("""
    ### 4.5 RMSE versus number of centers
    """)
    code("""
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
    plt.show()
    """)
    md("""
    ### 4.6 Full grid: choose width and center count jointly

    Evaluate **6 center counts × 9 widths = 54 combinations**. The same splits,
    scaling, and shuffled center prefixes are used for every pair. Select the pair
    with the lowest inner validation RMSE before calculating test-grid errors.
    """)
    code(grid_validation + "\nprint('Validation-selected pair:', grid_best_validation)")
    code(grid_evaluation + "\n" + textwrap.dedent("""
    grid_results_df = pd.DataFrame(grid_results)
    validation_grid = np.array([r["validation_rmse"] for r in grid_results]).reshape(len(CENTER_COUNTS), len(WIDTHS))
    display(Markdown("**Validation RMSE: rows = center count, columns = width**"))
    display(pd.DataFrame(validation_grid, index=CENTER_COUNTS, columns=WIDTHS).round(2))
    grid_chosen = next(r for r in grid_results if r["center_count"] == grid_selected_count
                       and r["width"] == grid_selected_width)
    display(Markdown(
        f"**Joint selection: M = {grid_selected_count}, width = {grid_selected_width:g}.** "
        f"Train RMSE: **{grid_chosen['train_rmse']:.2f}**; "
        f"validation RMSE: **{grid_chosen['validation_rmse']:.2f}**; "
        f"test RMSE: **{grid_chosen['test_rmse']:.2f} rentals/day**."
    ))
    """))
    md("""
    ### 4.7 Training, validation, and test heatmaps

    Each cell represents one fitted setting. All panels share the same color scale;
    orange boxes mark the validation-selected pair on all three panels.
    """)
    code(functions["plot_hyperparameter_grid"] + "\n\n"
         + "fig = plot_hyperparameter_grid(grid_results, grid_selected_count, grid_selected_width)\nplt.show()")
    md("""
    ### 4.8 The selected model's RBF features

    With M = 256, each day becomes 256 RBF values. The heatmap below shows 30 test
    days against all centers: brighter cells mean greater similarity to that center.
    This is the actual basis matrix, before adding the intercept or multiplying by
    learned weights. Center order follows the seeded sample, not a spatial ordering.

    The model has 257 weights for 584 outer training days (and 467 inner fitting
    days during validation). This is a relatively flexible model. Its selection at
    the largest tested M does not prove that 256 centers are necessary or globally optimal.
    """)
    code("""
    view_positions = np.argsort(test)[:30]
    displayed_rows = test[view_positions]
    activations = gaussian_basis(z_test[view_positions], grid_selected_centers, grid_selected_width)
    fig, ax = plt.subplots(figsize=(11, 7.5), layout="constrained")
    image = ax.imshow(activations, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(displayed_rows)), labels=[df["date"].iloc[int(i)] for i in displayed_rows], fontsize=8)
    ticks = list(range(0, grid_selected_count, 32))
    ax.set_xticks(ticks, labels=[str(i + 1) for i in ticks])
    ax.set(xlabel="RBF center index (sampled order)", ylabel="Test day",
           title=f"Actual RBF features: {len(displayed_rows)} days x {grid_selected_count} centers, width {grid_selected_width:g}")
    fig.colorbar(image, ax=ax, label="RBF output (similarity)")
    plt.show()
    """)
    md("""
    ## 5. Discussion and conclusions

    **Width study (M = 64).**

    Validation selected **width 2.0**, giving training RMSE **620.31** and test RMSE
    **697.09 rentals/day**. The training-mean baseline has test RMSE **1910.15**,
    so the selected model reduces test RMSE by **63.5%**.

    Narrow RBFs (width 0.5) give high training and test errors because they respond
    only near their centers. Intermediate widths represent the data better. Very
    broad RBFs become similar to one another, and test error rises slightly.
    Training error need not decrease monotonically because changing width does not
    create a nested sequence of models.

    Width 3 has the lowest test RMSE in this sweep (**679.55**). We retain width 2,
    chosen using training-only validation, to avoid tuning on the test set.

    **Center-count study (width = 2).** Training RMSE falls from **1259.38** at
    M = 8 to **400.00** at M = 256. Validation selects **M = 128**, giving training
    RMSE **514.28**, validation RMSE **718.93**, and test RMSE **699.28**.
    At M = 256 the lower training error accompanies higher validation and test errors
    (**722.04** and **754.98**), consistent with overfitting as capacity increases.

    M = 64 has slightly lower test RMSE than M = 128 at this width; the selected count
    remains 128 because selection uses validation. These are two sequential sweeps:
    width is selected at M = 64, then M is selected at that width. Width is not
    re-tuned separately for each center count.

    **Full grid.** Joint validation selects **M = 256, width = 1.5**, with validation
    RMSE **700.78**. After refitting on all outer training rows, train/test RMSE are
    **429.42 / 731.94**. This is the final selection from the tested grid. Its lower
    validation error does not translate into a lower test error than the sequential
    choice: searching more candidates does not guarantee better generalization on
    a single split. The full grid reveals how the preferred width changes with M.

    The minimum descriptive test RMSE over the grid is **679.55** at M = 64 and
    width = 3. We report that value but do not use it to choose the model.

    These results apply to this fixed random split. Nearby days can be correlated;
    a chronological split would be needed to evaluate future forecasting accuracy.
    """)
    md("""
    ## 6. References

    - Instructor's *Linear Regression - Programming Assignment - Part 1* brief.
    - Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository.
      [Dataset DOI](https://doi.org/10.24432/C5W894), [UCI record](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).
    - Fanaee-T, H. and Gama, J. (2013). *Event labeling combining ensemble detectors
      and background knowledge*. [Paper DOI](https://doi.org/10.1007/s13748-013-0040-3).
    - Original dataset archive's `Readme.txt`, preserved in the course workspace and submission ZIP.
    - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3.
    - [NumPy least-squares solver documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).

    """)
    notebook = dict(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (uv workspace)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }, nbformat=4, nbformat_minor=5)
    target = HERE / "assignment1.ipynb"
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {target}: {len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} code cells")


if __name__ == "__main__":
    main()
