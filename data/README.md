# Suggested datasets for homeworks 1 and 2

Instructor-provided instruction:

> برای انجام پروژه های تکالیف ۱ و ۲ یکی از دیتاست‌های زیر را انتخاب کنید:

Choose one of the following datasets for the projects in homeworks 1 and 2. Homework 2 extends homework 1 using the same dataset.

| # | Dataset | Source |
| --- | --- | --- |
| 1 | Indian Biogas Production | [Kaggle](https://www.kaggle.com/datasets/ucimachinelearning/indian-biogas-production-dataset) |
| 2 | Used car price | [Kaggle](https://www.kaggle.com/datasets/lepchenkov/usedcarscatalog) |
| 3 | Ore concentrate impurity | [Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) |
| 4 | Infrared Thermography Temperature | [UCI](https://archive.ics.uci.edu/dataset/925/infrared+thermography+temperature+dataset) |
| 5 | Instagram | [Kaggle](https://www.kaggle.com/datasets/kundanbedmutha/instagram-analytics-dataset/data) |
| 6 | Bike sharing | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |
| 7 | Miami Housing | [Kaggle](https://www.kaggle.com/datasets/deepcontractor/miami-housing-dataset) |

Links are preserved as supplied. The daily **UCI Bike Sharing** dataset has been selected and downloaded for homework 1, to be reused for homework 2. The other six sources have not been verified or downloaded.

## Storage

- `raw/`: original downloads, archives, and source documentation. Preserve the downloaded data unchanged.
- `processed/`: cleaned data and saved experiment splits derived from raw data.

Data files are excluded from version control by default. Record the exact source URL, downloaded filename/version, retrieval date, license, target column, chosen features, collection methodology, preprocessing decisions, and split seed when selecting a dataset. Do not invent provenance where the source does not document it.

## Selection record

- Selected dataset: UCI Bike Sharing, `day.csv`, 731 daily observations from 2011-01-01 to 2012-12-31.
- Regression target: `cnt`, total daily rentals. `casual` and `registered` are excluded because their sum equals the target.
- Original collection source and methodology: Capital Bikeshare operational rental logs from Washington, D.C.; Hadi Fanaee-T aggregated observations and added weather from Freemeteo and calendar information. The download script preserves the original dataset README locally.
- Download date: 2026-11-20. [UCI source](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); [dataset DOI](https://doi.org/10.24432/C5W894); current UCI license: CC BY 4.0. The archived README requests citation of [Fanaee-T and Gama (2013)](https://doi.org/10.1007/s13748-013-0040-3).
- Source archive: `bike-sharing.zip` (downloaded locally). SHA-256: `b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401`. The download script checks this hash and refuses to overwrite modified raw files.
- Features: temperature, humidity, wind speed; year, holiday, working-day indicators; season, weekday, and weather categories. Three continuous inputs are standardized on fitting rows only, binary variables remain unchanged, and category sets are encoded manually as indicators (21 numeric inputs total).
- Excluded inputs: record index, raw date, month, apparent temperature, casual rentals, and registered rentals. No missing-value imputation, outlier removal, or target transformation.
- Split strategy: seed 42 shuffles rows into 584 training and 147 test observations. A training-only split with seed 43 creates 467 inner-fitting and 117 validation observations for width selection. Seed 44 selects 64 centers for the width study; the subsequent center-count study uses nested prefixes of the same shuffled fitting rows for M = 8, 16, 32, 64, 128, 256. The full-grid study tests all 54 combinations of these six center counts with the nine widths, retaining the same splits and center prefixes. Exact row assignments and preprocessing parameters are saved with [homework 1 results](../assignments/hw01-linear-regression/results/).
- Source documentation caveat: current UCI metadata and the archived README disagree about season names and temperature normalization. The implementation uses the original season codes categorically and keeps normalized temperature without attempting an undocumented inverse conversion.

The random split evaluates interpolation within the observed years, not future forecasting. Reuse the saved outer split for homework 2; do not claim chronological generalization from these results.

Before choosing, check target suitability, missing values and categorical variables, and whether repeated subjects or temporal structure could cause leakage. Also consider whether an expressive basis can demonstrate the near-interpolation behavior required in homework 2 at a practical computation cost.
