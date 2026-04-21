# TMS088

Course project in Financial Time Series, Spring 2026.

[Project report in Overleaf](https://www.overleaf.com/3716575738zyrvtvnbwcns#c841af)


## Repository Structure

- `Project_Financial_Timeseries_2526.pdf`
  Assignment specification.
- `Data/spiff_data-2.csv`
  Raw Spiff price data with seven series: `gurkor`, `guitars`, `slingshots`, `stocks`, `sugar`, `water`, and `tranquillity`.
- `Subtasks/`
  Jupyter notebooks for the four project tasks plus the shared `series_analysis.py` module.
- `config/plot_config.py`
  Shared Matplotlib styling.
- `Output/`
  Generated outputs that might be useful in the report. This folder is synced to Overleaf.
- `.github/workflows/sync-output-to-overleaf.yml`
  GitHub Actions workflow that syncs `Output/` into the Overleaf-linked report repository.


## Setup

Create a Python environment and install the notebook dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, and `arch`. The `arch` package is needed for the optional GARCH candidate in Task 2.

## Data Workflow

The raw file contains placeholder rows where every asset is exactly `1000`. The shared cleaning step treats those rows as corrupted observations and replaces them with missing values. After cleaning, each asset has one internal 50-day missing block, and all assets share a final 200-day trailing missing block.

The notebooks generally follow this pattern:

1. Load the raw prices through `PriceAnalysis.from_csv("spiff_data-2.csv")`.
2. Clean the all-1000 placeholder rows with `.clean()`.
3. Work mainly with log-returns rather than raw price levels.
4. Save report-ready figures and tables under `Output/taskX/`.

Use this setup in new notebooks (located in the `Subtasks` folder):

```python
from pathlib import Path
import sys

repo_root = Path.cwd().parent
sys.path.append(str(repo_root))

from Subtasks.series_analysis import PriceAnalysis

raw_data = PriceAnalysis.from_csv("spiff_data-2.csv")
raw_prices = raw_data.prices

analysis = raw_data.clean()
cleaned_prices = analysis.prices

log_returns = analysis.log_returns
absolute_log_returns = analysis.absolute_log_returns
squared_log_returns = analysis.squared_log_returns
gap_table = analysis.gap_table()
```

Useful helper methods include:

- `analysis.overview()` for data types and missing-value counts.
- `analysis.placeholder_rows()` for all-1000 placeholder rows.
- `analysis.gap_table()` for internal and trailing missing blocks.
- `analysis.series_comparison()` for the combined Task 1 diagnostic table.
- `analysis.interpolate_internal_gap(column)` for the default log-price bridge interpolation.

## Generated Outputs And Overleaf Sync

Generated report resources live in `Output/`. The GitHub workflow syncs that folder to `Figures/resources/` in the Overleaf-linked repository when a push to `main` changes files under `Output/**`, or when the workflow is run manually.

After the workflow runs, open Overleaf and sync from GitHub so the report sees the latest figures and tables.

![Overleaf sync instructions](nonessential/Overleaf%20Instructions.png)
