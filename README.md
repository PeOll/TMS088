# TMS088 — Financial Time Series

Course project, Spring 2026.

The assignment specification is in [`Project_Financial_Timeseries_2526.pdf`](Project_Financial_Timeseries_2526.pdf).

---

## Repository Structure

```
TMS088/
├── Data/
│   └── spiff_data-2.csv          Raw Spiff price data (7 series)
├── Subtasks/
│   ├── helpers.py                 DataAnalysis helper class (shared across all tasks)
│   ├── task1.ipynb                Exploratory time-series analysis
│   ├── task2.ipynb                Internal gap interpolation
│   ├── task3.ipynb                Trailing gap extrapolation
│   ├── task4.ipynb                Algorithmic trading strategies & portfolio allocation
│   └── Development Phase Notebooks/
│       └── task4-merged.ipynb     Earlier collaborative draft merged into the final task4
├── config/
│   └── plot_config.py             Shared Matplotlib styling and output path
├── Output/
│   ├── task1/                     Figures and tables for Task 1
│   ├── task2/                     Figures and tables for Task 2
│   ├── task3/                     Figures and tables for Task 3
│   └── task4/                     Figures and tables for Task 4
└── requirements.txt
```

---

## Tasks

| Notebook | Topic |
|----------|-------|
| `task1.ipynb` | Exploratory analysis — stationarity, autocorrelation, volatility clustering, co-integration, PCA clustering |
| `task2.ipynb` | Internal gap filling — Brownian bridge and GARCH-based interpolation of the 50-day missing blocks |
| `task3.ipynb` | Trailing gap extrapolation — GARCH(1,1)-t forward simulation of the 200-day trailing missing block |
| `task4.ipynb` | Algorithmic trading strategies (MA crossover, pairs trading, long-short pairs) with walk-forward validation, bootstrap significance testing, and mean-variance portfolio allocation |

### Development Phase Notebooks

`Subtasks/Development Phase Notebooks/` contains earlier working drafts from the individual collaborators. These were later reviewed and merged into the final notebooks in `Subtasks/`. They are kept for reference but are not the authoritative versions.

---

## Data

`Data/spiff_data-2.csv` contains daily price levels for seven synthetic assets:

| Asset | Description |
|-------|-------------|
| `gurkor` | — |
| `guitars` | — |
| `slingshots` | — |
| `stocks` | — |
| `sugar` | — |
| `water` | — |
| `tranquillity` | — |

**Known data issues handled by the cleaning step:**
- Rows where every asset equals exactly `1000` are placeholder/corrupted observations and are replaced with `NaN`.
- Each asset has one internal ~50-day missing block and a shared ~200-day trailing missing block.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `arch`, `scikit-learn`.

---

## Usage

### helpers.py

`helpers.py` is heaviest in Task 1, where the `DataAnalysis` class centralises all the cleaning and diagnostic logic so the same pipeline can be reused without copy-pasting across notebooks. Some bulkier helper functions are also kept there simply to avoid cluttering the notebook cells with code that is correct but not interesting to read inline.

From Task 2 onward the reliance on `helpers.py` decreases deliberately — later tasks define their key functions directly in the notebook so the logic is visible and easy to follow without jumping to an external file.

### Data loading

All notebooks share the same data loading pattern:

```python
from pathlib import Path
import sys

repo_root = Path.cwd().parent
sys.path.append(str(repo_root))

from helpers import DataAnalysis

analysis = DataAnalysis.from_csv("spiff_data-2.csv").clean()
prices   = analysis.prices
```

Useful `DataAnalysis` methods:

| Method | Returns |
|--------|---------|
| `analysis.log_returns` | Daily log-returns |
| `analysis.gap_table()` | Summary of internal and trailing missing blocks |
| `analysis.overview()` | Data types and missing-value counts |
| `analysis.placeholder_rows()` | All-1000 placeholder rows |
| `analysis.interpolate_internal_gap(column)` | Log-price bridge interpolation for one asset |

Plots and tables are saved under `Output/taskX/` via `config/plot_config.py`:

```python
from config.plot_config import plot_config

output = plot_config()          # applies shared Matplotlib style, returns Output/ path
output_task = output / "task4"
output_task.mkdir(exist_ok=True)

fig.savefig(output_task / "my_figure.png", bbox_inches="tight")
```
