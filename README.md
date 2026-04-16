# TMS088
Course Project in Financial Time Series Spring 2026
[Project Report in Overleaf](https://www.overleaf.com/3716575738zyrvtvnbwcns#c841af)

## Repository Structure

This repository is organized around the project tasks from the course PDF.

- `Project_Financial_Timeseries_2526.pdf`
  The assignment specification.
- `Data/`
  The CSV file with the seven Spiff price series.
- `Subtasks/task1.ipynb`
  Exploratory analysis for Task 1.
- `Subtasks/task2.ipynb`
  Interpolation of the internal missing gaps for Task 2.
- `Subtasks/series_analysis.py`
  Shared data-loading and calculation helpers used by the notebooks.
- `config/plot_config.py`
  Plot styling used in the notebooks.

The general pattern is:

1. Keep reusable calculations in `Subtasks/series_analysis.py`.
2. Keep presentation, plots, and task-specific explanations inside the notebooks.
3. Let each notebook import the shared helper module rather than redefining loading and transformation logic.

## Loading The Data In A New Notebook

If you create a new notebook in this repository, these are the lines you need in order to access the data through the shared helper module:

```python
from pathlib import Path
import sys

repo_root = Path.cwd()
if not (repo_root / "config").exists():
    repo_root = repo_root.parent
sys.path.append(str(repo_root))

from Subtasks.series_analysis import PriceAnalysis

raw_data = PriceAnalysis.from_csv("spiff_data-2.csv")
raw_prices = raw_data.prices

analysis = raw_data.clean()
cleaned_prices = analysis.prices
```

`raw_prices` gives the original data as read from the CSV. `cleaned_prices` gives the same data after replacing the all-1000 placeholder rows with `NaN`.

If you only want the transformed return data in a notebook, continue with:

```python
log_returns = analysis.log_returns
absolute_log_returns = analysis.absolute_log_returns
squared_log_returns = analysis.squared_log_returns
```
