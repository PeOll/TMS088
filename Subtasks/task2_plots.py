from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


Interpolator = Callable[..., pd.DataFrame]


def plot_bridge_log_price_construction(
    prices: pd.DataFrame,
    interpolated_gap_values: pd.DataFrame,
    internal_gaps: pd.DataFrame,
    output_path: Path,
    context_window: int = 40,
    z_value: float = 1.96,
) -> None:
    """Show the Brownian-bridge mean and interval on the log-price scale."""
    fig, axes = plt.subplots(len(internal_gaps), 1, figsize=(12, 2.8 * len(internal_gaps)), squeeze=False)

    for row_idx, gap in enumerate(internal_gaps.itertuples(index=False)):
        ax = axes[row_idx, 0]
        left = max(int(prices.index.min()), gap.start - context_window)
        right = min(int(prices.index.max()), gap.end + context_window)
        window_index = range(left, right + 1)
        observed_log_price = np.log(prices.loc[window_index, gap.series])
        bridge = interpolated_gap_values.loc[gap.series]

        ax.plot(observed_log_price.index, observed_log_price, color="0.55", label="observed log-price")
        ax.scatter(
            [gap.start - 1, gap.end + 1],
            np.log(prices.loc[[gap.start - 1, gap.end + 1], gap.series]),
            color="black",
            s=24,
            zorder=3,
            label="conditioning endpoints" if row_idx == 0 else None,
        )
        ax.plot(
            bridge.index,
            bridge["log_price_mean"],
            color="tab:blue",
            linewidth=2,
            label="bridge mean" if row_idx == 0 else None,
        )
        ax.fill_between(
            bridge.index,
            bridge["log_price_mean"] - z_value * bridge["log_price_std"],
            bridge["log_price_mean"] + z_value * bridge["log_price_std"],
            color="tab:blue",
            alpha=0.18,
            label="95% bridge interval" if row_idx == 0 else None,
        )
        ax.axvspan(gap.start, gap.end, color="tab:orange", alpha=0.08)
        ax.set_title(f"{gap.series}: bridge construction for days {gap.start}-{gap.end}")
        ax.set_ylabel("log-price")

    axes[0, 0].legend(loc="upper left")
    axes[-1, 0].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_bridge_uncertainty_profiles(
    interpolated_gap_values: pd.DataFrame,
    internal_gaps: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the bridge log-price standard deviation inside each true gap."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for gap in internal_gaps.itertuples(index=False):
        bridge = interpolated_gap_values.loc[gap.series]
        hidden_step = np.arange(1, len(bridge) + 1)
        ax.plot(hidden_step, bridge["log_price_std"], label=gap.series)

    ax.set_title("Bridge uncertainty profile inside each internal gap")
    ax.set_xlabel("hidden day within gap")
    ax.set_ylabel("log-price standard deviation")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_backtest_metric_summary(overall_backtest: pd.DataFrame, output_path: Path) -> None:
    """Plot the main backtest metrics for model comparison."""
    metrics = [
        ("log_price_rmse", "Log-price RMSE", None),
        ("coverage_95", "95% interval coverage", 0.95),
        ("avg_interval_width", "Average interval width", None),
    ]
    summary = overall_backtest.reset_index()
    model_col = "model" if "model" in summary.columns else summary.columns[0]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)

    for ax, (metric, title, reference) in zip(axes[0], metrics):
        values = summary[metric].replace([np.inf, -np.inf], np.nan)
        ax.bar(summary[model_col], values.fillna(0), color="tab:blue", alpha=0.75)
        for tick, value in enumerate(values):
            if pd.isna(value):
                ax.text(tick, 0, "non-finite", rotation=90, va="bottom", ha="center", fontsize=8)
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("model")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_backtest_example_windows(
    prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    backtest_results: pd.DataFrame,
    interpolation_models: dict[str, Interpolator],
    output_path: Path,
    gap_length: int,
    variance_window: int = 60,
    z_value: float = 1.96,
    context_window: int = 40,
    max_examples: int = 3,
) -> None:
    """Plot selected pseudo-gaps with true hidden prices and model estimates."""
    bridge_results = backtest_results[backtest_results["model"] == "bridge"].dropna(subset=["log_price_rmse"])
    example_rows = bridge_results.sort_values("log_price_rmse", ascending=False).head(max_examples)
    if example_rows.empty:
        return

    fig, axes = plt.subplots(len(example_rows), 1, figsize=(12, 3.2 * len(example_rows)), squeeze=False)

    for row_idx, row in enumerate(example_rows.itertuples(index=False)):
        ax = axes[row_idx, 0]
        start = int(row.start)
        end = int(row.end)
        series_name = row.series
        left = max(int(prices.index.min()), start - context_window)
        right = min(int(prices.index.max()), end + context_window)
        window_index = range(left, right + 1)

        ax.plot(prices.loc[window_index, series_name].index, prices.loc[window_index, series_name], color="0.75", label="true context")
        ax.plot(prices.loc[start:end, series_name].index, prices.loc[start:end, series_name], color="black", linewidth=2, label="hidden truth")

        for model_name, interpolator in interpolation_models.items():
            try:
                interpolation = interpolator(
                    prices[series_name],
                    log_returns[series_name],
                    start,
                    end,
                    variance_window=variance_window,
                    z_value=z_value,
                )
            except Exception:
                continue

            finite_estimate = np.isfinite(interpolation["estimate"])
            if not finite_estimate.any():
                continue

            ax.plot(
                interpolation.index[finite_estimate],
                interpolation.loc[finite_estimate, "estimate"],
                linewidth=1.8,
                label=model_name,
            )
            if model_name == "bridge":
                finite_interval = finite_estimate & np.isfinite(interpolation["lower_95"]) & np.isfinite(interpolation["upper_95"])
                ax.fill_between(
                    interpolation.index[finite_interval],
                    interpolation.loc[finite_interval, "lower_95"],
                    interpolation.loc[finite_interval, "upper_95"],
                    color="tab:blue",
                    alpha=0.15,
                    label="bridge 95% interval",
                )

        ax.axvspan(start, end, color="tab:orange", alpha=0.08)
        ax.set_title(f"{series_name}: pseudo-gap backtest days {start}-{end}")
        ax.set_ylabel("price")

    axes[0, 0].legend(loc="upper left", ncol=2)
    axes[-1, 0].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_true_gap_model_comparison(
    prices: pd.DataFrame,
    true_gap_model_values: pd.DataFrame,
    internal_gaps: pd.DataFrame,
    output_path: Path,
    context_window: int = 40,
) -> None:
    """Overlay model estimates for the true internal gaps."""
    if true_gap_model_values.empty:
        return

    true_gap_model_values = true_gap_model_values.sort_index()
    fig, axes = plt.subplots(len(internal_gaps), 1, figsize=(12, 2.9 * len(internal_gaps)), squeeze=False)
    models = list(true_gap_model_values.index.get_level_values("model").unique())

    for row_idx, gap in enumerate(internal_gaps.itertuples(index=False)):
        ax = axes[row_idx, 0]
        left = max(int(prices.index.min()), gap.start - context_window)
        right = min(int(prices.index.max()), gap.end + context_window)
        window_index = range(left, right + 1)

        ax.plot(prices.loc[window_index, gap.series].index, prices.loc[window_index, gap.series], color="0.65", label="observed price")

        for model_name in models:
            try:
                model_frame = true_gap_model_values.xs((model_name, gap.series), level=("model", "series"))
            except KeyError:
                continue

            finite_estimate = np.isfinite(model_frame["estimate"])
            if not finite_estimate.any():
                continue

            ax.plot(
                model_frame.index[finite_estimate],
                model_frame.loc[finite_estimate, "estimate"],
                linewidth=1.8,
                label=model_name,
            )
            if model_name == "bridge":
                finite_interval = finite_estimate & np.isfinite(model_frame["lower_95"]) & np.isfinite(model_frame["upper_95"])
                ax.fill_between(
                    model_frame.index[finite_interval],
                    model_frame.loc[finite_interval, "lower_95"],
                    model_frame.loc[finite_interval, "upper_95"],
                    color="tab:blue",
                    alpha=0.15,
                    label="bridge 95% interval" if row_idx == 0 else None,
                )

        ax.axvspan(gap.start, gap.end, color="tab:orange", alpha=0.08)
        ax.set_title(f"{gap.series}: model comparison for true gap {gap.start}-{gap.end}")
        ax.set_ylabel("price")

    axes[0, 0].legend(loc="upper left", ncol=2)
    axes[-1, 0].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
