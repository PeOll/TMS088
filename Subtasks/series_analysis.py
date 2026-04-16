from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import pacf


def _default_data_dir() -> Path:
    module_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        Path.cwd() / "Data",
        Path.cwd().parent / "Data",
        module_root / "Data",
    ]

    for candidate in candidate_dirs:
        if candidate.exists():
            return candidate

    return module_root / "Data"


def read_data(file_name: str, data_dir: Path | None = None) -> pd.DataFrame:
    base_dir = data_dir if data_dir is not None else _default_data_dir()
    data = pd.read_csv(base_dir / file_name, index_col=1)
    return data.drop(columns=["Unnamed: 0"])


def clean_data(prices: pd.DataFrame, placeholder_value: float = 1000) -> pd.DataFrame:
    cleaned = prices.copy()
    placeholder_mask = (cleaned == placeholder_value).all(axis=1)
    cleaned.loc[placeholder_mask] = np.nan
    return cleaned


class PriceAnalysis:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.copy()

    @classmethod
    def from_csv(cls, file_name: str) -> "PriceAnalysis":
        return cls(read_data(file_name))

    def clean(self, placeholder_value: float = 1000) -> "PriceAnalysis":
        return PriceAnalysis(clean_data(self.prices, placeholder_value=placeholder_value))

    @property
    def log_returns(self) -> pd.DataFrame:
        return np.log(self.prices).diff()

    @property
    def absolute_log_returns(self) -> pd.DataFrame:
        return self.log_returns.abs()

    @property
    def squared_log_returns(self) -> pd.DataFrame:
        return self.log_returns.pow(2)

    def overview(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dtype": self.prices.dtypes.astype(str),
                "missing_values": self.prices.isna().sum(),
                "missing_share": self.prices.isna().mean(),
            }
        )

    def placeholder_rows(self, placeholder_value: float = 1000) -> pd.DataFrame:
        placeholder_mask = (self.prices == placeholder_value).all(axis=1)
        return self.prices.loc[placeholder_mask]

    def _missing_intervals(self, series: pd.Series) -> list[tuple[int, int, int]]:
        missing_index = series.index[series.isna()]
        intervals: list[tuple[int, int, int]] = []
        start = prev = None

        for idx in missing_index:
            if start is None:
                start = prev = idx
            elif idx == prev + 1:
                prev = idx
            else:
                length = prev - start + 1
                if length > 1:
                    intervals.append((start, prev, length))
                start = prev = idx

        if start is not None:
            length = prev - start + 1
            if length > 1:
                intervals.append((start, prev, length))

        return intervals

    def missing_summary(self) -> pd.DataFrame:
        summary = pd.DataFrame(
            {
                column: [f"{start}-{end} ({length} days)" for start, end, length in self._missing_intervals(self.prices[column])]
                for column in self.prices.columns
            }
        ).T
        summary.columns = [f"block_{i + 1}" for i in range(summary.shape[1])]
        return summary

    def gap_table(self) -> pd.DataFrame:
        rows = []
        final_index = self.prices.index.max()

        for column in self.prices.columns:
            for start, end, length in self._missing_intervals(self.prices[column]):
                rows.append(
                    {
                        "series": column,
                        "start": start,
                        "end": end,
                        "length": length,
                        "gap_type": "trailing" if end == final_index else "internal",
                    }
                )

        return pd.DataFrame(rows).sort_values(["series", "start"]).reset_index(drop=True)

    def rolling_mean(self, window: int = 30) -> pd.DataFrame:
        return self.log_returns.rolling(window).mean()

    def rolling_variance(self, window: int = 30) -> pd.DataFrame:
        return self.log_returns.rolling(window).var()

    def _single_series_summary(self, window: int = 30) -> pd.DataFrame:
        rolling_mean = self.rolling_mean(window)
        rolling_var = self.rolling_variance(window)

        return pd.DataFrame(
            {
                "mean": self.log_returns.mean(),
                "variance": self.log_returns.var(),
                "std": self.log_returns.std(),
                "min": self.log_returns.min(),
                "max": self.log_returns.max(),
                "skewness": self.log_returns.skew(),
                "kurtosis": self.log_returns.kurtosis(),
                "avg_30d_mean": rolling_mean.mean(),
                "avg_30d_variance": rolling_var.mean(),
            }
        ).sort_values("variance", ascending=False)

    def sample_acf(self, series: pd.Series, max_lag: int = 30) -> np.ndarray:
        clean = series.dropna()
        centered = clean - clean.mean()
        values = centered.to_numpy()

        if len(values) <= max_lag:
            raise ValueError("Series is too short for the requested number of lags.")

        denominator = np.dot(values, values)
        acf_values = [1.0]

        for lag in range(1, max_lag + 1):
            numerator = np.dot(values[lag:], values[:-lag])
            acf_values.append(numerator / denominator)

        return np.array(acf_values)

    def sample_pacf(self, series: pd.Series, max_lag: int = 30, method: str = "ywm") -> np.ndarray:
        clean = series.dropna()

        if len(clean) <= max_lag:
            raise ValueError("Series is too short for the requested number of lags.")

        return pacf(clean.to_numpy(), nlags=max_lag, method=method)

    def _acf_summary(self, data: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
        summary = []

        for column in data.columns:
            series = data[column].dropna()
            acf_values = self.sample_acf(series, max_lag=max_lag)
            confidence_band = 1.96 / np.sqrt(len(series))
            significant_lags = [lag for lag in range(1, max_lag + 1) if abs(acf_values[lag]) > confidence_band]
            strongest_lag = int(np.argmax(np.abs(acf_values[1:])) + 1)

            summary.append(
                {
                    "series": column,
                    "strongest_lag": strongest_lag,
                    "acf_at_strongest_lag": acf_values[strongest_lag],
                    "significant_lags_up_to_30": len(significant_lags),
                }
            )

        return pd.DataFrame(summary).set_index("series").sort_values("significant_lags_up_to_30", ascending=False)

    def ljung_box_summary(self, data: pd.DataFrame, lags: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
        rows = []

        for column in data.columns:
            result = acorr_ljungbox(data[column].dropna(), lags=list(lags), return_df=True)
            row: dict[str, float] = {"series": column}

            for lag in lags:
                row[f"lb_stat_lag_{lag}"] = result.loc[lag, "lb_stat"]
                row[f"lb_pvalue_lag_{lag}"] = result.loc[lag, "lb_pvalue"]

            rows.append(row)

        return pd.DataFrame(rows).set_index("series")

    def correlation_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.corr()

    def interpolate_internal_gap(self, column: str, variance_window: int = 60, z_value: float = 1.96) -> pd.DataFrame:
        gaps = self.gap_table()
        internal_gap = gaps[(gaps["series"] == column) & (gaps["gap_type"] == "internal")]

        if len(internal_gap) != 1:
            raise ValueError(f"Expected exactly one internal gap for {column}, found {len(internal_gap)}.")

        start = int(internal_gap.iloc[0]["start"])
        end = int(internal_gap.iloc[0]["end"])
        length = int(internal_gap.iloc[0]["length"])
        series = self.prices[column]

        left_price = series.loc[start - 1]
        right_price = series.loc[end + 1]
        if pd.isna(left_price) or pd.isna(right_price):
            raise ValueError(f"Gap endpoints for {column} must be observed.")

        left_log_price = np.log(left_price)
        right_log_price = np.log(right_price)
        steps = np.arange(1, length + 1)
        bridge_mean = left_log_price + steps / (length + 1) * (right_log_price - left_log_price)

        return_series = self.log_returns[column]
        local_returns = pd.concat(
            [
                return_series.loc[: start - 1].dropna().tail(variance_window),
                return_series.loc[end + 2 :].dropna().head(variance_window),
            ]
        )
        local_variance = local_returns.var(ddof=1)
        if pd.isna(local_variance) or local_variance <= 0:
            local_variance = return_series.dropna().var(ddof=1)

        bridge_variance = local_variance * steps * (length + 1 - steps) / (length + 1)
        bridge_std = np.sqrt(bridge_variance)
        gap_index = pd.Index(range(start, end + 1), name=self.prices.index.name)

        return pd.DataFrame(
            {
                "series": column,
                "estimate": np.exp(bridge_mean),
                "lower_95": np.exp(bridge_mean - z_value * bridge_std),
                "upper_95": np.exp(bridge_mean + z_value * bridge_std),
                "log_price_mean": bridge_mean,
                "log_price_std": bridge_std,
                "local_return_variance": local_variance,
            },
            index=gap_index,
        )

    def log_return_pair_summary(self) -> pd.DataFrame:
        log_return_correlation = self.correlation_matrix(self.log_returns)
        pairs = []
        columns = list(log_return_correlation.columns)

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                pairs.append((columns[i], columns[j], log_return_correlation.iloc[i, j]))

        pairs = sorted(pairs, key=lambda x: x[2])

        return pd.DataFrame(
            [pairs[0], pairs[-1]],
            columns=["asset_1", "asset_2", "log_return_correlation"],
            index=["least_correlated_pair", "most_correlated_pair"],
        )

    def correlation_groups(self, data: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        correlation = self.correlation_matrix(data).fillna(0.0)
        remaining = set(correlation.columns)
        group_rows: list[dict[str, object]] = []
        group_id = 1

        while remaining:
            seed = sorted(remaining)[0]
            stack = [seed]
            members: list[str] = []

            while stack:
                asset = stack.pop()
                if asset not in remaining:
                    continue

                remaining.remove(asset)
                members.append(asset)

                neighbors = [
                    other
                    for other in correlation.columns
                    if other in remaining and correlation.loc[asset, other] >= threshold
                ]
                stack.extend(sorted(neighbors, reverse=True))

            members = sorted(members)
            if len(members) > 1:
                within_group = correlation.loc[members, members]
                mask = np.triu(np.ones(within_group.shape, dtype=bool), k=1)
                avg_within_correlation = within_group.where(mask).stack().mean()
            else:
                avg_within_correlation = np.nan

            for asset in members:
                group_rows.append(
                    {
                        "asset": asset,
                        "group": group_id,
                        "group_members": ", ".join(members),
                        "group_size": len(members),
                        "avg_within_group_correlation": avg_within_correlation,
                    }
                )

            group_id += 1

        return pd.DataFrame(group_rows).set_index("asset").sort_values(["group", "asset"])

    def series_comparison(self, window: int = 30, max_lag: int = 30) -> pd.DataFrame:
        raw_price_acf_summary = self._acf_summary(self.prices, max_lag=max_lag)
        log_return_acf_summary = self._acf_summary(self.log_returns, max_lag=max_lag)
        abs_log_return_acf_summary = self._acf_summary(self.absolute_log_returns, max_lag=max_lag)
        squared_log_return_acf_summary = self._acf_summary(self.squared_log_returns, max_lag=max_lag)
        ljung_box_log_returns = self.ljung_box_summary(self.log_returns)
        ljung_box_abs_log_returns = self.ljung_box_summary(self.absolute_log_returns)
        ljung_box_squared_log_returns = self.ljung_box_summary(self.squared_log_returns)
        single_series_summary = self._single_series_summary(window=window)

        return pd.concat(
            [
                raw_price_acf_summary.add_prefix("raw_price_"),
                log_return_acf_summary.add_prefix("log_return_"),
                abs_log_return_acf_summary.add_prefix("abs_log_return_"),
                squared_log_return_acf_summary.add_prefix("sq_log_return_"),
                ljung_box_log_returns[["lb_pvalue_lag_10", "lb_pvalue_lag_20"]].rename(
                    columns={
                        "lb_pvalue_lag_10": "lb_log_returns_pvalue_lag_10",
                        "lb_pvalue_lag_20": "lb_log_returns_pvalue_lag_20",
                    }
                ),
                ljung_box_abs_log_returns[["lb_pvalue_lag_10", "lb_pvalue_lag_20"]].rename(
                    columns={
                        "lb_pvalue_lag_10": "lb_abs_returns_pvalue_lag_10",
                        "lb_pvalue_lag_20": "lb_abs_returns_pvalue_lag_20",
                    }
                ),
                ljung_box_squared_log_returns[["lb_pvalue_lag_10", "lb_pvalue_lag_20"]].rename(
                    columns={
                        "lb_pvalue_lag_10": "lb_sq_returns_pvalue_lag_10",
                        "lb_pvalue_lag_20": "lb_sq_returns_pvalue_lag_20",
                    }
                ),
                single_series_summary[["mean", "variance", "avg_30d_variance"]].rename(
                    columns={
                        "mean": "return_mean",
                        "variance": "return_variance",
                        "avg_30d_variance": "avg_30d_return_variance",
                    }
                ),
            ],
            axis=1,
        ).sort_values(
            [
                "abs_log_return_significant_lags_up_to_30",
                "sq_log_return_significant_lags_up_to_30",
                "raw_price_significant_lags_up_to_30",
            ],
            ascending=False,
        )
