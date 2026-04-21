from __future__ import annotations

import os
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, pacf


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


def adf_summary(data: pd.DataFrame, alpha: float = 0.05, autolag: str = "AIC") -> pd.DataFrame:
    """Run the Augmented Dickey-Fuller test column by column."""
    rows = []

    for column in data.columns:
        series = data[column].dropna()
        adf_stat, adf_pvalue, used_lag, nobs, _, _ = adfuller(series, autolag=autolag)
        conclusion = "Stationary" if adf_pvalue < alpha else "Non-stationary"

        rows.append(
            {
                "series": column,
                "adf_statistic": adf_stat,
                "adf_pvalue": adf_pvalue,
                "used_lag": used_lag,
                "nobs": nobs,
                "conclusion": conclusion,
            }
        )

    return pd.DataFrame(rows).round({"adf_statistic": 4, "adf_pvalue": 6})


def kpss_summary(
    data: pd.DataFrame,
    alpha: float = 0.05,
    regression: str = "c",
    nlags: str | int = "auto",
) -> pd.DataFrame:
    """Run the KPSS stationarity test column by column."""
    rows = []

    for column in data.columns:
        series = data[column].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_stat, kpss_pvalue, used_lag, _ = kpss(series, regression=regression, nlags=nlags)
        conclusion = "Stationary" if kpss_pvalue >= alpha else "Non-stationary"

        rows.append(
            {
                "series": column,
                "kpss_statistic": kpss_stat,
                "kpss_pvalue": kpss_pvalue,
                "used_lag": used_lag,
                "conclusion": conclusion,
            }
        )

    return pd.DataFrame(rows).round({"kpss_statistic": 4, "kpss_pvalue": 6})


def stationarity_summary(
    data: pd.DataFrame,
    alpha: float = 0.05,
    adf_autolag: str = "AIC",
    kpss_regression: str = "c",
    kpss_nlags: str | int = "auto",
) -> pd.DataFrame:
    """Compare ADF and KPSS conclusions for each column."""
    adf = adf_summary(data, alpha=alpha, autolag=adf_autolag).rename(
        columns={
            "used_lag": "adf_used_lag",
            "nobs": "adf_nobs",
            "conclusion": "adf_conclusion",
        }
    )
    kpss_result = kpss_summary(
        data,
        alpha=alpha,
        regression=kpss_regression,
        nlags=kpss_nlags,
    ).rename(
        columns={
            "used_lag": "kpss_used_lag",
            "conclusion": "kpss_conclusion",
        }
    )

    summary = adf.merge(kpss_result, on="series")
    summary["test_agreement"] = np.where(
        summary["adf_conclusion"] == summary["kpss_conclusion"],
        "Agree",
        "Conflict",
    )
    summary["stationarity_conclusion"] = np.where(
        summary["test_agreement"] == "Agree",
        summary["adf_conclusion"],
        "Mixed evidence",
    )
    return summary


def ljung_box_conclusion_table(ljung_box_table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Convert Ljung-Box p-values into compact test conclusions."""
    transform_names = {
        "log_return": "returns",
        "abs_log_return": "absolute_returns",
        "sq_log_return": "squared_returns",
    }
    transform_prefixes = [
        prefix
        for prefix in transform_names
        if any(column.startswith(f"{prefix}_lb_pvalue_lag_") for column in ljung_box_table.columns)
    ]

    def pvalue_columns(prefix: str) -> list[tuple[int, str]]:
        columns = []
        marker = f"{prefix}_lb_pvalue_lag_"
        for column in ljung_box_table.columns:
            if column.startswith(marker):
                columns.append((int(column.removeprefix(marker)), column))
        return sorted(columns)

    def lag_conclusion(row: pd.Series, lag_columns: list[tuple[int, str]]) -> str:
        rejected_lags = [lag for lag, column in lag_columns if row[column] < alpha]
        if not rejected_lags:
            return "No rejection"
        return "Reject at lags " + ", ".join(str(lag) for lag in rejected_lags)

    rows = []
    for series, row in ljung_box_table.iterrows():
        output_row = {"series": series}
        rejected_by_transform: dict[str, bool] = {}

        for prefix in transform_prefixes:
            lag_columns = pvalue_columns(prefix)
            output_row[transform_names[prefix]] = lag_conclusion(row, lag_columns)
            rejected_by_transform[prefix] = any(row[column] < alpha for _, column in lag_columns)

        linear_dependence = rejected_by_transform.get("log_return", False)
        volatility_clustering = rejected_by_transform.get("abs_log_return", False) or rejected_by_transform.get(
            "sq_log_return",
            False,
        )

        if linear_dependence and volatility_clustering:
            overall = "Linear dependence + volatility clustering"
        elif linear_dependence:
            overall = "Linear dependence only"
        elif volatility_clustering:
            overall = "Volatility clustering only"
        else:
            overall = "No clear serial dependence"

        output_row["overall_conclusion"] = overall
        rows.append(output_row)

    return pd.DataFrame(rows)


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
        """Rolling mean of one-period log-returns, not raw prices."""
        return self.log_returns.rolling(window).mean()

    def rolling_variance(self, window: int = 30) -> pd.DataFrame:
        """Rolling variance of one-period log-returns, not raw prices."""
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
                f"avg_{window}d_mean": rolling_mean.mean(),
                f"avg_{window}d_variance": rolling_var.mean(),
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
                    f"significant_lags_up_to_{max_lag}": len(significant_lags),
                }
            )

        significant_lag_column = f"significant_lags_up_to_{max_lag}"
        return pd.DataFrame(summary).set_index("series").sort_values(significant_lag_column, ascending=False)

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
                single_series_summary[["mean", "variance", f"avg_{window}d_variance"]].rename(
                    columns={
                        "mean": "return_mean",
                        "variance": "return_variance",
                        f"avg_{window}d_variance": f"avg_{window}d_return_variance",
                    }
                ),
            ],
            axis=1,
        ).sort_values(
            [
                f"abs_log_return_significant_lags_up_to_{max_lag}",
                f"sq_log_return_significant_lags_up_to_{max_lag}",
                f"raw_price_significant_lags_up_to_{max_lag}",
            ],
            ascending=False,
        )


def fill_missing_with_series_mean(data: pd.DataFrame) -> pd.DataFrame:
    """Keep assets with return history and fill internal missing returns by asset mean."""
    return data.dropna(axis=1, how="all").apply(lambda col: col.fillna(col.mean()), axis=0)


def build_return_feature_table(
    returns: pd.DataFrame,
    absolute_returns: pd.DataFrame,
    squared_returns: pd.DataFrame,
    analyzer: PriceAnalysis,
    acf_lag: int = 1,
) -> pd.DataFrame:
    """Create one row per asset with distribution and volatility-clustering features."""
    assets = returns.columns
    feature_table = pd.DataFrame(index=assets)
    feature_table["mean"] = returns.mean()
    feature_table["volatility"] = returns.std()
    feature_table["skewness"] = returns.skew()
    feature_table["excess_kurtosis"] = returns.kurtosis()
    feature_table["q05"] = returns.quantile(0.05)
    feature_table["q95"] = returns.quantile(0.95)
    feature_table[f"abs_return_acf_lag_{acf_lag}"] = [
        analyzer.sample_acf(absolute_returns[asset], max_lag=acf_lag)[acf_lag] for asset in assets
    ]
    feature_table[f"sq_return_acf_lag_{acf_lag}"] = [
        analyzer.sample_acf(squared_returns[asset], max_lag=acf_lag)[acf_lag] for asset in assets
    ]
    return feature_table


def remove_self_correlations(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """Remove the non-informative diagonal from an asset correlation-profile matrix."""
    profile = correlation_matrix.copy().fillna(0.0)
    for asset in profile.index.intersection(profile.columns):
        profile.loc[asset, asset] = 0.0
    return profile


def _numeric_model_values(
    feature_frame: pd.DataFrame,
    scale_columns: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler | None]:
    numeric_frame = feature_frame.select_dtypes(include=[np.number]).copy()
    if numeric_frame.isna().any().any():
        missing = numeric_frame.columns[numeric_frame.isna().any()].tolist()
        raise ValueError(f"Missing values remain in clustering features: {missing}")

    if scale_columns:
        scaler = StandardScaler()
        model_values = scaler.fit_transform(numeric_frame)
    else:
        scaler = None
        model_values = numeric_frame.to_numpy()

    return numeric_frame, model_values, scaler


def fit_kmeans_to_frame(
    feature_frame: pd.DataFrame,
    n_clusters: int,
    scale_columns: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler | None, KMeans]:
    """Fit KMeans to a numeric asset-feature table and return the model input used."""
    numeric_frame, model_values, scaler = _numeric_model_values(feature_frame, scale_columns=scale_columns)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50).fit(model_values)
    return numeric_frame, model_values, model.labels_, scaler, model


def kmeans_silhouette_table(model_values: np.ndarray, max_k: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compute silhouette scores for feasible KMeans cluster counts."""
    max_valid_k = min(max_k, model_values.shape[0] - 1)
    rows = []
    for n_clusters in range(2, max_valid_k + 1):
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50).fit_predict(model_values)
        rows.append(
            {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_score(model_values, labels),
            }
        )
    return pd.DataFrame(rows)


def cluster_groups(assignments: pd.DataFrame, cluster_col: str) -> dict[int, list[str]]:
    """Return assets grouped by cluster label."""
    return (
        assignments.assign(asset=assignments.index)
        .groupby(cluster_col)["asset"]
        .apply(list)
        .to_dict()
    )


def run_asset_clustering(
    feature_frame: pd.DataFrame,
    n_clusters: int,
    cluster_col: str,
    scale_columns: bool = True,
    max_k: int = 5,
    random_state: int = 42,
) -> dict[str, object]:
    """Standard clustering workflow reused by all asset-clustering views."""
    numeric_frame, model_values, labels, scaler, model = fit_kmeans_to_frame(
        feature_frame,
        n_clusters=n_clusters,
        scale_columns=scale_columns,
        random_state=random_state,
    )
    assignments = pd.DataFrame({cluster_col: labels}, index=numeric_frame.index)
    scores = kmeans_silhouette_table(model_values, max_k=max_k, random_state=random_state)
    return {
        "feature_frame": numeric_frame,
        "model_values": model_values,
        "assignments": assignments,
        "cluster_col": cluster_col,
        "groups": cluster_groups(assignments, cluster_col),
        "silhouette_scores": scores,
        "scaler": scaler,
        "model": model,
    }


def print_cluster_groups(groups: dict[int, list[str]], title: str) -> None:
    print(title)
    for cluster_id, assets in sorted(groups.items()):
        print(f"Cluster {cluster_id}: {assets}")


def plot_cluster_scatter(
    embedding: np.ndarray,
    explained_variance_ratio: np.ndarray | None,
    assignments: pd.DataFrame,
    cluster_col: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot a two-dimensional embedding colored by cluster label."""
    fig, ax = plt.subplots()
    labels = assignments[cluster_col].to_numpy()
    x_range = np.ptp(embedding[:, 0]) or 1.0
    y_range = np.ptp(embedding[:, 1]) or 1.0

    for cluster in sorted(assignments[cluster_col].unique()):
        idx = labels == cluster
        ax.scatter(embedding[idx, 0], embedding[idx, 1], label=f"Cluster {cluster}", s=80)

    for i, asset in enumerate(assignments.index):
        ax.text(
            embedding[i, 0] + 0.025 * x_range,
            embedding[i, 1] + 0.015 * y_range,
            asset,
            fontsize=9,
        )

    ax.set_title(title)
    if explained_variance_ratio is None:
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
    else:
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_pca_projection(
    clustering_result: dict[str, object],
    title: str,
    output_path: Path,
) -> tuple[np.ndarray, PCA]:
    """Project the clustering model input to two PCs and plot the assigned clusters."""
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(clustering_result["model_values"])
    plot_cluster_scatter(
        embedding,
        pca.explained_variance_ratio_,
        clustering_result["assignments"],
        clustering_result["cluster_col"],
        title,
        output_path,
    )
    return embedding, pca


def standardized_return_path_pca(
    returns: pd.DataFrame,
    n_components: int = 6,
    component_prefix: str = "path_pc",
) -> tuple[pd.DataFrame, PCA, np.ndarray]:
    """Standardize each asset return path and project paths to PCA scores."""
    path_values = returns.T.to_numpy()
    path_std = path_values.std(axis=1, keepdims=True)
    path_std[path_std == 0] = 1.0
    scaled_paths = (path_values - path_values.mean(axis=1, keepdims=True)) / path_std

    n_components = min(n_components, scaled_paths.shape[0], scaled_paths.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled_paths)
    score_frame = pd.DataFrame(
        scores,
        index=returns.columns,
        columns=[f"{component_prefix}_{i + 1}" for i in range(scores.shape[1])],
    )
    return score_frame, pca, scaled_paths


def plot_cumulative_explained_variance(pca: PCA, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_.cumsum(),
        marker="o",
    )
    ax.set_title(title)
    ax.set_xlabel("number of components")
    ax.set_ylabel("cumulative explained variance")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def rolling_pairwise_correlations(
    returns: pd.DataFrame,
    window: int = 126,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Rolling pairwise correlations for every asset pair."""
    if min_periods is None:
        min_periods = int(0.8 * window)

    rolling = pd.DataFrame(index=returns.index)
    for asset_1, asset_2 in itertools.combinations(returns.columns, 2):
        rolling[f"{asset_1} vs {asset_2}"] = returns[asset_1].rolling(
            window,
            min_periods=min_periods,
        ).corr(returns[asset_2])

    rolling.attrs["window"] = window
    rolling.attrs["min_periods"] = min_periods
    return rolling


VOLATILITY_REGIME_ORDER = ("low", "medium", "high")


def _regime_spans(regime_series: pd.Series) -> list[dict[str, object]]:
    """Contiguous non-missing regime spans by integer position."""
    spans: list[dict[str, object]] = []
    current_label = None
    start_pos = None
    end_pos = None
    length = 0
    block_id = 0

    def close_span() -> None:
        if current_label is None:
            return
        spans.append(
            {
                "label": current_label,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "length": length,
                "block_id": block_id,
            }
        )

    for position, label in enumerate(regime_series.to_numpy(dtype=object)):
        if pd.isna(label):
            close_span()
            current_label = None
            start_pos = None
            end_pos = None
            length = 0
            block_id += 1
            continue

        if current_label is None or label != current_label:
            close_span()
            current_label = label
            start_pos = position
            length = 1
        else:
            length += 1

        end_pos = position

    close_span()
    return spans


def enforce_minimum_regime_spell_length(
    regime_series: pd.Series,
    min_spell_length: int,
    values: pd.Series | None = None,
    centers: dict[str, float] | None = None,
) -> tuple[pd.Series, int]:
    """Merge short regime spells into neighboring states.

    Short spells are treated as boundary noise rather than independent regimes.
    They are merged into the adjacent spell with the same label when possible,
    otherwise into the longer neighboring spell. If both neighbors have the same
    length, the neighbor whose fitted center is closer to the short spell's
    median value is used.
    """
    if min_spell_length <= 1:
        return regime_series.copy(), 0

    smoothed = regime_series.copy()
    n_merged = 0
    max_iterations = max(1, len(smoothed))

    for _ in range(max_iterations):
        spans = _regime_spans(smoothed)
        short_candidates = [
            (span_index, span)
            for span_index, span in enumerate(spans)
            if span["length"] < min_spell_length
        ]

        if not short_candidates:
            break

        merge_plan = None
        for span_index, span in sorted(short_candidates, key=lambda item: item[1]["length"]):
            neighbors = []
            if span_index > 0 and spans[span_index - 1]["block_id"] == span["block_id"]:
                neighbors.append(spans[span_index - 1])
            if span_index + 1 < len(spans) and spans[span_index + 1]["block_id"] == span["block_id"]:
                neighbors.append(spans[span_index + 1])

            if not neighbors:
                continue

            if len(neighbors) == 2 and neighbors[0]["label"] == neighbors[1]["label"]:
                target_label = neighbors[0]["label"]
            else:
                longest_length = max(neighbor["length"] for neighbor in neighbors)
                longest_neighbors = [neighbor for neighbor in neighbors if neighbor["length"] == longest_length]

                if len(longest_neighbors) == 1 or values is None or centers is None:
                    target_label = longest_neighbors[0]["label"]
                else:
                    spell_values = values.iloc[int(span["start_pos"]) : int(span["end_pos"]) + 1].dropna()
                    if spell_values.empty:
                        target_label = longest_neighbors[0]["label"]
                    else:
                        spell_center = spell_values.median()

                        def center_distance(neighbor: dict[str, object]) -> float:
                            center = centers.get(str(neighbor["label"]), np.nan)
                            if pd.isna(center):
                                return np.inf
                            return abs(spell_center - center)

                        target_label = min(longest_neighbors, key=center_distance)["label"]

            if target_label != span["label"]:
                merge_plan = (span, target_label)
                break

        if merge_plan is None:
            break

        span, target_label = merge_plan
        smoothed.iloc[int(span["start_pos"]) : int(span["end_pos"]) + 1] = target_label
        n_merged += 1

    return smoothed, n_merged


def rolling_realized_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
    annualization: int | None = None,
) -> pd.DataFrame:
    """Rolling standard deviation of log-returns, with gaps left unclassified."""
    if min_periods is None:
        min_periods = int(0.8 * window)

    volatility = returns.rolling(window=window, min_periods=min_periods).std()
    volatility = volatility.where(returns.notna())

    if annualization is not None:
        volatility = volatility * np.sqrt(annualization)

    volatility.attrs["window"] = window
    volatility.attrs["min_periods"] = min_periods
    volatility.attrs["annualization"] = annualization
    return volatility


def classify_volatility_regimes(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
    lower_quantile: float = 1 / 3,
    upper_quantile: float = 2 / 3,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
    method: str = "gmm",
    max_regimes: int = 3,
    min_regime_share: float = 0.03,
    min_spell_length: int = 1,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Classify each asset's rolling volatility into relative volatility regimes.

    The default method fits Gaussian mixtures to log rolling volatility and
    selects the number of states by BIC, up to ``max_regimes``. The older
    quantile method is still available with ``method="quantile"``.

    Labels describe an asset's volatility relative to its own history, not its
    absolute scale relative to the other assets.
    """
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("Expected 0 < lower_quantile < upper_quantile < 1.")
    if len(labels) != 3:
        raise ValueError("Expected exactly three regime labels.")
    if method not in {"gmm", "quantile"}:
        raise ValueError("method must be either 'gmm' or 'quantile'.")
    if not 1 <= max_regimes <= len(labels):
        raise ValueError("max_regimes must be between 1 and the number of labels.")

    rolling_volatility = rolling_realized_volatility(
        returns,
        window=window,
        min_periods=min_periods,
    )
    regimes = pd.DataFrame(index=rolling_volatility.index, columns=rolling_volatility.columns, dtype="object")
    thresholds = []

    for column in rolling_volatility.columns:
        values = rolling_volatility[column].dropna()
        if values.empty:
            thresholds.append(
                {
                    "series": column,
                    "low_medium_threshold": np.nan,
                    "medium_high_threshold": np.nan,
                    "low_center": np.nan,
                    "medium_center": np.nan,
                    "high_center": np.nan,
                    "n_regimes": 0,
                    "n_fitted_regimes": 0,
                    "regime_method": method,
                    "min_spell_length": min_spell_length,
                    "n_short_spells_merged": 0,
                    "n_classified_windows": 0,
                }
            )
            continue

        if method == "quantile":
            low_threshold = values.quantile(lower_quantile)
            high_threshold = values.quantile(upper_quantile)
            observed = rolling_volatility[column].notna()

            regimes.loc[observed & (rolling_volatility[column] <= low_threshold), column] = labels[0]
            regimes.loc[
                observed
                & (rolling_volatility[column] > low_threshold)
                & (rolling_volatility[column] <= high_threshold),
                column,
            ] = labels[1]
            regimes.loc[observed & (rolling_volatility[column] > high_threshold), column] = labels[2]
            centers = {
                labels[0]: values[values <= low_threshold].median(),
                labels[1]: values[(values > low_threshold) & (values <= high_threshold)].median(),
                labels[2]: values[values > high_threshold].median(),
            }
            n_fitted_regimes = 3

        else:
            positive_values = values[values > 0]
            if positive_values.empty:
                regimes.loc[values.index, column] = labels[1]
                low_threshold = np.nan
                high_threshold = np.nan
                centers = {labels[0]: np.nan, labels[1]: np.nan, labels[2]: np.nan}
                n_fitted_regimes = 1
            else:
                volatility_floor = positive_values.min() * 0.5
                log_values = np.log(values.clip(lower=volatility_floor).to_numpy()).reshape(-1, 1)
                max_valid_regimes = min(max_regimes, len(np.unique(log_values)), len(log_values))

                candidates = []
                for n_components in range(1, max_valid_regimes + 1):
                    model = GaussianMixture(
                        n_components=n_components,
                        covariance_type="full",
                        n_init=10,
                        random_state=random_state,
                    ).fit(log_values)
                    component = model.predict(log_values)
                    component_shares = np.bincount(component, minlength=n_components) / len(component)
                    candidates.append(
                        {
                            "model": model,
                            "component": component,
                            "bic": model.bic(log_values),
                            "min_share": component_shares.min(),
                        }
                    )

                valid_candidates = [
                    candidate for candidate in candidates if candidate["min_share"] >= min_regime_share
                ]
                best = min(valid_candidates or candidates, key=lambda candidate: candidate["bic"])
                model = best["model"]
                component = best["component"]
                ordered_components = np.argsort(model.means_.ravel())
                n_fitted_regimes = len(ordered_components)

                if n_fitted_regimes == 1:
                    ordered_labels = [labels[1]]
                elif n_fitted_regimes == 2:
                    ordered_labels = [labels[0], labels[2]]
                else:
                    ordered_labels = list(labels)

                component_to_label = {
                    component_id: ordered_labels[rank]
                    for rank, component_id in enumerate(ordered_components)
                }
                regimes.loc[values.index, column] = [component_to_label[component_id] for component_id in component]

                center_by_label = {
                    component_to_label[component_id]: np.exp(model.means_.ravel()[component_id])
                    for component_id in ordered_components
                }
                centers = {label: center_by_label.get(label, np.nan) for label in labels}

                ordered_log_centers = np.sort(model.means_.ravel())
                if n_fitted_regimes == 1:
                    low_threshold = np.nan
                    high_threshold = np.nan
                elif n_fitted_regimes == 2:
                    low_threshold = np.exp(ordered_log_centers[:2].mean())
                    high_threshold = np.nan
                else:
                    low_threshold = np.exp(ordered_log_centers[:2].mean())
                    high_threshold = np.exp(ordered_log_centers[1:3].mean())

        smoothed, n_short_spells_merged = enforce_minimum_regime_spell_length(
            regimes[column],
            min_spell_length=min_spell_length,
            values=rolling_volatility[column],
            centers=centers,
        )
        regimes[column] = smoothed
        n_regimes = regimes[column].dropna().nunique()

        thresholds.append(
            {
                "series": column,
                "low_medium_threshold": low_threshold,
                "medium_high_threshold": high_threshold,
                "low_center": centers[labels[0]],
                "medium_center": centers[labels[1]],
                "high_center": centers[labels[2]],
                "n_regimes": n_regimes,
                "n_fitted_regimes": n_fitted_regimes,
                "regime_method": method,
                "min_spell_length": min_spell_length,
                "n_short_spells_merged": n_short_spells_merged,
                "n_classified_windows": int(values.notna().sum()),
            }
        )

    thresholds_frame = pd.DataFrame(thresholds).set_index("series")
    regimes.attrs["labels"] = labels
    regimes.attrs["window"] = window
    regimes.attrs["min_periods"] = rolling_volatility.attrs["min_periods"]

    return {
        "rolling_volatility": rolling_volatility,
        "regimes": regimes,
        "thresholds": thresholds_frame,
    }


def volatility_regime_spells(
    regimes: pd.DataFrame,
) -> pd.DataFrame:
    """Return contiguous same-regime episodes for each asset."""
    rows = []

    for column in regimes.columns:
        current_regime = None
        spell_start = None
        spell_end = None
        spell_length = 0

        def close_spell() -> None:
            if current_regime is None:
                return
            rows.append(
                {
                    "series": column,
                    "regime": current_regime,
                    "start": spell_start,
                    "end": spell_end,
                    "length": spell_length,
                }
            )

        for day, regime in regimes[column].items():
            if pd.isna(regime):
                close_spell()
                current_regime = None
                spell_start = None
                spell_end = None
                spell_length = 0
                continue

            if regime != current_regime:
                close_spell()
                current_regime = regime
                spell_start = day
                spell_length = 1
            else:
                spell_length += 1

            spell_end = day

        close_spell()

    if not rows:
        return pd.DataFrame(columns=["series", "regime", "start", "end", "length"])

    return pd.DataFrame(rows)


def summarize_volatility_regimes(
    regimes: pd.DataFrame,
    rolling_volatility: pd.DataFrame,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    """Summarize regime frequency, volatility level, and spell persistence."""
    spells = volatility_regime_spells(regimes)
    rows = []

    for column in regimes.columns:
        classified = regimes[column].notna()
        n_classified = int(classified.sum())

        for label in labels:
            mask = regimes[column] == label
            regime_spells = spells[(spells["series"] == column) & (spells["regime"] == label)]
            rows.append(
                {
                    "series": column,
                    "regime": label,
                    "n_windows": int(mask.sum()),
                    "share_windows": mask.sum() / n_classified if n_classified else np.nan,
                    "mean_rolling_volatility": rolling_volatility.loc[mask, column].mean(),
                    "median_rolling_volatility": rolling_volatility.loc[mask, column].median(),
                    "n_spells": len(regime_spells),
                    "avg_spell_length": regime_spells["length"].mean() if len(regime_spells) else np.nan,
                    "max_spell_length": regime_spells["length"].max() if len(regime_spells) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def volatility_regime_transition_summary(
    regimes: pd.DataFrame,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    """Summarize how persistent each asset's volatility regimes are."""
    rows = []

    for column in regimes.columns:
        current = regimes[column]
        previous = current.shift(1)
        comparable = current.notna() & previous.notna()
        n_comparable = int(comparable.sum())

        row: dict[str, object] = {
            "series": column,
            "n_comparable_steps": n_comparable,
            "transition_rate": np.nan,
            "same_regime_share": np.nan,
        }

        if n_comparable:
            changed = current[comparable] != previous[comparable]
            row["transition_rate"] = changed.mean()
            row["same_regime_share"] = 1 - changed.mean()

        for label in labels:
            label_previous = comparable & (previous == label)
            denominator = int(label_previous.sum())
            row[f"{label}_persistence"] = (
                ((current == label) & label_previous).sum() / denominator if denominator else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def volatility_regime_shares(
    regimes: pd.DataFrame,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    """Share of currently classified assets in each volatility regime."""
    valid_count = regimes.notna().sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({label: regimes.eq(label).sum(axis=1) / valid_count for label in labels})


def _regime_numeric_frame(
    regimes: pd.DataFrame,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    mapping = {label: value for value, label in enumerate(labels)}
    return regimes.apply(lambda column: column.map(mapping)).astype(float)


def _set_sparse_time_ticks(ax, index: pd.Index, n_ticks: int = 8) -> None:
    if len(index) == 0:
        return

    positions = np.linspace(0, len(index) - 1, min(n_ticks, len(index))).astype(int)
    ax.set_xticks(positions + 0.5)
    ax.set_xticklabels([str(index[position]) for position in positions], rotation=0)


def plot_volatility_regime_heatmap(
    regimes: pd.DataFrame,
    output_path: Path,
    title: str,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> None:
    """Plot low/medium/high volatility regimes through time for each asset."""
    from matplotlib.colors import ListedColormap

    numeric = _regime_numeric_frame(regimes, labels=labels)
    cmap = ListedColormap(["#4C78A8", "#BAB0AC", "#E45756"])

    fig, ax = plt.subplots(figsize=(14, 0.75 * len(regimes.columns) + 2))
    heatmap = sns.heatmap(
        numeric.T,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        cbar_kws={"ticks": range(len(labels)), "label": "volatility regime"},
        ax=ax,
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticklabels(labels)
    _set_sparse_time_ticks(ax, regimes.index)
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("series")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_rolling_volatility_regimes(
    rolling_volatility: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    """Plot rolling volatility with the low/medium/high threshold lines."""
    fig, axes = plt.subplots(
        len(rolling_volatility.columns),
        1,
        figsize=(14, 2.1 * len(rolling_volatility.columns)),
        sharex=True,
        squeeze=False,
    )

    for row, column in enumerate(rolling_volatility.columns):
        ax = axes[row, 0]
        ax.plot(rolling_volatility.index, rolling_volatility[column], color="tab:red", linewidth=1)
        if column in thresholds.index:
            low_threshold = thresholds.loc[column, "low_medium_threshold"]
            high_threshold = thresholds.loc[column, "medium_high_threshold"]
            if pd.notna(low_threshold):
                ax.axhline(
                    low_threshold,
                    color="tab:blue",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                )
            if pd.notna(high_threshold):
                ax.axhline(
                    high_threshold,
                    color="tab:purple",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                )
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_series_with_regime_overlay(
    data: pd.DataFrame,
    regimes: pd.DataFrame,
    output_path: Path,
    title: str,
    ylabel: str,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
    line_color: str = "black",
    regime_colors: dict[str, str] | None = None,
    regime_alpha: float = 0.16,
) -> None:
    """Plot each series with volatility-regime background shading."""
    from matplotlib.patches import Patch

    if regime_colors is None:
        regime_colors = {
            labels[0]: "#4C78A8",
            labels[1]: "#BAB0AC",
            labels[2]: "#E45756",
        }

    columns = [column for column in data.columns if column in regimes.columns]
    if not columns:
        raise ValueError("No data columns have matching regime labels.")

    fig, axes = plt.subplots(
        len(columns),
        1,
        figsize=(14, 2.1 * len(columns)),
        sharex=True,
        squeeze=False,
    )

    for row, column in enumerate(columns):
        ax = axes[row, 0]
        y = data[column]
        ax.plot(y.index, y, color=line_color, linewidth=0.9, zorder=2)

        aligned_regime = regimes[column].reindex(y.index)
        for span in _regime_spans(aligned_regime):
            label = str(span["label"])
            if label not in regime_colors:
                continue
            start = aligned_regime.index[int(span["start_pos"])]
            end = aligned_regime.index[int(span["end_pos"])]
            ax.axvspan(
                start,
                end,
                color=regime_colors[label],
                alpha=regime_alpha,
                linewidth=0,
                zorder=0,
            )

        ax.set_ylabel(column)
        ax.grid(True, alpha=0.2, zorder=1)

    handles = [
        Patch(facecolor=regime_colors[label], alpha=regime_alpha, label=f"{label} volatility")
        for label in labels
        if label in regime_colors
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=len(handles))
    axes[0, 0].set_title(title)
    fig.supylabel(ylabel)
    axes[-1, 0].set_xlabel("day")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_volatility_regime_shares(
    regimes: pd.DataFrame,
    output_path: Path,
    title: str,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    """Plot the cross-sectional share of assets in each volatility regime."""
    shares = volatility_regime_shares(regimes, labels=labels)
    colors = ["#4C78A8", "#BAB0AC", "#E45756"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(shares.index, [shares[label] for label in labels], labels=labels, colors=colors, alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("share of classified assets")
    ax.legend(loc="upper left", ncol=len(labels))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    return shares


def system_volatility_regime(
    regimes: pd.DataFrame,
    lower_quantile: float = 1 / 3,
    upper_quantile: float = 2 / 3,
    labels: tuple[str, str, str] = VOLATILITY_REGIME_ORDER,
    method: str = "gmm",
    max_regimes: int = 3,
    min_regime_share: float = 0.03,
    min_spell_length: int = 1,
    random_state: int = 42,
) -> dict[str, pd.Series | pd.DataFrame]:
    """Classify each day by the cross-sectional volatility regime score."""
    if not 0 < lower_quantile < upper_quantile < 1:
        raise ValueError("Expected 0 < lower_quantile < upper_quantile < 1.")
    if method not in {"gmm", "quantile"}:
        raise ValueError("method must be either 'gmm' or 'quantile'.")
    if not 1 <= max_regimes <= len(labels):
        raise ValueError("max_regimes must be between 1 and the number of labels.")

    numeric = _regime_numeric_frame(regimes, labels=labels)
    score = numeric.mean(axis=1)
    valid_score = score.dropna()
    state = pd.Series(index=score.index, dtype="object", name="system_volatility_regime")

    if valid_score.empty:
        thresholds = pd.Series(
            {"low_medium_threshold": np.nan, "medium_high_threshold": np.nan},
            name="system_regime_thresholds",
        )
        return {"score": score.rename("system_volatility_score"), "regime": state, "thresholds": thresholds}

    if method == "quantile":
        low_threshold = valid_score.quantile(lower_quantile)
        high_threshold = valid_score.quantile(upper_quantile)
        state.loc[score <= low_threshold] = labels[0]
        state.loc[(score > low_threshold) & (score <= high_threshold)] = labels[1]
        state.loc[score > high_threshold] = labels[2]
        centers = {
            labels[0]: valid_score[valid_score <= low_threshold].median(),
            labels[1]: valid_score[(valid_score > low_threshold) & (valid_score <= high_threshold)].median(),
            labels[2]: valid_score[valid_score > high_threshold].median(),
        }
        n_fitted_regimes = 3
    else:
        values = valid_score.to_numpy().reshape(-1, 1)
        max_valid_regimes = min(max_regimes, len(np.unique(values)), len(values))
        candidates = []
        for n_components in range(1, max_valid_regimes + 1):
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                n_init=10,
                random_state=random_state,
            ).fit(values)
            component = model.predict(values)
            component_shares = np.bincount(component, minlength=n_components) / len(component)
            candidates.append(
                {
                    "model": model,
                    "component": component,
                    "bic": model.bic(values),
                    "min_share": component_shares.min(),
                }
            )

        valid_candidates = [candidate for candidate in candidates if candidate["min_share"] >= min_regime_share]
        best = min(valid_candidates or candidates, key=lambda candidate: candidate["bic"])
        model = best["model"]
        component = best["component"]
        ordered_components = np.argsort(model.means_.ravel())
        n_fitted_regimes = len(ordered_components)

        if n_fitted_regimes == 1:
            ordered_labels = [labels[1]]
        elif n_fitted_regimes == 2:
            ordered_labels = [labels[0], labels[2]]
        else:
            ordered_labels = list(labels)

        component_to_label = {
            component_id: ordered_labels[rank]
            for rank, component_id in enumerate(ordered_components)
        }
        state.loc[valid_score.index] = [component_to_label[component_id] for component_id in component]

        center_by_label = {
            component_to_label[component_id]: model.means_.ravel()[component_id]
            for component_id in ordered_components
        }
        centers = {label: center_by_label.get(label, np.nan) for label in labels}

        ordered_centers = np.sort(model.means_.ravel())
        if n_fitted_regimes == 1:
            low_threshold = np.nan
            high_threshold = np.nan
        elif n_fitted_regimes == 2:
            low_threshold = ordered_centers[:2].mean()
            high_threshold = np.nan
        else:
            low_threshold = ordered_centers[:2].mean()
            high_threshold = ordered_centers[1:3].mean()

    state, n_short_spells_merged = enforce_minimum_regime_spell_length(
        state,
        min_spell_length=min_spell_length,
        values=score,
        centers=centers,
    )
    n_regimes = state.dropna().nunique()

    thresholds = pd.Series(
        {
            "low_medium_threshold": low_threshold,
            "medium_high_threshold": high_threshold,
            "low_center": centers[labels[0]],
            "medium_center": centers[labels[1]],
            "high_center": centers[labels[2]],
            "n_regimes": n_regimes,
            "n_fitted_regimes": n_fitted_regimes,
            "regime_method": method,
            "min_spell_length": min_spell_length,
            "n_short_spells_merged": n_short_spells_merged,
        },
        name="system_regime_thresholds",
    )
    return {"score": score.rename("system_volatility_score"), "regime": state, "thresholds": thresholds}


def regime_conditional_correlation_matrices(
    returns: pd.DataFrame,
    regime_state: pd.Series,
    labels: tuple[str, ...] = VOLATILITY_REGIME_ORDER,
    min_observations: int = 50,
) -> dict[str, pd.DataFrame]:
    """Estimate return-correlation matrices conditional on one regime state per day."""
    aligned_state = regime_state.reindex(returns.index)
    return {
        label: returns.loc[aligned_state == label].corr(min_periods=min_observations)
        for label in labels
    }


def regime_conditional_correlation_summary(
    returns: pd.DataFrame,
    regime_state: pd.Series,
    labels: tuple[str, ...] = VOLATILITY_REGIME_ORDER,
    min_observations: int = 50,
) -> pd.DataFrame:
    """Pairwise correlations conditional on one regime state per day."""
    aligned_state = regime_state.reindex(returns.index)
    full_correlation = returns.corr()
    rows = []

    for asset_1, asset_2 in itertools.combinations(returns.columns, 2):
        pair_returns = returns[[asset_1, asset_2]]
        observed_pair = pair_returns.notna().all(axis=1)
        full_pair_corr = full_correlation.loc[asset_1, asset_2]

        for label in labels:
            mask = observed_pair & (aligned_state == label)
            n_observations = int(mask.sum())
            correlation = (
                pair_returns.loc[mask, asset_1].corr(pair_returns.loc[mask, asset_2])
                if n_observations >= min_observations
                else np.nan
            )
            rows.append(
                {
                    "pair": f"{asset_1} vs {asset_2}",
                    "asset_1": asset_1,
                    "asset_2": asset_2,
                    "regime": label,
                    "n_observations": n_observations,
                    "correlation": correlation,
                    "full_sample_correlation": full_pair_corr,
                    "delta_from_full_sample": correlation - full_pair_corr if pd.notna(correlation) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def pair_same_regime_correlation_summary(
    returns: pd.DataFrame,
    regimes: pd.DataFrame,
    labels: tuple[str, ...] = VOLATILITY_REGIME_ORDER,
    min_observations: int = 50,
) -> pd.DataFrame:
    """Pairwise correlations when both assets are in the same volatility regime."""
    available_assets = [asset for asset in returns.columns if asset in regimes.columns]
    full_correlation = returns.corr()
    rows = []

    for asset_1, asset_2 in itertools.combinations(available_assets, 2):
        pair_returns = returns[[asset_1, asset_2]]
        observed_pair = pair_returns.notna().all(axis=1)
        full_pair_corr = full_correlation.loc[asset_1, asset_2]

        for label in labels:
            mask = observed_pair & (regimes[asset_1] == label) & (regimes[asset_2] == label)
            n_observations = int(mask.sum())
            correlation = (
                pair_returns.loc[mask, asset_1].corr(pair_returns.loc[mask, asset_2])
                if n_observations >= min_observations
                else np.nan
            )
            rows.append(
                {
                    "pair": f"{asset_1} vs {asset_2}",
                    "asset_1": asset_1,
                    "asset_2": asset_2,
                    "regime": label,
                    "n_observations": n_observations,
                    "correlation": correlation,
                    "full_sample_correlation": full_pair_corr,
                    "delta_from_full_sample": correlation - full_pair_corr if pd.notna(correlation) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def wide_regime_correlation_summary(
    summary: pd.DataFrame,
    labels: tuple[str, ...] = VOLATILITY_REGIME_ORDER,
) -> pd.DataFrame:
    """Convert a long regime-correlation table into one row per asset pair."""
    index_columns = ["pair", "asset_1", "asset_2"]
    correlation_wide = (
        summary.pivot(index=index_columns, columns="regime", values="correlation")
        .reindex(columns=labels)
        .reset_index()
    )
    observation_wide = (
        summary.pivot(index=index_columns, columns="regime", values="n_observations")
        .reindex(columns=labels)
        .add_prefix("n_")
        .reset_index()
    )
    full_sample = summary[index_columns + ["full_sample_correlation"]].drop_duplicates(index_columns)

    output = correlation_wide.merge(full_sample, on=index_columns).merge(observation_wide, on=index_columns)
    if labels[0] in output.columns and labels[-1] in output.columns:
        output[f"{labels[-1]}_minus_{labels[0]}"] = output[labels[-1]] - output[labels[0]]
    return output


def plot_regime_conditional_correlation_matrices(
    matrices: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
    labels: tuple[str, ...] = VOLATILITY_REGIME_ORDER,
    cmap: str = "coolwarm",
) -> None:
    """Plot one correlation heatmap for each volatility regime."""
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 4.8), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0, col]
        matrix = matrices[label]
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            cbar=col == len(labels) - 1,
            ax=ax,
        )
        ax.set_title(f"{label} volatility")

    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_regime_correlation_differences(
    wide_summary: pd.DataFrame,
    output_path: Path,
    title: str,
    low_regime: str = "low",
    high_regime: str = "high",
    top_n: int | None = None,
) -> None:
    """Plot high-minus-low conditional correlation differences by pair."""
    diff_col = f"{high_regime}_minus_{low_regime}"
    if diff_col not in wide_summary.columns:
        raise ValueError(f"Expected a '{diff_col}' column in wide_summary.")

    data = wide_summary.dropna(subset=[diff_col]).sort_values(diff_col)
    if top_n is not None and len(data) > top_n:
        half = max(1, top_n // 2)
        data = pd.concat([data.head(half), data.tail(top_n - half)]).sort_values(diff_col)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(data))))
    colors = np.where(data[diff_col] >= 0, "tab:red", "tab:blue")
    ax.barh(data["pair"], data[diff_col], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"{high_regime} correlation minus {low_regime} correlation")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def _threshold_label(threshold: float) -> str:
    return str(threshold).replace("-", "minus_").replace(".", "_")


def rolling_correlation_stability_summary(
    rolling_correlations: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Summarize the strength and stability of each rolling pairwise correlation."""
    threshold_label = _threshold_label(threshold)
    positive_share_col = f"share_corr_ge_{threshold_label}"
    negative_share_col = f"share_corr_le_minus_{threshold_label}"
    absolute_share_col = f"share_abs_corr_ge_{threshold_label}"
    rows = []

    for pair_name in rolling_correlations.columns:
        series = rolling_correlations[pair_name].dropna()
        asset_1, asset_2 = pair_name.split(" vs ", 1)

        if series.empty:
            rows.append(
                {
                    "pair": pair_name,
                    "asset_1": asset_1,
                    "asset_2": asset_2,
                    "n_windows": 0,
                    "mean": np.nan,
                    "mean_abs_corr": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "start": np.nan,
                    "end": np.nan,
                    "end_minus_start": np.nan,
                    "abs_change": np.nan,
                    "annualized_linear_slope": np.nan,
                    positive_share_col: np.nan,
                    negative_share_col: np.nan,
                    absolute_share_col: np.nan,
                }
            )
            continue

        x = np.arange(len(series))
        slope = np.polyfit(x, series.to_numpy(), 1)[0] * 252 if len(series) > 1 else np.nan
        start = series.iloc[0]
        end = series.iloc[-1]
        rows.append(
            {
                "pair": pair_name,
                "asset_1": asset_1,
                "asset_2": asset_2,
                "n_windows": len(series),
                "mean": series.mean(),
                "mean_abs_corr": series.abs().mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "start": start,
                "end": end,
                "end_minus_start": end - start,
                "abs_change": abs(end - start),
                "annualized_linear_slope": slope,
                positive_share_col: (series >= threshold).mean(),
                negative_share_col: (series <= -threshold).mean(),
                absolute_share_col: (series.abs() >= threshold).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [positive_share_col, absolute_share_col, "mean_abs_corr"],
        ascending=False,
    )


def rolling_correlation_stability_matrix(
    summary: pd.DataFrame,
    value_col: str,
    assets: list[str] | pd.Index | None = None,
) -> pd.DataFrame:
    """Create a symmetric asset-pair matrix from a rolling-correlation summary column."""
    if assets is None:
        assets = pd.Index(pd.unique(summary[["asset_1", "asset_2"]].to_numpy().ravel()))

    matrix = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    for _, row in summary.iterrows():
        matrix.loc[row["asset_1"], row["asset_2"]] = row[value_col]
        matrix.loc[row["asset_2"], row["asset_1"]] = row[value_col]
    return matrix


def plot_pairwise_stability_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    cmap: str = "coolwarm",
    vmin: float = 0,
    vmax: float = 1,
) -> None:
    """Plot a lower-triangle heatmap for pairwise rolling-correlation stability metrics."""
    fig, ax = plt.subplots()
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    sns.heatmap(
        matrix.round(2),
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
