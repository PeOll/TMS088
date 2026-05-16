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


class DataAnalysis:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.copy()

    @classmethod
    def from_csv(cls, file_name: str) -> "DataAnalysis":
        return cls(read_data(file_name))

    def clean(self, placeholder_value: float = 1000) -> "DataAnalysis":
        return DataAnalysis(clean_data(self.prices, placeholder_value=placeholder_value))

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

    def rolling_mean(self, window: int = 30) -> pd.DataFrame:
        return self.log_returns.rolling(window).mean()

    def rolling_variance(self, window: int = 30) -> pd.DataFrame:
        return self.log_returns.rolling(window).var()

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

        return pd.DataFrame(rows)


def fill_missing_with_series_mean(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna(axis=1, how="all").apply(lambda col: col.fillna(col.mean()), axis=0)


def build_return_feature_table(
    returns: pd.DataFrame,
    absolute_returns: pd.DataFrame,
    squared_returns: pd.DataFrame,
    analyzer: DataAnalysis,
    acf_lag: int = 1,
) -> pd.DataFrame:
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
    numeric_frame, model_values, scaler = _numeric_model_values(feature_frame, scale_columns=scale_columns)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=50).fit(model_values)
    return numeric_frame, model_values, model.labels_, scaler, model


def kmeans_silhouette_table(model_values: np.ndarray, max_k: int = 5, random_state: int = 42) -> pd.DataFrame:
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


def plot_cluster_scatter(
    embedding: np.ndarray,
    explained_variance_ratio: np.ndarray | None,
    assignments: pd.DataFrame,
    cluster_col: str,
    title: str,
    output_path: Path,
) -> None:
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


def rolling_pairwise_correlations(
    returns: pd.DataFrame,
    window: int = 126,
    min_periods: int | None = None,
) -> pd.DataFrame:
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


def _threshold_label(threshold: float) -> str:
    return str(threshold).replace("-", "minus_").replace(".", "_")


def rolling_correlation_stability_summary(
    rolling_correlations: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
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


def rolling_realized_volatility(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: int | None = None,
    annualization: int | None = None,
) -> pd.DataFrame:
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


def plot_dual_serial_correlation_grid(data, analysis, max_lag, label, output, acf_color="tab:blue", pacf_color="tab:green"):
    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(len(data.columns), 2, figsize=(15, 2.8 * len(data.columns)), squeeze=False)

    for row, column in enumerate(data.columns):
        series = data[column].dropna()
        acf_values = analysis.sample_acf(series, max_lag=max_lag)
        pacf_values = analysis.sample_pacf(series, max_lag=max_lag)
        confidence_band = 1.96 / np.sqrt(len(series))

        for ax, values, title, color in [
            (axes[row, 0], acf_values, "ACF", acf_color),
            (axes[row, 1], pacf_values, "PACF", pacf_color),
        ]:
            ax.axhline(0, color="black", linewidth=1)
            ax.axhline(confidence_band, color="tab:red", linestyle="--", linewidth=1)
            ax.axhline(-confidence_band, color="tab:red", linestyle="--", linewidth=1)
            ax.stem(lags, values, basefmt=" ", linefmt=color, markerfmt="o")
            ax.set_title(f"{column}: {title} of {label}")
            ax.set_ylabel(title)
            ax.set_xlim(-0.5, max_lag + 0.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("lag")

    plt.tight_layout()
    safe_label = label.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(output / f"acf_pacf_{safe_label}.png", bbox_inches="tight")
    plt.show()
