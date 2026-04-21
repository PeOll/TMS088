# Data Analysis

This markdown provides a analysis of seven financial time series from the Spiff market. The objective is to understand the statistical properties, relationships, and structural characteristics of the data in order to inform subsequent interpolation and forecasting.

## Price Level Analysis

### Trends

The price series show heterogenous behaviour:
- **Strong upward trends:** *slingshots, guitars, water, gurkor*
- **Moderate trend:** *tranquility*
- **Declining assets:** *stocks, sugar*

This suggests that the dataset includes both growth assets and declining assets, indicating different underlying processes.

### Structural Features

- Some series (e.g., *sugar*) exhibit potential regime changes, consistent with earlier bimodal distributions.
- Visual inspection suggests periods of increased variabilty, hinting at time-varying volatility.

### Scale Differences

The series differ significantly in magnitude, making direct comparisons difficult. This was addressed by normalising all series to ac ommon starting value.

### Normalised Price Analysis

After normalisation (start = 100):
- **Strong co-movement:** *gurkor and water* (almost identical trajectories)
- **Cluster behaviour:** *slingshots joins the above group*
- **Partial alignment with higher volatility:** *guitars*
- **Independent behaviour:** *stocks and sugar*
- **Hybrid behaviour:** *tranquility*

This confirms the presence of multiple underlying market factors rather than a single integrated market.

## Relationships Between Series

### Correlation Analysis (Returns)

The correlation matrix reveals:
- Strong positive relationships:
    - *gurkor ↔ water* (~0.57)
    - *guitars ↔ slingshots* (~0.53)
- Weak or near-zero correlations for *stocks*
- Negative correlations between clusters

Interpretation:
- Assets can be grouped into clusters driven by common factors
- *stocks* acts as a diversifier / hedge asset

### Rolling Correlation

Rolling correlation analysis indicates that:
- Relationships between assets are not constant over time
- Even strongly correlated assets exhibit time-varying dependence

This suggests that static correlation assumptions are insufficient, especially for modelling and risk management.

## Grouping and Clustering

Using both visual inspection and correlation structure:

**Cluster 1: Commodity-like / Growth Cluster**
- *gurkor, water, slingshots*
- High correlation and similar growth dynamics

**Cluster 2: Volatile Hybrid Assets**
- *guitars, tranquility*
- Partial co-movement with additional idiosyncratic volatility

**Cluster 3: Independent / Anti-correlated Assets**
- *stocks, sugar*
- Weak or negative correlation with other assets

This indicates multiple latent factors and supports the use of multivariate modelling approaches.

## Return Distribution Analysis

### Summary Statistics
- Mean returns ≈ 0 for all assets
- Significant variation in volatility across assets
- Negative skewness observed in several series
- Excess kurtosis (fat tails), particularly:
    - *sugar* (very high kurtosis)
    - *slingshots, guitars*

Interpretation:
- Returns are not normally distributed
- Extreme events occur more frequently than predicted by Gaussian models

### Histogram & QQ-Plot Analysis
- Most series show heavy tails
- Deviations from normality are especially visible in the tails
- *stocks* appears closest to normal

This confirms that:
- Normal distribution assumptions are insufficient
- Models must account for tail risk

## Time Series Properties

### Stationarity (ADF Test)
- Prices: Non-stationary (fail to reject unit root)
- Returns: Stationary (strong rejection of unit root)

Interpretation:
- Prices follow a random walk
- Returns are suitable for statistical modeling

### Autocorrelation (ACF/PACF)
- Returns show no significant autocorrelation
- Suggests:
    - Limited predictability in mean
    - Behavior close to white noise

### Volatility Clustering
- Squared returns show significant autocorrelation
- Strongest in:
    - *slingshots, guitars, sugar*

Interpretation:
- Presence of conditional heteroskedasticity
- Volatility is time-varying and persistent

## Implications for Task 2 (Interpolation)

The analysis provides important guidance:

- Correlated assets:
    
    → Use multivariate methods (e.g., regression, PCA)

- Volatility clustering:
    
    → Avoid simple linear interpolation
    
    → Consider volatility-aware models

- Stationary returns:
    
    → Interpolate in return space rather than price space

- Structural differences across assets:
    
    → Use different models for different series

## Implications for Task 3 (Extrapolation)

- Random walk behavior of prices:

    → Suggests ARIMA-type models

- Volatility clustering:

    → GARCH-type models are appropriate

- Fat tails:

    → Gaussian assumptions may underestimate risk

- Weak autocorrelation in returns:

    → Mean prediction is difficult → focus on volatility modeling