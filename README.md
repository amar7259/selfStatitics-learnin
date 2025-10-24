# Stats Foundation Project (Healthcare-style)

Demonstrates core statistics with Python on synthetic data:

- Descriptive statistics (mean, median, variance, std, IQR, skewness, kurtosis)
- Frequency distribution (grouped, relative %, cumulative %)
- Histograms and box plots
- Correlation, covariance, scatterplots
- Hypothesis testing: Welch t-test, ANOVA, Chi-square
- Expected value (weighted average)
- Empirical probability demo
- Extra: Monthly revenue distribution

## Structure
```
stats_foundation_project/
├─ data/
│  ├─ claims.csv
│  └─ revenue_monthly.csv
├─ figures/                # Generated charts
├─ outputs/                # CSV/txt results
├─ src/
│  └─ run_analysis.py
└─ README.md
```

## Run
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pandas numpy scipy matplotlib
python src/run_analysis.py
```
