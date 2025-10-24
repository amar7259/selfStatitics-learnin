#!/usr/bin/env python3
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

claims = pd.read_csv(os.path.join(DATA_DIR, "claims.csv"))
revenue = pd.read_csv(os.path.join(DATA_DIR, "revenue_monthly.csv"), parse_dates=["Month"])

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    return path

# 1) Descriptive statistics
desc = claims["ClaimAmount"].describe()
iqr = claims["ClaimAmount"].quantile(0.75) - claims["ClaimAmount"].quantile(0.25)
skew = claims["ClaimAmount"].skew()
kurt = claims["ClaimAmount"].kurtosis()
summary = pd.Series({
    "count": desc["count"], "mean": desc["mean"], "std": desc["std"],
    "min": desc["min"], "25% (Q1)": claims["ClaimAmount"].quantile(0.25),
    "50% (median)": desc["50%"], "75% (Q3)": claims["ClaimAmount"].quantile(0.75),
    "max": desc["max"], "IQR": iqr, "skewness": skew, "kurtosis": kurt
}).round(2)
summary.to_csv(os.path.join(OUT_DIR, "descriptive_stats_claims.csv"))

# 2) Frequency distribution
bins = np.arange(0, claims["ClaimAmount"].max() + 401, 400)
labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins)-1)]
cut = pd.cut(claims["ClaimAmount"], bins=bins, labels=labels, right=True, include_lowest=True)
freq = cut.value_counts().sort_index()
freq_df = pd.DataFrame({"Range": freq.index, "Frequency": freq.values})
freq_df["Relative %"] = (freq_df["Frequency"]/freq_df["Frequency"].sum()*100).round(2)
freq_df["Cumulative %"] = freq_df["Relative %"].cumsum().round(2)
freq_df.to_csv(os.path.join(OUT_DIR, "frequency_distribution_claims.csv"), index=False)

# 3) Histogram
plt.figure(figsize=(7,4))
plt.hist(claims["ClaimAmount"], bins=bins, edgecolor="black")
plt.title("Histogram of Claim Amounts"); plt.xlabel("Claim Amount"); plt.ylabel("Frequency")
savefig("hist_claim_amounts.png")

# 4) Box plots
plt.figure(figsize=(6,4))
plt.boxplot(claims["ClaimAmount"].dropna(), vert=False, showfliers=True)
plt.title("Box Plot: Claim Amounts (All)"); plt.xlabel("Claim Amount")
savefig("box_claim_all.png")

dept_names = sorted(claims["Department"].unique())
dept_groups = [claims.loc[claims["Department"]==d, "ClaimAmount"].dropna() for d in dept_names]
plt.figure(figsize=(8,4))
plt.boxplot(dept_groups, showfliers=True, labels=dept_names)
plt.title("Box Plot: Claim Amounts by Department"); plt.ylabel("Department"); plt.xlabel("Claim Amount")
savefig("box_claim_by_dept.png")

# 5) Correlation & covariance
numeric = claims[["Age","ClaimAmount","IsSmoker","Denied"]].copy()
corr = numeric.corr(); cov = numeric.cov()
corr.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
cov.to_csv(os.path.join(OUT_DIR, "covariance_matrix.csv"))

# Scatterplot
plt.figure(figsize=(6,4))
plt.scatter(claims["Age"], claims["ClaimAmount"], s=12, alpha=0.7)
plt.title("Scatterplot: Age vs Claim Amount"); plt.xlabel("Age"); plt.ylabel("Claim Amount")
savefig("scatter_age_claim.png")

# 6) Hypothesis tests
# a) t-test smokers vs non-smokers
sm = claims.loc[claims["IsSmoker"]==1, "ClaimAmount"]
ns = claims.loc[claims["IsSmoker"]==0, "ClaimAmount"]
t_stat, p_val = stats.ttest_ind(sm, ns, equal_var=False)
with open(os.path.join(OUT_DIR, "t_test_smoker_vs_nonsmoker.txt"), "w") as f:
    f.write(f"Welch t-test: t={t_stat:.4f}, p={p_val:.6f}\n")

# b) ANOVA by department
groups = [claims.loc[claims["Department"]==d, "ClaimAmount"] for d in dept_names]
f_stat, p_anova = stats.f_oneway(*groups)
with open(os.path.join(OUT_DIR, "anova_by_department.txt"), "w") as f:
    f.write(f"ANOVA: F={f_stat:.4f}, p={p_anova:.6f}\n")

# c) Chi-square: Denied vs IsSmoker
cont = pd.crosstab(claims["Denied"], claims["IsSmoker"])
chi2, p_chi, dof, exp = stats.chi2_contingency(cont)
with open(os.path.join(OUT_DIR, "chi_square_denied_vs_smoker.txt"), "w") as f:
    f.write(f"Chi-square: chi2={chi2:.4f}, p={p_chi:.6f}, dof={dof}\n")
    f.write("Observed:\n" + str(cont) + "\n\nExpected:\n" + str(pd.DataFrame(exp, index=cont.index, columns=cont.columns)) + "\n")

# 7) Expected value demo
cats = pd.DataFrame({"Type":["Routine","Specialist","Emergency"],"Prob":[0.62,0.28,0.10],"AvgCost":[1000,3000,10000]})
cats["Contribution"] = cats["Prob"]*cats["AvgCost"]
ev = cats["Contribution"].sum()
cats.to_csv(os.path.join(OUT_DIR, "expected_value_table.csv"), index=False)
with open(os.path.join(OUT_DIR, "expected_value_result.txt"), "w") as f:
    f.write(f"Expected Claim Cost = {ev:.2f}\n")

# 8) Empirical probability
p_gt_2500 = (claims["ClaimAmount"] > 2500).mean()
with open(os.path.join(OUT_DIR, "probability_demo.txt"), "w") as f:
    f.write(f"P(ClaimAmount > 2500) ≈ {p_gt_2500:.4f}\n")

# 9) Revenue histogram + KPIs
plt.figure(figsize=(7,4))
plt.hist(revenue["Revenue"], bins=12, edgecolor="black")
plt.title("Histogram of Monthly Revenue"); plt.xlabel("Revenue"); plt.ylabel("Frequency")
savefig("hist_revenue.png")

revenue.describe().round(2).to_csv(os.path.join(OUT_DIR, "revenue_descriptive_stats.csv"))

print("Analysis complete. See figures/ and outputs/. ")
