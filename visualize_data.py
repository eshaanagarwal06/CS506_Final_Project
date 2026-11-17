# Exploratory charts for the phonetic features:
# 1) Correlation of each phoneme's share with WER
# 2) Correlation of every other feature with WER
# 3) Distribution of WER across clips

import matplotlib
matplotlib.use("Agg")  # write files only; works on any OS without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = "dev-other_base.en_features.csv"
ID_COLUMNS = ["wer", "audio_id", "speaker_id"]

print("Loading csv file...")
data = pd.read_csv(DATA_FILE)
wer_col = data["wer"]

# A Pearson correlation is significant at p < 0.05 when |r| > ~1.96 / sqrt(n)
significance = 1.96 / np.sqrt(len(data))


def correlation_chart(corr, title, ylabel, output_file):
    fig, ax = plt.subplots(figsize=(10, max(6, len(corr) * 0.28)))

    colors = ["#c0504d" if value > 0 else "#4f81bd" for value in corr.values]
    ax.barh(corr.index, corr.values, color=colors)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvspan(-significance, significance, color="gray", alpha=0.15,
               label=f"Not significant (|r| < {significance:.3f}, p ≥ 0.05)")

    ax.set_title(title)
    ax.set_xlabel("Pearson correlation with WER (red = more errors, blue = fewer)")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.01)
    ax.legend(loc="lower right")

    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_file}")


# 1. Phoneme correlations
phon_cols = data[[col for col in data.columns if col.startswith("ph_")]]
phon_corr = phon_cols.corrwith(wer_col).sort_values()
phon_corr.index = phon_corr.index.str.replace("ph_", "", regex=False)

correlation_chart(
    phon_corr,
    f"Correlation of phoneme share with WER ({len(data):,} clips)",
    "Phoneme (ARPAbet)",
    "phoneme_corr.svg",
)

# 2. All other feature correlations
feat_cols = data[[
    col for col in data.columns
    if col not in ID_COLUMNS and not col.startswith("ph_")
]]
feat_corr = feat_cols.corrwith(wer_col).sort_values()

correlation_chart(
    feat_corr,
    f"Correlation of non-phoneme features with WER ({len(data):,} clips)",
    "Feature",
    "feature_corr.svg",
)

# 3. WER distribution
fig, ax = plt.subplots(figsize=(9, 5))

bins = np.arange(0, wer_col.max() + 0.05, 0.05)
ax.hist(wer_col, bins=bins, color="#4f81bd", edgecolor="white")

ax.axvline(wer_col.mean(), color="#c0504d", linestyle="--",
           label=f"Mean WER = {wer_col.mean():.3f}")
ax.axvline(wer_col.median(), color="black", linestyle=":",
           label=f"Median WER = {wer_col.median():.3f}")

ax.set_title(
    f"Distribution of WER per clip ({(wer_col == 0).mean():.0%} of clips "
    f"have no errors; {(wer_col > 1).sum()} have WER > 1)"
)
ax.set_xlabel("Word error rate (WER)")
ax.set_ylabel("Number of clips")
ax.legend()

fig.savefig("wer_distribution.svg", bbox_inches="tight")
plt.close(fig)
print("Saved wer_distribution.svg")
