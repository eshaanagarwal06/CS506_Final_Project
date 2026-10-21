# This file aims to build 2 graphs:
# 1) A bar chart, demonstrating which phonemes appear to be linked to more or less WER
# 2) Something else with more features (later!) (maybe a correlation heatmap or something)

# Importing modules relevant for correlation heatmap
# import matplotlib.pyplot as mp
import pandas as pd
# import seaborn as sb

# Import for barchart
import matplotlib.pyplot as plt
# import numpy as np
import os as os

# load data (currently aimed to load feature data run on large set of audio clips)
print("Loading csv file...")
data = pd.read_csv("dev-other_base.en_features.csv")

print("Generating Bar Chart(s)...")
phon_col = data[[col for col in data.columns if col.startswith("ph_")]]
wer_col = data["wer"]
phon_corr = phon_col.corrwith(wer_col).sort_values()
        
# remove "ph" for graph readability

fig_width = max(12, len(phon_corr) * 0.18)
fig_height = max(8, len(phon_corr) * 0.5)
plt.figure(figsize=(fig_width, fig_height))

bar_x = phon_corr.index.str.replace("ph_", "", regex=False)
bar_y = phon_corr.values

plt.barh(bar_x,bar_y)
plt.title('Phoneme WER Correlation')
plt.ylabel('Phonemes')
plt.xlabel('Word Error Rate (WER)')
# plt.xticks(rotation=90)
plt.show(block=False)
print("Bar Chart Generated (1/2)")
plt.savefig("phoneme_corr.svg")
plt.close()
os.startfile("phoneme_corr.svg")

# Now going to check correlations of all features to WER, and output highest correlating
feat_col = data[[col for col in data.columns if ( (col != "wer") and (col != "audio_id") and (col != "speaker_id") and (not col.startswith("ph_")) )]]
feat_corr = feat_col.corrwith(wer_col).sort_values()

bar_x = feat_corr.index
bar_y = feat_corr.values

fig_width = max(12, len(feat_corr) * 0.18)
fig_height = max(8, len(feat_corr) * 0.5)
plt.figure(figsize=(fig_width, fig_height))

plt.barh(bar_x,bar_y)
plt.title('All Feature WER Correlation')
plt.ylabel('Features')
plt.xlabel('Word Error Rate (WER)')
# plt.xticks(rotation=90)
plt.margins(x=0.0001, y=0.05)
plt.show(block=False)
print("Bar Chart Generated (2/2)")
plt.savefig("feature_corr.svg", bbox_inches="tight")
plt.close()
os.startfile("feature_corr.svg")