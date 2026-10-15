## This repository contains the Python code and processed data supporting the manuscript:

# Comparability of radioligand and fluorescence-based bioassays for monoamine transporter inhibition profiling: evaluation in stimulant-type new psychoactive substances
(submitted manuscript)

## Repository contents
File	                Description
analysis_plots.py	  Main analysis script: statistics, figures, and data export
merged_summary.csv	Processed dataset: IC50 values and 95% CIs for all compounds at DAT, NET, and SERT

## Data description
merged_summary.csv contains IC50 values (in M) and 95% confidence intervals for 11 stimulant-type new psychoactive substances and reference compounds, assessed at the dopamine (DAT), norepinephrine (NET), and serotonin (SERT) reuptake transporters using two bioassays:
-	Radioligand uptake inhibition assay  (e.g.10.1016/j.euroneuro.2014.12.012, 10.1016/j.neuropharm.2017.07.026, 10.1016/j.neuropharm.2017.08.020)
-	Fluorescence-based uptake inhibition assay (e.g. 10.1002/dta.3547, 10.1111/bph.70587)

## Analysis script
analysis_plots.py produces the following outputs:
-	figure1_combined.png — Figure 1: scatter plots (top row) and Bland-Altman plots (bottom row) per transporter
-	scatter_plots_supplementary.png — Figure S1: scatter plots with compound labels (per-transporter axes)
-	ratio_plots.png — Figure 2: DAT/SERT selectivity ratio scatter plot and Bland-Altman analysis
-	statistics_output.csv — descriptive statistics, correlations, and hypothesis tests per transporter
-	statistics_ratio.csv — correlation and Bland-Altman statistics for DAT/SERT selectivity ratio
-	ratio_table.csv — formatted DAT/SERT ratios with 95% CIs per compound

## Requirements
See requirements.txt for package versions. 

## License
The processed data (merged_summary.csv) is derived from published sources — please cite the original publications as listed in the manuscript.

## Contact
For questions regarding the data or analysis, please contact manuela-monti (corresponding author).
