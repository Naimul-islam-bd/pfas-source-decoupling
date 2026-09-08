# PFAS Source-Decoupling: national-scale analysis of UCMR5 drinking-water data

Analysis code for the study:

**Unregulated PFAS dominate U.S. drinking-water detection and are decoupled from local point sources: national occurrence, population exposure, and screening implications from UCMR5.** Naimul Islam, Independent Researcher, Dhaka, Bangladesh. ORCID: 0009-0002-3442-8980.

## Summary

Using the U.S. EPA Fifth Unregulated Contaminant Monitoring Rule (UCMR5), covering 24,837 sampling locations across 10,091 public water systems and 29 PFAS analytes, this project asks whether drinking-water PFAS detection tracks proximity to local point sources (PFAS-relevant industrial facilities, wastewater treatment plants, airports, military sites) once broad regional structure is accounted for. It then joins independent public datasets to translate that result into a population-exposure figure, a monitoring-efficiency assessment, a comparison against systems' own reported sources, and a national screening layer for systems not yet sampled.

## Main findings

- 24.31% of locations (6,038) had at least one detectable PFAS.
- Four analytes are detected more often than PFOA (8.76%) or PFOS (8.70%): PFPeA (13.25%), PFHxA (11.47%), PFBS (10.99%), PFBA (10.69%). Three of the four carry no federal drinking-water standard.
- After upgrading locations to facility-grade coordinates (median about 30 m) and adjusting for regional and system covariates, source-distance features add little: repeated spatial-block gain in AUC of 0.010 to 0.023 by compound class, against covariate models at AUC about 0.73 to 0.81.
- In the facility-grade subset (n = 12,621), where positional error is minimal, every confidence interval on the lift spans zero.
- Source-density (count) features carry more information than nearest distance (any-PFAS lift 0.018 to 0.034), but the combined contribution remains small.
- Two-stage hurdle: among detected locations, a weak concentration signal remains only for PFOS (R2 lift 0.048) and PFOA (0.052), not for the unregulated compounds.
- Regulatory blind spot: 1,497 systems detect an unregulated short-chain compound while detecting neither PFOA nor PFOS. Joined to SDWIS population records, these systems serve an estimated 44 to 65 million people (reported as a range because summing population served double-counts people supplied by more than one system).
- Monitoring efficiency: ranking systems by distance to the nearest source and sampling the nearest quarter captures only 29.8% of unregulated detections, against the 25% expected from random sampling. Proximity-based targeting is close to random selection.
- Self-reported sources: systems reporting a potential PFAS source detect an unregulated compound at 43.3%, against 24.3% for systems reporting no known source. About a quarter of no-source systems still detect, indicating diffuse occurrence beyond catalogued point sources.
- National screening: a transferable model using only features available for every system (state-grouped AUC 0.650 for any PFAS, 0.659 for unregulated) is applied to the 39,437 active community water systems not sampled under UCMR5, producing a ranked screening map of where to test next.

The core claim is scoped to local point-source proximity. Regional structure may itself partly encode aggregate source density, so the result does not imply that sources are irrelevant at every scale.

## Data

Raw data are not included in this repository; they are large and publicly maintained. Download them and place under `data/raw/`:

- UCMR5 monitoring data: https://www.epa.gov/dwucmr
- Facility Registry Service (FRS): https://www.epa.gov/frs
- Safe Drinking Water Information System (SDWIS), via ECHO data downloads (population served): https://echo.epa.gov/tools/data-downloads

## Repository layout

```
src/                  analysis scripts (see pipeline below)
data/raw/             downloaded UCMR5, FRS, and SDWIS files (not tracked)
data/interim/         intermediate tables built by the pipeline (not tracked)
outputs/figures/      generated figures
outputs/tables/       generated result tables
```

## Pipeline

Scripts are in `src/`. Run from the repository root with the virtual environment active. The `probe_*.py` scripts are one-off data-inspection utilities and are not part of the main run.

| Step | Script | Produces |
|---|---|---|
| 1 | `acquire.py` | downloads and stages the raw UCMR5 and FRS inputs |
| 2 | `clean.py` | cleans and standardises the raw monitoring records |
| 3 | `geolocate.py` | initial ZIP-centroid coordinates for water systems |
| 4 | `geolocate_v2.py` | upgrades coordinates to facility-grade (FRS, median about 30 m) |
| 5 | `spatial.py` | nearest-source distances (`source_distances.csv`) |
| 6 | `source_counts.py` | source counts within 5 km and 10 km (`source_counts.csv`) |
| 7 | `features.py` | assembles per-location covariates |
| 8 | `dataset.py` | builds `data/interim/model_table.csv` (24,837 locations) |
| 9 | `dataset_concentration.py` | builds the detected-only concentration table |
| 10 | `model.py` | XGBoost detection models and SHAP (`model_results.csv`, `shap_importance.csv`) |
| 11 | `diagnostic.py` | state-shuffle test and single-split CV diagnostics |
| 12 | `robustness.py` | grouped CV: spatial-block and utility (PWS) grouped |
| 13 | `robustness_repeated.py` | 5x5 repeated CV, `robustness_repeated_delta.csv` (Table 2) |
| 14 | `robustness_geocoding.py` | full vs facility-grade subset, `robustness_geocoding_quality.csv` |
| 15 | `model_counts.py` | adds source-count features, `robustness_counts_delta.csv` |
| 16 | `model_concentration.py` | two-stage concentration, `stage2_concentration_delta.csv` |
| 17 | `exposure.py` | joins SDWIS population, population served by affected systems (`exposure_population.csv`, `exposure_breakdown.csv`) |
| 18 | `impact_analyses.py` | monitoring efficiency, blind-spot vulnerability profile, self-reported source (`monitoring_efficiency.csv`, `vulnerability_profile.csv`, `self_reported_source.csv`) |
| 19 | `predict_national.py` | transferable screening model for untested community water systems (`national_predictions.csv`, `top_priority_untested.csv`) |
| 20 | `make_figures.py` | main-text detection and decoupling figures |
| 21 | `make_exploratory_figures.py` | exploratory covariate and distance figures |
| 22 | `make_figure3_monitoring.py` | monitoring-efficiency capture curve |
| 23 | `make_graphical_abstract.py` | graphical abstract (Environmental Pollution spec) |
| — | `audit_numbers.py` | prints every number the manuscript cites, for verification |
| — | `make_toc_graphic.py` | an earlier abstract graphic kept for transparency |
| — | `make_adjusted_figure.py` | a covariate-adjusted residual figure explored during analysis but not used in the paper (kept for transparency) |

### Figure scripts read results, not hard-coded numbers

`make_figures.py`, `make_exploratory_figures.py`, and `make_figure3_monitoring.py` read every value at run time from `data/interim/model_table.csv` and `outputs/tables/*.csv`. They contain no hard-coded results, so regenerating them after a pipeline change cannot leave the figures out of sync with the data, and they stop with a clear error if a required table is missing.

## Method notes

- Spatial leakage. Random cross-validation leaks between neighbouring systems. Two grouped schemes are used: spatially blocked folds (one-degree grid cells held out whole) and utility-grouped folds (all sampling points of a system kept together).
- Intervals. Five-fold cross-validation repeated five times (25 estimates); both t-based and percentile 95% intervals are reported. Conclusions rest on the conservative percentile interval.
- Geocoding. The ZIP-to-facility shift (median 3.25 km) exceeds the median distance to the nearest industrial source (2.2 km), so the facility-grade subset analysis is the key robustness check.
- Population exposure. Population served is summed across affected systems and reported as a range, from a lower bound that excludes wholesale suppliers to an upper bound that includes them. Figures are population served, not unique individuals.
- National screening. The transferable model uses only features available for every U.S. public water system and is evaluated by state-grouped cross-validation, so its skill reflects generalisation to unseen states. It is intended as a prioritisation of where to sample next, not as an exposure estimate.

## Environment

```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ with pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP, Matplotlib, Pillow, pgeocode, and requests.

## Reproducibility

- Each modelling stage was first tested on synthetic data matching the real schema, then run on the full dataset.
- Large source files are read in chunks to control memory.
- Fixed random seeds throughout.
- The analysis code in this repository was developed with the assistance of an AI coding tool, and was reviewed, executed, and verified by the author on the real data. All results reported in the paper come from those verified runs.

## License

MIT. See LICENSE.

## Citation

Please cite the paper (citation to be added on publication) and this repository; see `CITATION.cff`.

## Contact

Naimul Islam. naimul.islam.bangladesh@gmail.com. https://github.com/Naimul-islam-bd
