# UMI Dataset & IWOA-RF Analysis Code

**Repository for:** *Research on Informatization Maturity Modeling and Prediction of University Management Based on Improved Whale Optimization Algorithm and Random Forest*

## Overview

This repository accompanies the manuscript submitted to SAGE Open. It contains:

- `UMI_dataset.csv` — the UMI (University Management Informatization) dataset (2,868 records × 17 columns)
- `experiment_umi_iwoa_rf.py` — main experiment script (data generation → normalization → PCA → IWOA-RF → 8 baselines → tables/figures)
- `run_staged.py` — staged driver for reproducible long-running experiments
- `DATA_SOURCES.md` — full list of the official public data sources used for calibration

## Data description (read this first)

The UMI dataset is a **real-data-calibrated synthetic dataset**.

- **Institution attributes are real**: the 2,868 regular higher-education institutions (1,308 undergraduate, 1,560 vocational colleges) come from the official *National List of Higher Education Institutions* (Ministry of Education, China, 2024). Region and institution-type distributions follow that real list.
- **Indicator parameters are calibrated to real official statistics**: the *National Education Development Statistical Bulletins*, the *China Higher Education Informatization/Digitalization Development Reports (2020–2024)*, and the *Basic School-Running Condition Indicators for Regular Higher Education Institutions (Jiao Fa [2004] No. 2)*.
- **The per-institution indicator values are simulated** by a fixed-seed simulator (seed = 42). Per-institution informatization indicators (e.g., server virtualization rate, cross-department data-sharing rate) are not published for individual universities in any public channel, which is why a calibrated simulation is used.

Because the framework reads only the columns defined in Table 2 of the manuscript, this dataset can be replaced by any real aggregated institution-level dataset sharing those columns without modifying the method.

**No ethics approval is required**: the dataset contains no personally identifiable information and no individual-level data; no human participants are involved.

## Columns (17)

`University_ID`, `Region`, `Institution_Type`, `Student_Scale`, `Faculty_Scale`, `Education_Budget_wan`, `Per100_Computers`, `Network_Coverage_%`, `IT_Budget_Ratio_%`, `IT_Staff_Ratio_%`, `Info_System_Count`, `Data_Governance_Score`, `Online_Service_Rate_%`, `Digital_Literacy_Score`, `Cybersecurity_Investment_%`, `System_Uptime_%`, `Management_Status` (target: Optimal / Suboptimal)

## Reproduction

Requirements: Python 3.12, NumPy, Pandas, Scikit-learn, XGBoost, SciPy, Matplotlib, python-docx.

```bash
# 1. Full experiment (data, preprocessing, PCA, IWOA-RF, baselines, tables, figures)
python experiment_umi_iwoa_rf.py

# 2. Quick smoke test
python experiment_umi_iwoa_rf.py --quick

# 3. Staged run (long experiments, resumable)
python run_staged.py --stage main
python run_staged.py --stage cv1
python run_staged.py --stage cv2
python run_staged.py --stage frep
python run_staged.py --stage assemble
```

On Windows, set `JOBLIB_TEMP_FOLDER` to an ASCII path before running (the username contains non-ASCII characters).

## Results summary

- IWOA-RF: 86.99% accuracy, 88.24% F1, 0.928 AUC (best among random-forest variants)
- Strongest baseline: RBF-SVM 87.69% (margin small but significant, p = 0.0008)
- 5-fold CV: 85.88% ± 1.59%
- 28 references; all figures at 300 dpi

## Data sources

See `DATA_SOURCES.md` for the complete list of official public sources (Ministry of Education national list, statistical bulletins, informatization development reports, school-running condition indicators, provincial evaluation framework, policy documents) and international references (EDUCAUSE CDS, EUNIS BencHEIT, THE Digital Maturity Index).

## License

The dataset and code are released for reproducibility of the associated manuscript. Please cite the manuscript when using this material.

## Contact

Yunlong Wang — Anhui Wenda University of Information Engineering, Hefei, Anhui, China.
