# Data Sources — UMI Dataset

The UMI dataset is a **real-data-calibrated synthetic dataset**. Its institution attributes follow the official national list; its indicator distributions are calibrated against official public statistics.

## Domestic official sources

| # | Source | Used for |
|---|---|---|
| S1 | National List of Higher Education Institutions (Ministry of Education, China, 2024-06-20) — http://www.moe.gov.cn/jyb_xxgk/s5743/s5744/202406/t20240621_1136990.html | Institution roster, province, institution type (2,868 institutions: 1,308 undergraduate, 1,560 vocational) |
| S2 | 2024 National Education Development Statistical Bulletin (Ministry of Education) — http://www.moe.gov.cn/jyb_sjzl/sjzl_fztjgb/202506/t20250611_1193760.html | Student/faculty scale, budget totals |
| S3 | China Higher Education Informatization/Digitalization Development Reports (2020–2024, Research Center for Scientific Development of Higher Education, MOE) — http://www.cutech.edu.cn/detail/46-423 ; https://nic.yxnu.edu.cn/info/1009/2002.htm | IT budget ratio, IT staff ratio, network coverage, data governance, cybersecurity distributions |
| S4 | Higher Education Informatization Monitoring Questionnaire (2024) — http://www.cutech.edu.cn/images/20240531/becb424c9a9ce106a443057e.pdf | Field definitions and collection criteria |
| S5 | Basic School-Running Condition Indicators for Regular Higher Education Institutions (Jiao Fa [2004] No. 2) — https://www.moe.gov.cn/srcsite/A03/s7050/200402/t20040206_180515.html | Per-100-student computers, school-running condition thresholds |
| S6 | Anhui Provincial Higher Education Informatization Evaluation Indicator System 2.0 (trial) | Maturity evaluation framework for the target variable |
| S7 | Education Informatization 2.0 Action Plan (Jiao Ji [2018] No. 6) — http://www.moe.gov.cn/srcsite/A16/s3342/201804/t20180425_334188.html | Policy background |

## International reference sources

| # | Source | Used for |
|---|---|---|
| I1 | EDUCAUSE Core Data Service (CDS) Interactive Almanac / IT Spending and Staffing | Cross-check on IT budget/staff ratios (US institutions) |
| I2 | EUNIS BencHEIT (European Universities IT Benchmarking) — https://eunis.org/task-forces/benchmarking/ | Cross-check on IT benchmarking (Europe) |
| I3 | THE Digital Maturity Index — https://resources.timeshighereducation.com/media/digital-maturity-index | Reference on digital maturity |

## Provenance statement

Institution attributes (S1) are real. Per-institution indicator values are simulated by a fixed-seed simulator (seed = 42) whose distribution parameters are calibrated against S1–S7. Per-institution informatization indicators are not published for individual universities in any public channel; a calibrated simulation is therefore used, and is disclosed transparently in the manuscript and here.

For the full journal-ready Data Availability Statement, see the manuscript.
