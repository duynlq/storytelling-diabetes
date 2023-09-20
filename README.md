![banner](images/hotel_incentive.png)

![Tools](https://img.shields.io/badge/Tools-Python,_SQL,_Tableau-yellow)
![Methods](https://img.shields.io/badge/Methods-Webscraping,_NLP,_Supervised_ML-red)
![GitHub last commit](https://img.shields.io/github/last-commit/duynlq/storytelling-diabetes)
![GitHub repo size](https://img.shields.io/github/repo-size/duynlq/storytelling-diabetes)
Badge [source](https://shields.io/)

## Problem Statement
- Diabetes is a serious health problem that affects many Americans. In clinical care, there are many risk factors that directly affects the likelihood of a patient with diabetes to be readmitted within 30 days of discharge.
- A Logistic Regression model with SMOTE was performed to classify patients in this category with an accuracy of 62% ([more details here](https://shields.io/)), however it was a good starting point to pinpoint the top 10 important features that contribute to the increase of classification for hospital readmission.

## Key Findings
- The top 10 important features are listed in the table below, along with their regressor weight.
- A [Tableau Dashboard](https://public.tableau.com/app/profile/duy.nguyen7683/viz/USHospitalsDiabetesHub/Dashboard2?publish=yes) was created to visualize these 10 important features. 

| Num | Feature | Weight | Note |
| -------- | ------- | ------- | -------- |
| 1 | discharge_disposition_id_8 | 0.612653 | Transferred to home under care of Home IV provider |
| 2 | admission_source_id_7 | 0.269167 | Admission by Emergency Room |
| 3 | gender_Female | 0.268416 | Distribution of females are slightly less than males |
| 4 | metformin-rosiglitazone_No | 0.175598 | Medicine combination used to treat type 2 diabetes |
| 5 | admission_source_id_8 | 0.169067 | Admission by Court/Law Enforcement |
| 6 | num_procedures | 0.113014 | Number of procedures done |
| 7 | max_glu_serum_>300 | 0.070908 | Simple and direct single test for diabetes |
| 8 | diag_3_Diabetes | 0.060908 | Diabetes as one of patient’s diagnoses |
| 9 | miglitol_No | 0.058262 | Oral anti-diabetic drug that helps patient breaks down complex carbohydrates into glucose |
| 10 | glipizide-metformin_Steady | 0.043848 | Medicine combination used to treat high blood sugar levels caused by type 2 diabetes |

## Data Source
- [Diabetes 130-US hospitals for years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes%20130-us%20hospitals%20for%20years%201999-2008)
- [Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records](https://www.hindawi.com/journals/bmri/2014/781670/tab2/)
