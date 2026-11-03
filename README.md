# Salary Prediction with MATLAB

A course project for **CSC104: Computing for Scientists and Engineers**. This project explores how years of work experience relate to salary using simple regression models in MATLAB.

## Project overview

The scripts clean the data, report the ranges of experience and salary, remove records with missing values, and exclude salaries of $200,000 or more. They then make an 80/20 train/test split and compare predictions:

- [`final.m`](./final.m) standardizes experience and fits linear and quadratic polynomial regression models. It reports each model's test-set RMSE and plots the linear fit.
- [`dt.m`](./dt.m) uses a 3-nearest-neighbors (KNN) regression model and reports its test-set RMSE.

Both scripts use a fixed random seed so the train/test split is repeatable. They use years of experience as the only predictor; the results are an educational demonstration, not a compensation recommendation or a validated salary-estimation tool.

## Requirements

- MATLAB with support for `readtable` and `isfile`
- The project CSV file, named `job_salary_prediction_dataset.csv`, placed in the same folder as the scripts

The dataset is not included in this repository because its source and redistribution terms have not been verified. Obtain a copy you are permitted to use, save it under the filename above, and ensure it has numeric `experience_years` and `salary` columns.

## Run

From MATLAB, set the current folder to this repository and run either script:

```matlab
run('final.m')
run('dt.m')
```

Each script opens figures and prints diagnostic ranges, a test-set RMSE, and a prediction for five years of experience in the Command Window.

## Notes

- The $200,000 salary cutoff is a simplifying choice made for this class project and may affect the results.
- `dt.m` computes exact distances to the training set for each test sample, so it can take a while on a large dataset.
- This repository contains coursework examples; model evaluation and data provenance should be improved before using the results beyond the assignment.
