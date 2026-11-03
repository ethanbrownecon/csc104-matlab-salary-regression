# Salary Prediction with MATLAB

A course project for CSC104, Computing for Scientists and Engineers.

## Question

How well do years of work experience predict salary, and does a simple nearest neighbors model do better than a polynomial fit?

## Why it matters

Salary and experience are a familiar pair, which makes them a good setting for learning how to fit, compare, and evaluate regression models on held out data.

## Approach

The scripts clean the data, report the ranges of experience and salary, remove records with missing values, and exclude salaries of $200,000 or more. They then make an 80/20 train and test split with a fixed random seed and compare test set RMSE across models:

- [`final.m`](./final.m) standardizes experience and fits linear and quadratic polynomial regressions, then plots the linear fit.
- [`dt.m`](./dt.m) fits a 3 nearest neighbors regression.

Years of experience is the only predictor, so the results are an educational demonstration and not a salary estimation tool. The $200,000 cutoff is a simplifying choice that may affect results.

## Requirements

- MATLAB with support for `readtable` and `isfile`.
- A file named `job_salary_prediction_dataset.csv` in the same folder as the scripts, with numeric `experience_years` and `salary` columns.

The dataset is not included because its source and redistribution terms have not been verified.

## Run

```matlab
run('final.m')
run('dt.m')
```

Each script opens figures and prints data ranges, a test set RMSE, and a prediction for five years of experience. `dt.m` computes exact distances to every training point, so it can be slow on large datasets.
