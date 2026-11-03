clear; clc; close all;

%% Importing Data
script_dir = fileparts(mfilename('fullpath'));
data_path = fullfile(script_dir, 'job_salary_prediction_dataset.csv');
if ~isfile(data_path)
    error('Dataset not found: %s. Place the CSV beside this script.', data_path);
end
data = readtable(data_path);

%% Explore Data & Extract Features
experience = data.experience_years;
salary = data.salary;

% Remove missing values
valid = ~isnan(experience) & ~isnan(salary);
experience = experience(valid);
salary = salary(valid);

%% Basic Data Diagnostics
disp('Experience Range:');
disp([min(experience), max(experience)])

disp('Salary Range:');
disp([min(salary), max(salary)])

%% Remove Outliers
valid = salary < 200000;   % cap extreme salaries
experience = experience(valid);
salary = salary(valid);

%% Quick visualization
figure;
scatter(experience, salary, 'filled');
xlabel('Years of Experience');
ylabel('Salary');
title('Salary vs Experience (Cleaned Data)');
grid on;

%% Split Data (Train/Test)
rng(1);
n = length(experience);
idx = randperm(n);

train_size = round(0.8 * n);

train_idx = idx(1:train_size);
test_idx = idx(train_size+1:end);

X_train = experience(train_idx);
y_train = salary(train_idx);

X_test = experience(test_idx);
y_test = salary(test_idx);

%% Normalize Data (NEW)
mu = mean(X_train);
sigma = std(X_train);

X_train_norm = (X_train - mu) / sigma;
X_test_norm  = (X_test - mu) / sigma;

%% Train Model (Linear)
p = polyfit(X_train_norm, y_train, 1);

%% Evaluate Performance
y_pred = polyval(p, X_test_norm);

% Calculate RMSE
rmse = sqrt(mean((y_test - y_pred).^2));
disp(['Linear RMSE (Normalized): ', num2str(rmse)]);

%% Deploy Model
new_experience = 5;
new_experience_norm = (new_experience - mu) / sigma;

predicted_salary = polyval(p, new_experience_norm);
disp(['Predicted salary for 5 years experience: ', num2str(predicted_salary)]);

%% Improve Performance (Quadratic Model)
p2 = polyfit(X_train_norm, y_train, 2);
y_pred2 = polyval(p2, X_test_norm);

rmse2 = sqrt(mean((y_test - y_pred2).^2));
disp(['Quadratic RMSE (Normalized): ', num2str(rmse2)]);

%% Compare Predictions
figure;
scatter(X_test, y_test, 'filled'); hold on;
scatter(X_test, y_pred, 'r');
xlabel('Years of Experience');
ylabel('Salary');
title('Actual vs Predicted (Normalized Model)');
legend('Actual', 'Predicted');
grid on;

%% Final Visualization (Model Fit)
figure;
scatter(X_train, y_train, 'filled'); hold on;

x_line = linspace(min(X_train), max(X_train), 100);
x_line_norm = (x_line - mu) / sigma;

y_line = polyval(p, x_line_norm);

plot(x_line, y_line, 'r', 'LineWidth', 2);

xlabel('Years of Experience');
ylabel('Salary');
title('Regression Fit (Normalized Linear Model)');
legend('Data', 'Model');
grid on;
