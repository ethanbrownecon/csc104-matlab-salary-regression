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
valid = salary < 200000;
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

%% KNN Regression
k = 3;   % number of neighbors
y_pred = zeros(size(X_test));

for i = 1:length(X_test)
    distances = abs(X_train - X_test(i));     % distance from test point to all training points
    [~, sorted_idx] = sort(distances);        % sort distances
    nearest_idx = sorted_idx(1:k);            % take k nearest neighbors
    nearest_values = y_train(nearest_idx);    % salaries of nearest neighbors
    y_pred(i) = mean(nearest_values);         % average salary
end

%% Evaluate Performance
rmse = sqrt(mean((y_test - y_pred).^2));
disp(['KNN RMSE: ', num2str(rmse)]);

%% Deploy Model
new_experience = 5;
distances_new = abs(X_train - new_experience);
[~, sorted_idx_new] = sort(distances_new);
nearest_idx_new = sorted_idx_new(1:k);
predicted_salary = mean(y_train(nearest_idx_new));

disp(['Predicted salary for 5 years experience: ', num2str(predicted_salary)]);

%% Compare Predictions
figure;
scatter(X_test, y_test, 'filled'); hold on;
scatter(X_test, y_pred, 'r');
xlabel('Years of Experience');
ylabel('Salary');
title('KNN: Actual vs Predicted');
legend('Actual', 'Predicted');
grid on;

%% Final Visualization
x_line = linspace(min(X_train), max(X_train), 100)';
y_line = zeros(size(x_line));

for i = 1:length(x_line)
    distances = abs(X_train - x_line(i));
    [~, sorted_idx] = sort(distances);
    nearest_idx = sorted_idx(1:k);
    y_line(i) = mean(y_train(nearest_idx));
end

figure;
scatter(X_train, y_train, 'filled'); hold on;
plot(x_line, y_line, 'r', 'LineWidth', 2);
xlabel('Years of Experience');
ylabel('Salary');
title('KNN Regression Fit');
legend('Data', 'KNN Model');
grid on;
