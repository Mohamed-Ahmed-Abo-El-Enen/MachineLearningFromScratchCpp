import MLP as mlp
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

dataset_file_path = "Pharmacy Sales Forecasting Dataset.xlsx"
n_steps_in, n_steps_out = 5, 1

mlp = mlp.MLP()
X, y = mlp.GetDataset(dataset_file_path, n_steps_in, n_steps_out)
X, y = mlp.normalize_x_y(X, y)
X_train, X_test, y_train, y_test = mlp.split_dataset_x_y(X, y, 0.05)

regression_model = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train)
yhat = regression_model.predict(X_train)
yhat = mlp.inv_normalize_y(yhat)
print("------------------------------\n------------------------------\n------------------------------\n")
print("------------------------------\n------------------------------\n------------------------------\n")
print("Y Train Predicted")
print(yhat)
print("------------------------------\n------------------------------\n------------------------------\n")
print("Y Train True")
y_train = mlp.inv_normalize_y(y_train)
print(y_train)
print(mean_absolute_error(y_train, yhat))

print("------------------------------\n------------------------------\n------------------------------\n")
print("------------------------------\n------------------------------\n------------------------------\n")
yhat = regression_model.predict(X_test)
yhat = mlp.inv_normalize_y(yhat)
print("Y Test True")
print(yhat)
print("------------------------------\n------------------------------\n------------------------------\n")
print("Y Train True")
y_test = mlp.inv_normalize_y(y_test)
print(y_test)
print(mean_absolute_error(y_test, yhat))

