import MLP

if __name__ == "__main__":
    dataset_file_path = "Pharmacy Sales Forecasting Dataset.xlsx"
    n_steps_in, n_steps_out = 5, 1

    mlp = MLP.MLP()

    X, y = mlp.GetDataset(dataset_file_path, n_steps_in, n_steps_out)
    X, y = mlp.normalize_x_y(X, y)
    X_train, X_test, y_train, y_test = mlp.split_dataset_x_y(X, y, 0.05)

    model_params = {"input_dimension": X_train.shape[1],
                    "layers": [(128, "relu"), (6, "linear")],
                    "loss_function": 'mean-square-error'}

    model = mlp.define_mlp_model(model_params)
    batch_size, epochs = 1, 1000

    train_log, val_log = mlp.train_with_min_batch(model, X_train, y_train, X_test, y_test, batch_size, epochs)
    mlp.plot_cost_function(epochs, train_log, val_log)

    yhat = mlp.predict(model, X_train).T
    mlp.print_mean_square_error(y_train, yhat)

    print("------------------------------\n------------------------------\n------------------------------\n")
    yhat = mlp.inv_normalize_y(yhat)
    print("Train Predicted values")
    print(yhat)

    print("------------------------------\n------------------------------\n------------------------------\n")
    y_train = mlp.inv_normalize_y(y_train).T
    print("Train True values")
    print(y_train)

    print("------------------------------\n------------------------------\n------------------------------\n")
    yhat = mlp.predict(model, X_test).T
    yhat = mlp.inv_normalize_y(yhat)
    print("Test Predicted values")
    print(yhat)

    print("------------------------------\n------------------------------\n------------------------------\n")
    y_test = mlp.inv_normalize_y(y_test).T
    print("Test True values")
    print(y_test)

    print("------------------------------\n------------------------------\n------------------------------\n")
    predicted_csv_file_path = "predicted_result.csv"
    yhat = mlp.predict(model, X).T
    mlp.print_mean_square_error(y, yhat)

    yhat = mlp.inv_normalize_y(yhat)
    mlp.save_result_to_csv(yhat, predicted_csv_file_path)