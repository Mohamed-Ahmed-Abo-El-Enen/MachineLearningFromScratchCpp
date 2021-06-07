import DataManipulation as Dp
import NeuralNetwork as Nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class MLP:

    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.full_dataset_index = None
        self.columns_name = None

    def GetDataset(self, xlsx_file_path, n_steps_in, n_steps_out):
        data_manipulation = Dp.DataManipulation()
        df, self.columns_name = data_manipulation.read_dataset(xlsx_file_path)

        X, y = data_manipulation.split_sequences(df.values, n_steps_in, n_steps_out)

        self.full_dataset_index = y[:, :, 0]
        X = X[:, :, 1:].astype(float)
        y = y[:, :, 1:-2].astype(float)

        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])

        return X, y

    def normalize_x_y(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)

        return X, y

    def inv_normalize_y(self, y):
        return self.scaler_y.inverse_transform(y)

    def split_dataset_x_y(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True)
        return X_train, X_test, y_train, y_test

    def define_mlp_model(self, model_params):
        model = Nn.NeuralNetwork(model_params["loss_function"])

        model.add_layer(input_dimension=model_params["input_dimension"], units=model_params["layers"][0][0],
                        activation=model_params["layers"][0][1])

        layers = model_params["layers"]
        for index in range(1, len(layers)):
            model.add_layer(units=layers[index][0], activation=layers[index][1])

        print(model)
        return model

    def train(self, model, X, y):
        A = model.forward(X)
        model.backward(A, y)
        model.update()
        return model

    def predict(self, model, X):
        return model.forward(X)

    def mean_square_error(self, y_true, yhat):
        return 1 / y_true.shape[1] * np.sum(np.square(y_true - yhat))

    def print_mean_square_error(self, y_true, yhat):
        print("Model Mean Square Error : "+ str(np.squeeze(self.mean_square_error(y_true, yhat))))

    def __iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]
        indices = range(inputs.shape[0])
        if shuffle:
            indices = np.random.permutation(inputs.shape[0])
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

    def train_with_min_batch(self, model, X_train, y_train, X_test, y_test, batch_size, epochs):
        train_log = []
        val_log = []

        for epoch in range(epochs):
            for x_batch, y_batch in self.__iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=False):
                self.train(model, x_batch, y_batch)
            train_log.append(self.mean_square_error(y_train, self.predict(model, X_train).T))
            val_log.append(self.mean_square_error(y_test, self.predict(model, X_test).T))
            if epoch % 100 == 0:
                print("Cost in epoch "+str(epoch)+" : "+str(train_log[-1]))

        return train_log, val_log

    def plot_cost_function(self, epoch, train_log, val_log):
        print("Epoch", epoch)
        print("Train mea:", train_log[-1])
        print("Val mea:", val_log[-1])
        plt.plot(train_log, label='train mea')
        plt.plot(val_log, label='val mea')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def save_result_to_csv(self, data, output_file):
        df = pd.DataFrame(data, columns=self.columns_name, index=self.full_dataset_index[:,0])
        df.to_csv(output_file)