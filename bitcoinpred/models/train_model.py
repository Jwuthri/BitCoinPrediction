# -*- coding: utf-8 -*-
"""
@author: JulienWuthrich
"""
import os
from math import sqrt
import plotly.offline as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

from bitcoinpred.config.settings import converged_data_path
py.init_notebook_mode(True)


class TrainLSTM(object):
    """Module to build, and train an LSTM"""

    def __init__(self, filename):
        """Initialize a `TrainLSTM` object.

            Arg:
                filename (str): path of the file
        """
        self.filename = filename
        self.split = 0.6

    def read_file(self):
        """Read the file.

            Return:
                data (pd.DataFrame): data
        """
        return pd.read_csv(self.filename)

    @staticmethod
    def show_price_evolution(df):
        """Plot the bitcoin values over the time.

            Arg:
                df (pd.DataFrame): data
        """
        btc_trace = go.Scatter(x=df['stamp'], y=df["price"], name="Price")
        py.iplot([btc_trace])

    @staticmethod
    def replace_0(df):
        """Clean NaN values.
            
            Arg:
                df (pd.DataFrame): data

            Return:
                df (pd.DataFrame): data cleaned
        """
        df["price"].replace(0, np.nan, inplace=True)
        df["price"].fillna(method="ffill", inplace=True)

        return df

    @staticmethod
    def scale(df):
        """Scale the price.
        
            Arg:
                df (pd.DataFrame): data

            Returns:
                scaled (array): data scaled
                scaler (MinMaxScaler obj): scaler
        """
        values = df["price"].values.reshape(-1, 1)
        values = values.astype("float32")
        scaler = MinMaxScaler(feature_range=(0, 1))

        return scaler.fit_transform(values), scaler

    def split_data(self, scaled):
        """Split the data in train/test.

            Arg:
                scaled (array): scaled

            Returns:
                train (array): train data
                test (array): test data
                train_size (int): split size
        """
        train_size = int(len(scaled) * self.split)
        test_size = len(scaled) - train_size
        train, test = scaled[0: train_size, :], scaled[train_size: len(scaled), :]

        return train, test, train_size

    @staticmethod
    def create_dataset(dataset, look_back):
        """Split dataset into X and Y.

            Arg:
                dataset (array): data
                look_back (int): shift

            Returns:
                X (array): X
                y (array): y
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            res = dataset[i: (i + look_back), 0]
            dataX.append(res)
            dataY.append(dataset[i + look_back, 0])

        return np.array(dataX), np.array(dataY)

    @staticmethod
    def reshape_X(df):
        """Reshape the values into 0 & 1.

            Arg:
                df (array): data

            Returns:
                X (array): X reshape
        """
        return np.reshape(df, (df.shape[0], 1, df.shape[1]))

    @staticmethod
    def build_model(trainX):
        """Build lstm model.

            Arg:
                trainX (array): data

            Returns:
                model (keras.lstm): model compiled
        """
        model = Sequential()
        model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss="mae", optimizer="adam")

        return model

    @staticmethod
    def build_model2(trainX):
        """Build lstm model.

            Arg:
                trainX (array): data

            Returns:
                model (keras.lstm): model compiled
        """
        model = Sequential()
        model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')

        return model

    @staticmethod
    def show_val_loss(history):
        """Show learning over the epochs.

            Arg:
                history (keras.lstm): model evolution
        """
        train = go.Scatter(y=history.history["loss"], name="train")
        test = go.Scatter(y=history.history["val_loss"], name="test")
        py.iplot([train, test])

    @staticmethod
    def show_pred_real(real, pred):
        """Show prediction vs True.

            Args:
                real (array): real values
                pred (array): predicted values
        """
        real = pd.DataFrame(real, columns=["real"])
        pred = pd.DataFrame(pred, columns=["pred"])
        x = go.Scatter(y=pred["pred"], name="pred")
        y = go.Scatter(y=real["ream"], name="real")
        py.iplot([y, x])

    @staticmethod
    def inverse_scale(scaler, val):
        """Reshape to original values.

            Args:
                scaler (MinMax obj): scaler
                val (array): values to reshape

            Return:
                reshaped (array): values reshaped
        """
        return scaler.inverse_transform(val.reshape(-1, 1))

    @staticmethod
    def reshape_Y(df):
        """Reshape the values into 0 & 1.

            Arg:
                df (array): data

            Returns:
                y (array): y reshape
        """
        return df.reshape(len(df))

    @staticmethod
    def show_by_date(predDates, real, pred):
        """Show perdiction over the dates.

            Args:
                predDates (array): list of dates
                real (array): real values
                pred (array): predicted values
        """
        actual = go.Scatter(x=predDates, y=real, name="Actual")
        pred = go.Scatter(x=predDates, y=pred, name="Pred")
        py.iplot([pred, actual])

    @staticmethod
    def show_corr(df):
        """Show correlation of columns.

            Arg:
                df (pd.DataFrame): data
        """
        data = go.Heatmap(
            z=df.corr().values.tolist(),
            x=list(df.corr().columns),
            y=list(df.corr().index)
        ) 
        py.iplot([data])

    @staticmethod
    def series_to_supervised(scaled):
        """Split the data in train/test.

            Arg:
                scaled (pd.DataFrame): scaled

            Return:
                df_scaled (pd.DataFrame): data with shifted columns
        """
        df_scaled = pd.DataFrame(scaled, columns=["var1(t-1)"])
        df_scaled["var1(t)"] = df_scaled.shift(-1)

        return df_scaled.dropna()

    def create_dataset_sentiment(dataset, look_back, sentiment):
        """Split dataset into X and Y.

            Arg:
                dataset (array): data
                look_back (int): shift

            Returns:
                X (array): X
                y (array): y
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            if i >= look_back:
                a = dataset[i - look_back: i + 1, 0].tolist()
                a.append(sentiment[i].tolist()[0])
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

        return np.array(dataX), np.array(dataY)

    def fit_transform(self):
        """need to be cleaned, REALLY."""
        df = self.read_file()
        df.dropna(axis=1, how="all")
        df["stamp"] = pd.to_datetime(df["stamp"].apply(str), format="%Y%m%d")
        self.show_price_evolution(df)

        df = self.replace_0(df)
        self.show_price_evolution(df)

        scaled, scaler = self.scale(df)
        sentiment = df["sentiment"].values.reshape(-1, 1).astype("float32")
        
        train, test, train_size = self.split_data(scaled)
        trainX, trainY = self.create_dataset(train, 1, sentiment[0: train_size])
        testX, testY = self.create_dataset(test, 1, sentiment[train_size:])
        
        trainX = self.reshape_X(trainX)
        testX = self.reshape_X(testX)
        
        model = self.build_model(trainX)
        history = model.fit(trainX, trainY, epochs=150, batch_size = 64, validation_data=(testX, testY), verbose=1, shuffle=False)
        self.show_val_loss(history)
        
        pred = model.predict(testX)
        self.show_pred_real(testY, pred)

        pred_inv = self.inverse_scale(scaler, pred)
        real_inv = self.inverse_scale(scaler, testX)
        rmse = sqrt(mean_squared_error(real_inv, pred_inv))
        print("test", rmse)
        self.show_pred_real(real_inv, pred_inv)

        predDates = df.tail(len(testX))["stamp"]
        real_inv_resh = self.reshape_Y(real_inv)
        pred_inv_resh = self.reshape_Y(pred_inv)
        self.show_by_date(predDates, real_inv_resh, pred_inv_resh)

        self.show_corr(df)
        scaled, scaler = self.scale(df)
        reframed = self.series_to_supervised(scaled)

        train, test, train_size = self.split_data(reframed.values)
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        multi_model = self.build_model(train_X)
        multi_history = multi_model.fit(trainX, trainY, epochs=150, batch_size = 64, validation_data=(test_X, test_y), verbose=1, shuffle=False)
        self.show_val_loss(multi_history)

        multi_pred = model.predict(test_X)
        self.show_pred_real(test_y, multi_pred)

        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1]))
        inv_multi_pred = np.concatenate((multi_pred, test_X[:, 1:]), axis=1)
        inv_multi_pred = scaler.inverse_transform(inv_multi_pred)
        inv_multi_pred = inv_multi_pred[:, 0]

        test_y = test_y.reshape((len(test_y), 1))
        inv_multi_real = np.concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_multi_real = scaler.inverse_transform(inv_multi_real)
        inv_multi_real = inv_multi_real[:, 0]

        rmse = sqrt(mean_squared_error(inv_multi_real, inv_multi_pred))
        print("test", rmse)

        actual_chart = go.Scatter(x=predDates, y=inv_multi_real, name= 'Actual Price')
        multi_predict_chart = go.Scatter(x=predDates, y=inv_multi_pred, name= 'Multi Predict Price')
        predict_chart = go.Scatter(x=predDates, y=pred_inv_resh, name= 'Predict Price')
        py.iplot([predict_chart, multi_predict_chart, actual_chart])

        reframed["sentiment"] = df["sentiment"].head(-1)
        train, test, train_size = self.split_data(reframed.values)

        trainX, trainY = create_dataset(train, 1, sentiment[:train_size],sent=True)
        testX, testY = create_dataset(test, 1, sentiment[train_size:], sent=True)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        multi_model2 = self.build_model2(train_X)
        multi_history2 = multi_model2.fit(trainX, trainY, epochs=150, batch_size = 64, validation_data=(testX, testY), verbose=1, shuffle=False)
        self.show_val_loss(multi_history2)

        multi_pred2 = model.predict(testX)
        self.show_pred_real(testY, multi_pred2)

        pred_inverse_sent = scaler.inverse_transform(multi_pred2.reshape(-1, 1))
        testY_inverse_sent = scaler.inverse_transform(testY.reshape(-1, 1))

        rmse_sent = sqrt(mean_squared_error(testY_inverse_sent, pred_inverse_sent))
        print('Test RMSE: %.3f' % rmse_sent)

        a = pd.DataFrame(pred_inverse_sent, columns=["a"])
        b = pd.DataFrame(testY_inverse_sent, columns=["a"])
        actual_chart = go.Scatter(x=predDates, y=a["a"], name= 'Actual Price')
        multi_predict_chart = go.Scatter(x=predDates, y=b["a"], name= 'Multi Predict Price')
        py.iplot([multi_predict_chart, actual_chart])

        
if __name__ == '__main__':
    path = os.path.join(converged_data_path, "merged.csv")
    TrainLSTM(path).fit_transform()
