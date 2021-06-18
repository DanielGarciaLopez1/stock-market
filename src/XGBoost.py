

import pandas as pd
import numpy as np
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import mean_absolute_error, mean_squared_error


class SalesPredictionModel:
    def __init__(self, data_url, n_inputs=1, n_outputs=1, debug=False):
        self.data_url = data_url
        self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        self.__created = False
        self.__processed = False
        self.__debug = debug
        return

    def preprocess(self):
        self.preprocessed_data = self.data_url
        self.final_data = self.__series_to_supervised(self.preprocessed_data)
        self.__processed = True
        return

    def optimize(self, n_test):
        # GridSearchCV to find the best parameters for the XGBRegressor
        if not self.__processed:
            raise Exception('You must get and process the data. Call \'yourmodelname.preprocess()\' before.')
        train, test = self.__train_test_split(n_test)
        train_X, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        param_grid = {
            'objective': ['reg:squarederror', 'reg:linear'],
            'n_estimators': [400, 700, 1000],
            'colsample_bytree': [0.7, 0.8, 1],
            'max_depth': [3, 6, 9],
            'reg_alpha': [0, 0.2, 0.4],
            'reg_lambda': [1, 1.2, 1.4],
            'subsample': [0.7, 0.8, 0.9]
        }
        opt_model = xgb.XGBRegressor()
        grid = GridSearchCV(opt_model,
                            param_grid,
                            scoring='neg_mean_absolute_error',
                            cv=5,
                            n_jobs=5,
                            verbose=self.__debug)
        grid.fit(train_X, train_y)
        return grid.best_score_, grid.best_params_

    def fit(self, n_test, history=None):
        if not self.__processed:
            raise Exception('You must get and process the data. Call \'yourmodelname.preprocess()\' before.')
        if history is None:
            train, test = self.__train_test_split(n_test)
            train_X, train_y = train.drop('Ventas_totales(t)', axis=1), train.drop(train.columns.difference(['Ventas_totales(t)']), axis=1)
            self.__created = True
            self.__model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            self.__model.fit(train_X, train_y)
            return
        else:
            train_X, train_y = history.drop('Ventas_totales(t)', axis=1), history.drop(history.columns.difference(['Ventas_totales(t)']), axis=1)
            validation_model = xgb.XGBRegressor(objective='reg:squarederror',
                                                n_estimators=700,
                                                colsample_bytree=1,
                                                max_depth=3,
                                                reg_alpha=0,
                                                reg_lambda=1,
                                                subsample=0.9)
            validation_model.fit(train_X, train_y)
            return validation_model

    def predict(self, test_X):
        if not self.__created:
            raise Exception('You must fit the model with data before predict a value. Call \'yourmodelname.fit()\' before.')
        return self.__model.predict(test_X)

    def validate(self, n_test):
        """
        Function that evaluates all the predictions obtained with our model with a walk forward validation method
            - data
            - n_test: desired number of predictions
        """
        if not self.__created:
            raise Exception('You must fit the model with data before predict a value. Call \'yourmodelname.fit()\' before.')
        predictions = list()
        expected = list()
        train, test = self.__train_test_split(n_test)
        test_X, test_y = test.drop('Ventas_totales(t)', axis=1), test.drop(test.columns.difference(['Ventas_totales(t)']), axis=1)
        history = train.copy(deep=True)
        for i in range(n_test):
            test_X_iter, test_y_iter = test_X.iloc[i:i+1, :], test_y.iloc[i:i+1, :]
            test_model = self.fit(n_test, history=history)
            y_prediction = test_model.predict(test_X_iter)
            new_row = [test_X_iter.iloc[0, j] for j in range(len(test_X_iter.columns))]
            new_row.append(y_prediction[0])
            history.loc[len(history)] = new_row
            if self.__debug:
                print("Prediction: %.1f | Expected: %.1f | GAP: %.1f" % (y_prediction[0], test_y_iter.iloc[0, 0], test_y_iter.iloc[0, 0] - y_prediction[0]))
            predictions.append(y_prediction[0])
            expected.append(test_y_iter.iloc[0, 0])
        mae = mean_absolute_error(expected, predictions)
        rmse = np.sqrt(mean_squared_error(expected, predictions))
        return expected, predictions, mae, rmse

    def __series_to_supervised(self, data):
        """
        Function to transform time-series data to supervised training data
            - data
            - n_inputs: Number of days to look back
            - n_outputs: Number of days to look after
            - dropnan
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        new_data = pd.DataFrame(data)
        columns, names = list(), list()
        # With this loop we extract the input values, i.e. the values of the variables
        # for the previous n_inputs time steps
        for i in range(self.__n_inputs, 0, -1):
            columns.append(new_data.shift(i))
            names += [('%s(t-%d)' % (data.columns[j], i)) for j in range(n_vars)]
        # With this loop we extract the actual values of the variables for the actual
        # time step and, if desired, we will add the next n_outputs time steps as columns
        for i in range(0, self.__n_outputs):
            columns.append(new_data.shift(-i))  
            if i == 0:
                names += [('%s(t)' % (data.columns[j])) for j in range(n_vars)]
            else:
                names += [('%s(t+%d)' % (data.columns[j], i)) for j in range(n_vars)]
        data_result = pd.concat(columns, axis=1)
        data_result.columns = names
        data_result.dropna(inplace=True)
        data_result = data_result.infer_objects()
        return data_result

    def __train_test_split(self, n_test):
        """
        Function to split dataset into train and test
            - data
            - n_test: desired number of rows in test dataset
        """
        return self.final_data[:-n_test].reset_index(drop=True), self.final_data[-n_test:].reset_index(drop=True)