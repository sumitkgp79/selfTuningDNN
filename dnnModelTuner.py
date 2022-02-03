import logging
import pdb
import pandas as pd
import numpy as np
import ray
from ray import tune
from tensorflow.python import keras
import xgboost as xgb
import time
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)  # mean_absolute_percentage_error
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
import sys
from sklearn.model_selection import train_test_split
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer
from tensorflow.keras.layers import BatchNormalization
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from tensorflow.keras.constraints import NonNeg

# ray._private.services.address_to_ip = lambda _node_ip_address: '127.0.0.1'
ray.init(include_dashboard=False)
# # ray.init(local_mode=True)


def dnnAutoTuner(dfTrain, dfTest):
    '''
    Function: dnnAutoTuner
    Summary: for tuning hyperParams of DNN with defined space 
    Attributes: 
        @param (dfTrain):dict for training data having keys as 'xData' and 'yData'
        @param (dfTest):dict for testing data having keys as 'xData' and 'yData'
    Returns: trained model and best hyper parameters
    '''
    try:
        # pdb.set_trace()
        startTime = time.time()
        X_train = dfTrain["xData"]
        X_test = dfTest["xData"]
        y_train = dfTrain["yData"]
        y_test = dfTest["yData"]

        space = {
            "units1": hp.choice("units1", [2, 4, 6, 8, 10, 12, 16, 24, 48]),
            "units2": hp.choice("units2", [2, 4, 6, 8, 10, 12, 16, 24, 48]),
            "dropout1": hp.choice("dropout1", [0.05, 0.15, 0.25, 0.5, 0.75]),
            "dropout2": hp.choice("dropout2", [0.05, 0.15, 0.25, 0.5, 0.75]),
            "batch_size": hp.choice("batch_size", [4, 8, 12, 16]),
            "epochs": hp.choice("epochs", [500, 1000, 2000, 3000, 4000, 5000, 6000]),
            "activation": "relu",
            "lr": hp.choice("lr", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
        }

        def train_dnn(space):
            params = space
            model = Sequential()

            model.add(
                Dense(
                    units=int(params["units1"]),
                    input_dim=X_train.shape[1],
                    kernel_initializer="normal",
                    bias_initializer="zeros",
                    activation="relu",
                    kernel_constraint=NonNeg(),
                    bias_constraint=NonNeg()
                )
            )
            model.add(Dropout(params["dropout1"]))
            model.add(BatchNormalization())
            model.add(
                Dense(
                    units=int(params["units2"]),
                    activation="relu",
                    bias_initializer="zeros",
                    kernel_initializer="normal",
                    kernel_constraint=NonNeg(),
                    bias_constraint=NonNeg()
                )
            )
            model.add(Dropout(params["dropout2"]))
            model.add(BatchNormalization())
            model.add(Dense(units=1, kernel_initializer="normal", kernel_constraint=NonNeg(),
                            bias_constraint=NonNeg()))

            adam = Adam(learning_rate=params["lr"])
            model.compile(loss="mse", metrics=[
                          "mean_squared_error"], optimizer=adam)

            model.fit(
                X_train,
                y_train,
                epochs=int(params["epochs"]),
                validation_data=(X_test, y_test),
                batch_size=int(params["batch_size"]),
                verbose=0,
                callbacks=[TuneReportCallback(
                    {"mean_se": "mean_squared_error"})],
            )

        def tune_dnn():

            sched = AsyncHyperBandScheduler(
                time_attr="training_iteration", max_t=1000, grace_period=100
            )
            search_alg = HyperOptSearch(space, metric="mean_se", mode="min")
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=8)

            analysis = tune.run(
                train_dnn,
                search_alg=search_alg,
                config=space,
                metric="mean_se",
                mode="min",
                scheduler=sched,
                verbose=3,
                num_samples=100,
            )
            return analysis.best_config

        best = tune_dnn()
        print(best)

        def dnn_returner(space):
            params = space
            model = Sequential()

            model.add(
                Dense(
                    units=int(params["units1"]),
                    input_dim=X_train.shape[1],
                    kernel_initializer="normal",
                    bias_initializer="zeros",
                    activation="relu",
                    kernel_constraint=NonNeg(),
                    bias_constraint=NonNeg()
                )
            )
            model.add(Dropout(params["dropout1"]))
            model.add(BatchNormalization())
            model.add(
                Dense(
                    units=int(params["units2"]),
                    activation="relu",
                    bias_initializer="zeros",
                    kernel_initializer="normal",
                    kernel_constraint=NonNeg(),
                    bias_constraint=NonNeg()
                )
            )
            model.add(Dropout(params["dropout2"]))
            model.add(BatchNormalization())
            model.add(Dense(units=1, kernel_initializer="normal", kernel_constraint=NonNeg(),
                            bias_constraint=NonNeg()))

            adam = Adam(learning_rate=params["lr"])
            model.compile(loss="mse", metrics=[
                          "mean_squared_error"], optimizer=adam)

            history = model.fit(
                X_train,
                y_train,
                epochs=int(params["epochs"]),
                validation_data=(X_test, y_test),
                batch_size=int(params["batch_size"]),
                verbose=1,
            )
            return model
        model = dnn_returner(best)
        print(model)

        # pdb.set_trace()
        messageCode = 200
        message = "successfully merged the data"
        # print("sumit")
        elapsedTime = time.time() - startTime

    except Exception as e:
        logging.error("Error: dnnAutoTuner()")
        logging.error({"Error": e})
        messageCode = 400
        message = e
        finalDf = {}
        elapsedTime = {}

    return {"model": model, "best": best}
