# -*- coding: utf-8 -*-


import warnings
import numpy as np
import scipy.stats
from dipy.denoise.noise_estimate import piesno
import GPy


class Model():

    def __init__(self, data_handler, kernel, data_handler_pred=None,
                 grid_dims=None, verbose=False):
        self.data_handler = data_handler
        self.data_handler_pred = data_handler_pred
        self.GP_model = GPy.models.GPRegressionGrid(self.data_handler.X,
                                                    self.data_handler.y,
                                                    kernel,
                                                    grid_dims=grid_dims)
        self.verbose = verbose

    def train(self, restarts=True):
        if restarts:
            self.GP_model.optimize_restarts(messages=self.verbose)
        else:
            self.GP_model.optimize(messages=self.verbose)

        if self.verbose:
            print(self.GP_model)

    def predict(self, compute_var=False):
        if self.data_handler_pred is None:
            raise ValueError("Data for prediction not defined.")

        self._sync_box_cox_lambdas()

        if self.verbose:
            print("Prediction started.")

        return self.GP_model.predict_noiseless(self.data_handler_pred.X,
                                               compute_var=compute_var)

    def optimize_box_cox_lambda(self):
        _, lmbda = scipy.stats.boxcox(self.data_handler.data.flatten(),
                                      lmbda=None)
        self._update_box_cox_lambda(lmbda)

    def _update_box_cox_lambda(self, lmbda):
        self.data_handler.box_cox_lambda = lmbda
        if self.data_handler_pred is not None:
            self.data_handler_pred.box_cox_lambda = lmbda

    def _sync_box_cox_lambdas(self):
        if not (self.data_handler.box_cox_lambda ==
                self.data_handler_pred.box_cox_lambda):
            warnings.warn("Box-Cox lambdas are not equal, resetting to {}."
                          .format(self.data_handler.box_cox_lambda))
            self._update_box_cox_lambda(self.data_handler.box_cox_lambda)


def estimateBoxCoxLambdaFromBackground(data):
    idx = getBackgroundIdxUsingPIESNO(data)
    x = data[idx[0], idx[1], idx[2], :].flatten()

    # Box-Cox requires positive values
    x = x[x > 0]

    _, lmbda = scipy.stats.boxcox(x)
    return lmbda


def getBackgroundIdxUsingPIESNO(data):
    _, mask = piesno(data, N=1, return_mask=True)
    return np.nonzero(mask)
