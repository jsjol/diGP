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

    def train(self, restarts=False, **kwargs):
        if restarts:
            self.GP_model.optimize_restarts(messages=self.verbose,
                                            robust=True, **kwargs)
        else:
            self.GP_model.optimize(messages=self.verbose, **kwargs)

        if self.verbose:
            print(self.GP_model)

    def predict(self, mean=None, compute_var=False):
        if self.data_handler_pred is None:
            raise ValueError("Data for prediction not defined.")

        self._sync_box_cox_lambdas()

        if self.verbose:
            print("Prediction started.")

        if compute_var:
            mu, var = self.GP_model.predict_noiseless(
                        self.data_handler_pred.X, compute_var=compute_var)
            return [self.data_handler_pred.untransform(mu),
                    self.data_handler_pred.untransform(var)]
        else:
            mu = self.GP_model.predict_noiseless(
                        self.data_handler_pred.X, compute_var=compute_var)
            return self.data_handler_pred.untransform(mu)

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


def get_default_kernel(data_handler, n_max=6, spatial_lengthscale=None,
                       q_lengthscale=1, coefficients=None):

    spatial_dims = len(data_handler.originalShape) - 1
    if spatial_lengthscale is None:
        spatial_lengthscale = 2 * np.ones(spatial_dims)

    orders = np.arange(0, n_max + 1, 2)
    if coefficients is None:
        coefficients = 10 ** (-(1 + np.arange(0, n_max/2 + 1, 1)))

    if spatial_dims == 2:
        kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                               lengthscale=spatial_lengthscale[0]) *
                  GPy.kern.RBF(input_dim=1, active_dims=[1],
                               lengthscale=spatial_lengthscale[1]) *
                  GPy.kern.RBF(input_dim=1, active_dims=[2],
                               lengthscale=q_lengthscale) *
                  GPy.kern.LegendrePolynomial(
                     input_dim=3,
                     coefficients=coefficients,
                     orders=orders,
                     active_dims=(3, 4, 5)))

        kernel.parts[0].variance.fix(value=1)
        kernel.parts[1].variance.fix(value=1)
        kernel.parts[2].variance.fix(value=1)

        return kernel
    elif spatial_dims == 3:
        kernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                               lengthscale=spatial_lengthscale[0]) *
                  GPy.kern.RBF(input_dim=1, active_dims=[1],
                               lengthscale=spatial_lengthscale[1]) *
                  GPy.kern.RBF(input_dim=1, active_dims=[2],
                               lengthscale=spatial_lengthscale[2]) *
                  GPy.kern.RBF(input_dim=1, active_dims=[3],
                               lengthscale=q_lengthscale) *
                  GPy.kern.LegendrePolynomial(
                     input_dim=3,
                     coefficients=coefficients,
                     orders=orders,
                     active_dims=(4, 5, 6)))

        kernel.parts[0].variance.fix(value=1)
        kernel.parts[1].variance.fix(value=1)
        kernel.parts[2].variance.fix(value=1)
        kernel.parts[3].variance.fix(value=1)

        return kernel
    else:
        raise ValueError("Only 2 or 3 spatial dimensions are supported, {} \
                         provided".format(spatial_dims))


def get_default_independent_kernel(data_handler, n_max=6,
                                   q_lengthscale=1, coefficients=None):

    spatial_dims = len(data_handler.originalShape) - 1

    orders = np.arange(0, n_max + 1, 2)
    if coefficients is None:
        coefficients = 10 ** (-(1 + np.arange(0, n_max/2 + 1, 1)))

    if spatial_dims == 2:
        kernel = (GPy.kern.Fixed(input_dim=1, active_dims=[0],
                    covariance_matrix=np.eye(data_handler.originalShape[0])) *
                  GPy.kern.Fixed(input_dim=1, active_dims=[1],
                    covariance_matrix=np.eye(data_handler.originalShape[1])) *
                  GPy.kern.RBF(input_dim=1, active_dims=[2],
                               lengthscale=q_lengthscale) *
                  GPy.kern.LegendrePolynomial(
                     input_dim=3,
                     coefficients=coefficients,
                     orders=orders,
                     active_dims=(3, 4, 5)))

        kernel.parts[0].variance.fix(value=1)
        kernel.parts[1].variance.fix(value=1)
        kernel.parts[2].variance.fix(value=1)

        return kernel
    elif spatial_dims == 3:
        kernel = (GPy.kern.Fixed(input_dim=1, active_dims=[0],
                    covariance_matrix=np.eye(data_handler.originalShape[0])) *
                  GPy.kern.Fixed(input_dim=1, active_dims=[1],
                    covariance_matrix=np.eye(data_handler.originalShape[1])) *
                  GPy.kern.Fixed(input_dim=1, active_dims=[1],
                    covariance_matrix=np.eye(data_handler.originalShape[2])) *
                  GPy.kern.RBF(input_dim=1, active_dims=[3],
                               lengthscale=q_lengthscale) *
                  GPy.kern.LegendrePolynomial(
                     input_dim=3,
                     coefficients=coefficients,
                     orders=orders,
                     active_dims=(4, 5, 6)))

        kernel.parts[0].variance.fix(value=1)
        kernel.parts[1].variance.fix(value=1)
        kernel.parts[2].variance.fix(value=1)
        kernel.parts[3].variance.fix(value=1)

        return kernel
    else:
        raise ValueError("Only 2 or 3 spatial dimensions are supported, {} \
                         provided".format(spatial_dims))


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
