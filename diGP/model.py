# -*- coding: utf-8 -*-


import numpy as np
import scipy.stats
from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.data import get_data
from dipy.core.gradients import gradient_table
from dipy.reconst.dsi import (DiffusionSpectrumModel,
                              DiffusionSpectrumDeconvModel)
from dipy.denoise.noise_estimate import piesno
import GPy
from diGP.dataManipulations import DataHandler


class GaussianProcessModel(ReconstModel):

    def __init__(self, gtab, spatial_dims=3, kernel=None, grid_dims=None,
                 box_cox_lambda=None, q_magnitude_transform=None,
                 verbose=False):
        self.gtab = gtab
        self.spatial_dims = spatial_dims
        self.box_cox_lambda = box_cox_lambda
        self.q_magnitude_transform = q_magnitude_transform
        self.verbose = verbose

        if kernel is None:
            self.kernel = get_default_kernel(spatial_dims=spatial_dims)
        else:
            self.kernel = kernel

        if grid_dims is None:
            self.grid_dims = get_default_grid_dims(spatial_dims)
        else:
            self.grid_dims = grid_dims

    def fit(self, data, **kwargs):
        return GaussianProcessFit(self, data, **kwargs)


class GaussianProcessFit(ReconstFit):

    def __init__(self, model, data, mean=None, voxel_size=None,
                 image_origin=None, spatial_idx=None, retrain=True,
                 restarts=False, noise_variance=1.):
        self.model = model
        self.voxel_size = voxel_size
        self.spatial_shape = data.shape[:-1]
        self.data_handler = DataHandler(
                        self.model.gtab,
                        data=data,
                        mean=mean,
                        voxelSize=self.voxel_size,
                        image_origin=image_origin,
                        spatialIdx=spatial_idx,
                        box_cox_lambda=self.model.box_cox_lambda,
                        qMagnitudeTransform=self.model.q_magnitude_transform)

        self.GP_model = GPy.models.GPRegressionGrid(
                                            self.data_handler.X,
                                            self.data_handler.y,
                                            self.model.kernel,
                                            grid_dims=self.model.grid_dims)
        self.GP_model.Gaussian_noise.variance = noise_variance

        if retrain:
            self.train(restarts=restarts)

    def train(self, restarts=False, robust=True, **kwargs):
        if restarts:
            self.GP_model.optimize_restarts(messages=self.model.verbose,
                                            robust=robust, **kwargs)
        else:
            self.GP_model.optimize(messages=self.model.verbose, **kwargs)

        if self.model.verbose:
            print(self.GP_model)

    def predict(self, gtab_pred, mean=None, voxel_size=None, image_origin=None,
                spatial_idx=None, spatial_shape=None, compute_var=False):

        if voxel_size is None:
            voxel_size = self.voxel_size

        if spatial_shape is None:
            spatial_shape = self.spatial_shape

        data_handler_pred = DataHandler(
                        gtab_pred,
                        data=None,
                        mean=mean,
                        voxelSize=voxel_size,
                        image_origin=image_origin,
                        spatialIdx=spatial_idx,
                        spatial_shape=spatial_shape,
                        box_cox_lambda=self.model.box_cox_lambda,
                        qMagnitudeTransform=self.model.q_magnitude_transform)

        if self.model.verbose:
            print("Prediction started.")

        if compute_var:
            mu, var = self.GP_model.predict_noiseless(
                        data_handler_pred.X, compute_var=compute_var)
            return [data_handler_pred.untransform(mu),
                    data_handler_pred.untransform(var)]
        else:
            mu = self.GP_model.predict_noiseless(
                        data_handler_pred.X, compute_var=compute_var)
            return data_handler_pred.untransform(mu)

    def odf(self, sphere, gtab_dsi=None, mean=None, voxel_size=None,
            image_origin=None, spatial_idx=None, spatial_shape=None):

        if gtab_dsi is None:
            btable = np.loadtxt(get_data('dsi4169btable'))
            gtab_dsi = gradient_table(btable[:, 0], btable[:, 1:],
                                      big_delta=self.model.gtab.big_delta,
                                      small_delta=self.model.gtab.small_delta)

        pred = self.predict(gtab_dsi,
                            mean=mean,
                            voxel_size=voxel_size,
                            image_origin=image_origin,
                            spatial_idx=spatial_idx,
                            spatial_shape=spatial_shape,
                            compute_var=False)

        dsi_model = DiffusionSpectrumModel(gtab_dsi, qgrid_size=25,
                                           r_end=50, r_step=0.4,
                                           filter_width=np.inf)
#        dsi_model = DiffusionSpectrumModel(gtab_dsi, filter_width=np.inf)
#        dsi_model = DiffusionSpectrumDeconvModel(gtab_dsi)

        odf = dsi_model.fit(pred).odf(sphere)
        return odf

    def optimize_box_cox_lambda(self):
        _, lmbda = scipy.stats.boxcox(self.data_handler.data.flatten(),
                                      lmbda=None)
        self.model.box_cox_lambda = lmbda
        self.data_handler.box_cox_lambda = lmbda


def get_default_grid_dims(spatial_dims):
    if spatial_dims == 2:
        return [[0], [1], [2, 3, 4, 5]]
    elif spatial_dims == 3:
        return [[0], [1], [2], [3, 4, 5, 6]]
    else:
        raise ValueError("Only 2 or 3 spatial dimensions are supported, {} \
                         provided".format(spatial_dims))


def get_default_kernel(spatial_dims, n_max=6, spatial_lengthscale=None,
                       q_lengthscale=1, coefficients=None):

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


def get_default_independent_kernel(spatial_dims=3, n_max=6,
                                   q_lengthscale=1, coefficients=None):

    orders = np.arange(0, n_max + 1, 2)
    if coefficients is None:
        coefficients = 10 ** (-(1 + np.arange(0, n_max/2 + 1, 1)))

    if spatial_dims == 2:
        kernel = (GPy.kern.White(input_dim=1, active_dims=[0]) *
                  GPy.kern.White(input_dim=1, active_dims=[1]) *
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
        kernel = (GPy.kern.White(input_dim=1, active_dims=[0]) *
                  GPy.kern.White(input_dim=1, active_dims=[1]) *
                  GPy.kern.White(input_dim=1, active_dims=[2]) *
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
