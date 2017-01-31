# -*- coding: utf-8 -*-


import unittest
import numpy as np
import numpy.testing as npt
import GPy
from generateSyntheticData import generatebValsAndbVecs, generateCoordinates


class integration_test_LegendrePolynomial(unittest.TestCase):

    def setUp(self):
        self.smallDelta = 12.9
        self.bigDelta = 21.8
        self.verbose = True
        np.random.seed(0)

    def test_LegendrePolynomial(self):

        # Latent parameters (to be estimated)
        lengthScale = 2.
        noiseVariance = 0.01
        coefficients = (1, 0.5, 0.1)

        # Experimental setup
        voxelsInEachDim = (1, 2, 25)
        uniquebVals = np.array([1000])
        numbVecs = np.array([30])
        legendreOrders = (0, 2, 4)

        # Make inputs (skip qMagnitude feature for simplicity)
        _, bVecs = generatebValsAndbVecs(uniquebVals, numbVecs)
        coordinates = generateCoordinates(voxelsInEachDim)

        # Generate combined covariance as Kronecker product
        # between spatial kernel and hard coded Legendre covariance
        spatialKernel = GPy.kern.RBF(input_dim=3, variance=1.,
                                     lengthscale=lengthScale)
        spatialK = spatialKernel.K(coordinates)
        bvecK = _KhardCodedLegendrePolynomials(coefficients, bVecs)
        K = np.kron(spatialK, bvecK)
        latentCovariance = K + noiseVariance * np.eye(K.shape[0])

        # Sample outputs with these covariance matrix
        outputs = np.random.multivariate_normal(
            np.zeros(latentCovariance.shape[0],), latentCovariance)

        outputs = np.reshape(outputs,
                             (bVecs.shape[0], coordinates.shape[0]),
                             order='F')

        # Specify model
        spatialKernel = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=1.)
        spatialKernel.variance.fix(value=1.)
        bvecKernel = GPy.kern.LegendrePolynomial(
            input_dim=3, coefficients=(1., 1., 1.), orders=legendreOrders)

        model = GPy.models.GPKroneckerGaussianRegression(
            bVecs, coordinates, outputs, bvecKernel, spatialKernel)
        model.optimize()

        estimatedLengthScale = model.rbf.lengthscale
        estimatedNoiseVariance = model.Gaussian_noise.variance
        self.assertTrue(np.isclose(estimatedLengthScale,
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedNoiseVariance,
                                   noiseVariance, rtol=0.1))
        npt.assert_allclose(model.LegendrePolynomial.coefficients,
                            coefficients, rtol=0.2, atol=0.05)

        if self.verbose:
            print('\nOptimized model with spatial RBF and\
                  Legendre polynomial kernel for bvecs')
            print(model)
            print(model.LegendrePolynomial.coefficients)


def _KhardCodedLegendrePolynomials(c, X):
    def P_0(y): return 1

    def P_2(y): return 0.5*(3*y**2-1)

    def P_4(y): return 1/8*(35*y**4 - 30*y**2 + 3)

    dot_prod = np.dot(X, X.T)
    K = c[0]*P_0(dot_prod) + c[1]*P_2(dot_prod) + c[2]*P_4(dot_prod)
    return K


def main():
    unittest.main()

if __name__ == '__main__':
    main()
