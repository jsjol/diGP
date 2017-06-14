# -*- coding: utf-8 -*-


import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import GPy
from dipy.core.gradients import gradient_table
from diGP.dataManipulations import (generateCoordinates,
                                    combineCoordinatesAndqVecs)
from diGP.generateSyntheticData import (
        generatebValsAndbVecs,
        generateSyntheticInputs,
        generateSyntheticOutputsFromMultiTensorModel)


class integration_test_generateSyntheticData(unittest.TestCase):

    def setUp(self):
        # Diffusion coefficients for white matter tracts, in mm^2/s
        #
        # Based roughly on values from:
        #
        #   Pierpaoli, Basser, "Towards a Quantitative Assessment of Diffusion
        #   Anisotropy", Magnetic Resonance in Medicine, 1996; 36(6):893-906.
        #
        whiteMatterDiffusionEigenvalues = np.array([1500e-6, 400e-6, 400e-6])
        self.tensorEigenvalues = np.tile(whiteMatterDiffusionEigenvalues,
                                         (2, 1))
        self.smallDelta = 12.9
        self.bigDelta = 21.8
        self.verbose = False
        np.random.seed(0)

    def test_dataGeneration(self):
        voxelsInEachDim = (2, 3, 4)
        bvals = np.concatenate([[0], 1500*np.ones(6), 3000*np.ones(6)])
        sq2 = np.sqrt(2) / 2
        setOfbvecs = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1],
                               [sq2, sq2, 0],
                               [sq2, 0, sq2],
                               [0, sq2, sq2]])
        bvecs = np.vstack([np.array([[0, 0, 0]]),
                          setOfbvecs,
                          setOfbvecs])
        gtab = gradient_table(bvals, bvecs, big_delta=self.bigDelta,
                              small_delta=self.smallDelta)

        inputs = generateSyntheticInputs(voxelsInEachDim, gtab)
        outputsLatent = generateSyntheticOutputsFromMultiTensorModel(
            voxelsInEachDim, gtab, self.tensorEigenvalues, snr=None)
        self.assertTrue(np.min(outputsLatent) >= 0 and
                        np.max(outputsLatent) <= 1)

        totalNumberOfSamples = np.prod(voxelsInEachDim)*len(bvals)
        npt.assert_array_equal(inputs.shape,
                               (totalNumberOfSamples, 7),
                               'Input shapes don\'t match')
        npt.assert_array_equal(outputsLatent.shape, (totalNumberOfSamples, 1),
                               'Output shapes don\'t match')

    def test_singleVoxelPrediction(self):
        voxelsInEachDim = (1, 1, 1)
        uniquebVals = np.array([0, 1000, 3000, 5000, 10000])
        uniquebValsValidation = np.array([500, 2000, 4000, 7500])

        numbVecs = np.array([10, 64, 64, 128, 256])
        numbVecsValidation = np.array([100, 200, 200, 200])

        inputsTrain, outputsLatent = self.makeInputsAndOutputs(
            voxelsInEachDim, uniquebVals, numbVecs)
        inputsValidation, outputsValidation = self.makeInputsAndOutputs(
            voxelsInEachDim, uniquebValsValidation, numbVecsValidation)

        noiseStd = 0.05
        noise = np.random.normal(0, noiseStd, size=outputsLatent.shape)
        outputsTrain = outputsLatent + noise

        kernel = _composeKernel()
        model = GPy.models.GPRegression(inputsTrain, outputsTrain, kernel)
        model.optimize()

        predictionTrain = model.predict(inputsTrain,
                                        include_likelihood=False)[0]
        predictionValidation, variancePrediction = model.predict(
            inputsValidation, include_likelihood=False)

        errorsTrain = outputsLatent - predictionTrain
        errorsValidation = outputsValidation - predictionValidation

        rmsErrorTrain = _rootMeanSquareError(predictionTrain, outputsLatent)
        rmsErrorValidation = _rootMeanSquareError(predictionValidation,
                                                  outputsValidation)

        estimatedNoiseStd = np.sqrt(model.Gaussian_noise.variance)
        self.assertTrue(np.isclose(estimatedNoiseStd, noiseStd, rtol=0.1))

        # Check that the predictions are within the 95 % confidence interval,
        # according to a Z-test
        muTrain = np.mean(errorsTrain)
        muValidation = np.mean(errorsValidation)
        n = np.sum(numbVecs)
        standardError = noiseStd/np.sqrt(n)
        zTrain = (muTrain - 0.)/standardError
        zValidation = (muValidation - 0.)/standardError
        self.assertTrue(np.abs(zTrain) < 1.96)
        self.assertTrue(np.abs(zValidation) < 1.96)

        self.assertTrue(rmsErrorTrain < noiseStd/2)  # Ad hoc level
        self.assertTrue(rmsErrorValidation < noiseStd)  # Ad hoc level

        if self.verbose:
            print('\nOptimized model for single voxel case')
            print(model)

            plt.figure(1)
            plt.hist(errorsValidation, bins=10)
            plt.xlabel('Validation error')

            plt.figure(2)
            plt.scatter(outputsValidation, predictionValidation)
            plt.xlim([0., 1.])
            plt.ylim([0., 1.])
            plt.xlabel('True output')
            plt.ylabel('Predicted output')
            plt.show()

            print('\nRMS error on training data: {}'
                  .format(rmsErrorTrain))
            print('RMS error on validation data: {}'
                  .format(rmsErrorValidation))

    def test_spatialCorrelation(self):
        n = 500
        lengthScale = 2.
        scaling = 0.5
        noiseVariance = 0.01
        inputs, outputs = self.generateSpatiallyCorrelatedData(
            n=n, lengthScale=lengthScale, scaling=scaling,
            noiseVariance=noiseVariance)

        kernel = _composeKernel()
        model = GPy.models.GPRegression(inputs, outputs, kernel)

        model.optimize()

        estimatedScaling = np.sqrt(model.mul.rbf.variance)
        estimatedLengthScale = model.mul.rbf.lengthscale
        estimatedNoiseVariance = model.Gaussian_noise.variance / scaling**2
        self.assertTrue(np.isclose(estimatedScaling,
                                   scaling, rtol=0.1))
        self.assertTrue(np.isclose(estimatedLengthScale,
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedNoiseVariance,
                                   noiseVariance, rtol=0.1))

        if self.verbose:
            print('\nOptimized model for spatial correlation case')
            print(model)

    def test_factorizedSpatialCorrelation(self):
        lengthScale = 2.
        scaling = 0.5
        noiseVariance = 0.01

        voxelsInEachDim = (3, 6, 25)

        bVals = np.array([0])
        bVecs = np.array([[1, 0, 0]])
        n = np.prod(voxelsInEachDim)*len(bVals)
        gtab = gradient_table(bVals, bVecs, big_delta=self.bigDelta,
                              small_delta=self.smallDelta)
        qMagnitudes = gtab.qvals[:, np.newaxis]
        qFeatures = np.column_stack((qMagnitudes, gtab.bvecs))

        coordinates = generateCoordinates(voxelsInEachDim)
        trueSpatialKernel = GPy.kern.RBF(input_dim=3, variance=scaling**2,
                                         lengthscale=lengthScale)
        latentCovariance = (trueSpatialKernel.K(coordinates) +
                            noiseVariance * np.eye(n))
        outputs = np.random.multivariate_normal(
            np.zeros(n,), latentCovariance)

        outputs = outputs[:, np.newaxis]

        spatialKernel = GPy.kern.RBF(input_dim=3, variance=1., lengthscale=1.)

        qKernel = _qKernel()

        model = GPy.models.GPKroneckerGaussianRegression(
            qFeatures, coordinates, outputs.T, qKernel, spatialKernel)
        model.optimize()

        estimatedScaling = np.sqrt(model.rbf.variance)
        estimatedLengthScale = model.rbf.lengthscale
        estimatedNoiseVariance = model.Gaussian_noise.variance
        self.assertTrue(np.isclose(estimatedScaling,
                                   scaling, rtol=0.1))
        self.assertTrue(np.isclose(estimatedLengthScale,
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedNoiseVariance,
                                   noiseVariance, rtol=0.1))
        if self.verbose:
            print('\nOptimized model for factorized spatial correlation case')
            print(model)

    def test_fullFactorizedModel(self):
        lengthScale = 2.
        scaling = 0.5
        noiseVariance = 0.01

        bValLengthScale = 2

        voxelsInEachDim = (5, 5, 5)

        uniquebVals = np.array([0, 1000, 3000, 5000])
        numbVecs = np.array([1, 5, 5, 5])
        bVals, bVecs = generatebValsAndbVecs(uniquebVals, numbVecs)

        n = np.prod(voxelsInEachDim)*len(bVals)
        gtab = gradient_table(bVals, bVecs, big_delta=self.bigDelta,
                              small_delta=self.smallDelta)

        qMagnitudes = gtab.qvals[:, np.newaxis]
        qMagnitudes = self.qMagnitudeTransform(qMagnitudes)
        qFeatures = np.column_stack((qMagnitudes, gtab.bvecs))

        coordinates = generateCoordinates(voxelsInEachDim)
        inputs = combineCoordinatesAndqVecs(coordinates, qFeatures)

        trueKernel = _composeKernel(scaling=scaling,
                                    spatialLengthScale=lengthScale,
                                    bValLengthScale=bValLengthScale)
        latentCovariance = (trueKernel.K(inputs) +
                            noiseVariance * np.eye(n))
        outputs = np.random.multivariate_normal(
            np.zeros(n,), latentCovariance)

        kernel = _composeKernel()
        grid_dims = [[0], [1], [2], [3, 4, 5, 6]]
        outputs = outputs[:, np.newaxis]
        model = GPy.models.GPRegressionGrid(inputs, outputs, kernel,
                                            grid_dims=grid_dims)        

        model.optimize(messages=self.verbose)

        estimatedScaling = np.sqrt(model.kern.parts[0].variance)
        estimatedLengthScale = [model.mul.rbf.lengthscale,
                                model.mul.rbf_1.lengthscale,
                                model.mul.rbf_2.lengthscale]
        estimatedNoiseVariance = model.Gaussian_noise.variance
        estimatedbValLengthScale = model.mul.rbf_3.lengthscale
        self.assertTrue(np.isclose(estimatedScaling,
                                   scaling, rtol=0.1))
        self.assertTrue(np.isclose(estimatedLengthScale[0],
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedLengthScale[1],
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedLengthScale[2],
                                   lengthScale, rtol=0.1))
        self.assertTrue(np.isclose(estimatedNoiseVariance,
                                   noiseVariance, rtol=0.1))
        self.assertTrue(np.isclose(estimatedbValLengthScale,
                                   bValLengthScale, rtol=0.1))

        if self.verbose:
            print('\nOptimized model for full factorized case')
            print(model)

    def qMagnitudeTransform(self, x):
        return np.log(np.e + x ** 2)

    def makeInputsAndOutputs(self, voxelsInEachDim, uniquebVals, numbVecs):
        bVals, bVecs = generatebValsAndbVecs(uniquebVals, numbVecs)

        gtab = gradient_table(bVals, bVecs, big_delta=self.bigDelta,
                              small_delta=self.smallDelta)

        inputs = generateSyntheticInputs(voxelsInEachDim, gtab,
            qMagnitudeTransform=self.qMagnitudeTransform)

        outputs = generateSyntheticOutputsFromMultiTensorModel(
            voxelsInEachDim, gtab, self.tensorEigenvalues, snr=None)
        return inputs, outputs

    def generateSpatiallyCorrelatedData(self, n=100, lengthScale=1.,
                                        scaling=1., noiseVariance=0.01):
        voxelsInEachDim = (n, 1, 1)
        bVals = np.array([0])
        bVecs = np.array([[1, 0, 0]])
        gtab = gradient_table(bVals, bVecs, big_delta=self.bigDelta,
                              small_delta=self.smallDelta)
        inputs = generateSyntheticInputs(voxelsInEachDim, gtab)
        xCoordinate = inputs[:, 0]
        distances = np.abs(xCoordinate[:, np.newaxis] -
                           xCoordinate[np.newaxis, :])
        spatialCovariance = (np.exp(-scaling * (distances/lengthScale) ** 2) +
                             noiseVariance * np.eye(n))
        outputs = scaling * np.random.multivariate_normal(np.zeros(n,),
                                                          spatialCovariance)
        outputs = outputs[:, np.newaxis]
        return inputs, outputs


def _composeKernel(scaling=1., spatialLengthScale=1., bValLengthScale=1.):
    combinedKernel = (GPy.kern.RBF(input_dim=1, active_dims=[0],
                                   variance=scaling**2,
                                   lengthscale=spatialLengthScale) *
                      GPy.kern.RBF(input_dim=1, active_dims=[1],
                                   variance=1,
                                   lengthscale=spatialLengthScale) *
                      GPy.kern.RBF(input_dim=1, active_dims=[2],
                                   variance=1,
                                   lengthscale=spatialLengthScale) *
                      GPy.kern.RBF(input_dim=1, active_dims=[3],
                                   variance=1,
                                   lengthscale=bValLengthScale) *
                      GPy.kern.LegendrePolynomial(
                         input_dim=3,
                         coefficients=np.array((1/3, 2/3)),
                         orders=(0, 2),
                         active_dims=(4, 5, 6)))
    combinedKernel.parts[1].variance.fix(value=1)
    combinedKernel.parts[2].variance.fix(value=1)
    combinedKernel.parts[3].variance.fix(value=1)
    combinedKernel.parts[4].coefficients.fix(value=(1/3, 2/3))
    return combinedKernel


def _qKernel(bvalLengthScale=1.):
    bvalKernel = GPy.kern.RBF(input_dim=1, active_dims=[0],
                              lengthscale=bvalLengthScale)
    bvalKernel.variance.fix(value=1.)

    bvecKernel = GPy.kern.Poly(input_dim=3, active_dims=(1, 2, 3), order=2)
    bvecKernel.variance.fix(value=1.)
    bvecKernel.scale.fix(value=1.)
    bvecKernel.bias.fix(value=0.)

    return bvalKernel*bvecKernel


def _rootMeanSquareError(predicted, target):
    return np.sqrt(np.mean((predicted - target)**2))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
