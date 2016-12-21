# -*- coding: utf-8 -*-


import unittest
import numpy as np
import numpy.testing as npt
from dipy.core.gradients import gradient_table
from generateSyntheticData import (generateSyntheticInputs,
                                   generateSyntheticOutputsFromMultiTensorModel)

class integration_test_generateSyntheticData(unittest.TestCase):
    
    def test_dataGeneration(self):
        voxelsInEachDim = (2, 3, 4)
        
        bvals=1500*np.ones(7)
        bvals[0]=0
        sq2=np.sqrt(2)/2
        bvecs=np.array([[0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [sq2, sq2, 0],
                        [sq2, 0, sq2],
                        [0, sq2, sq2]])
        smallDelta = 12.9
        bigDelta = 21.8
        gtab = gradient_table(bvals, bvecs, big_delta=bigDelta,
                              small_delta=smallDelta)
        
        # Diffusion coefficients for white matter tracts, in mm^2/s
        #
        # Based roughly on values from:
        #
        #   Pierpaoli, Basser, "Towards a Quantitative Assessment of Diffusion
        #   Anisotropy", Magnetic Resonance in Medicine, 1996; 36(6):893-906.
        #
        whiteMatterDiffusionEigenvalues = np.array([1500e-6, 400e-6, 400e-6])
        tensorEigenvalues = np.tile(whiteMatterDiffusionEigenvalues, (2,1))
        
        inputs = generateSyntheticInputs(voxelsInEachDim, gtab)
        outputs = generateSyntheticOutputsFromMultiTensorModel(
            voxelsInEachDim, gtab, tensorEigenvalues)
        
        totalNumberOfSamples = np.prod(voxelsInEachDim)*len(bvals)
        npt.assert_array_equal(inputs.shape, 
                               (totalNumberOfSamples, 7),
                               'Input shapes don\'t match')
        npt.assert_array_equal(outputs.shape, (totalNumberOfSamples,),
                               'Output shapes don\'t match')

        # add spatial correlations - easiest to mock an exact signal in each 
        # voxel with known correlation.


def main():
    unittest.main()

if __name__ == '__main__':
    main()