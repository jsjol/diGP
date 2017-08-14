# -*- coding: utf-8 -*-


import numpy as np
import unittest
import unittest.mock as mock
import numpy.testing as npt
from diGP.model import GaussianProcessModel


class TestBoxCox(unittest.TestCase):

    def setUp(self):
        self.gtab = mock.MagicMock()

        data = 0.05*np.ones((2, 2, 1, 3))
        data[1, 1, 0, 0] += 1
        data[1, 1, 0, 1] += 0.6
        data[1, 1, 0, 2] += 0.3
        self.data = data

    @mock.patch('scipy.stats.boxcox')
    @mock.patch('GPy.models.GPRegressionGrid')
    def test_optimize_box_cox_lambda(self, GPGrid, mock_boxcox):
        lmbda = 3
        mock_boxcox.return_value = [np.array([]), lmbda]

        m = GaussianProcessModel(self.gtab)
        fit = m.fit(self.data)
        fit.optimize_box_cox_lambda()

        npt.assert_array_equal(np.array(mock_boxcox.call_args[0]).flatten(),
                               self.data.flatten())
        npt.assert_(mock_boxcox.call_args[1]['lmbda'] is None)
        npt.assert_equal(m.box_cox_lambda, lmbda)
        npt.assert_equal(fit.data_handler.box_cox_lambda, lmbda)
