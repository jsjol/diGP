# -*- coding: utf-8 -*-

#    Copyright 2017 Jens Sjölund

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import json
import numpy as np
import dipy.reconst.mapmri as mapmri

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from diGP.preprocessing_pipelines import get_SPARC_train_and_test
from diGP.evaluation import get_SPARC_metrics


def map_mri_for_SPARC():
    with open('../config.json', 'r') as json_file:
        conf = json.load(json_file)
    data_paths = conf['SPARC']['data_paths']
    q_test_path = conf['SPARC']['q_test_path']

    for source in ['gradient_20', 'gradient_30', 'gradient_60']:
        gtab, data, voxelSize = get_SPARC_train_and_test(
                                    data_paths[source],
                                    data_paths['goldstandard'],
                                    q_test_path)

        print('\nMaking predictions for {}.'.format(source))
        fitted_data, pred = make_predictions(gtab, data)

        get_SPARC_metrics(gtab['test'], data['test'], pred, verbose=True)

        print('\nSaving predictions in {}'.format(data_paths[source]))
        np.save(os.path.join(data_paths[source], 'map_mri_train'), fitted_data)
        np.save(os.path.join(data_paths[source], 'map_mri_test'), pred)


def make_predictions(gtab, data):
    map_model = mapmri.MapmriModel(gtab['train'], positivity_constraint=True,
                                   laplacian_weighting='GCV',
                                   radial_order=8, anisotropic_scaling=True)
    mapfit = map_model.fit(data['train'])

    fitted_data = mapfit.fitted_signal()
    pred = mapfit.predict(gtab['test'])
    return fitted_data, pred


def main():
    map_mri_for_SPARC()

if __name__ == '__main__':
    main()
