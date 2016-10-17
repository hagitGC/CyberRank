#*************************************************************************************
#
#                   CREATE SYNTHETIC DATA
#
# We generate pairs from a list of examples generated form sampling possible values
# for each feature and calculating the features list rank using the AHP weights.
#*************************************************************************************
import itertools
import numpy as np
import pandas as pd


AHPwights= [0.030420825, 0.133504122, 0.01506668, 0.103644788, 0.106375118, 0.032145798, 0.059287755, 0.061884989, 0.201151162, 0.190558878, 0.030574248, 0.035385638]

synth_file_name = r"recordsForPython2504numeric.csv"
synth_data = pd.read_csv(synth_file_name)
a = synth_data.index.get_values()
names = list(synth_data.keys()[0:-1])
synthX = pd.read_csv(synth_file_name, usecols =names)
synth_y = pd.read_csv(synth_file_name, usecols =['sens_rank'])
synthX= synthX.get_values().__array__()
synth_y = np.dot(synthX, AHPwights)

##crate pairs
comb = itertools.combinations(range(synthX.shape[0]), 2)
k = 0
synthXp, synth_yp, synth_diff = [], [], []
for (i, j) in comb:
    print i, j
    print synth_y[i], synth_y[j]
    if synth_y[i] == synth_y[j]:
        # skip if same target or different group
        continue
    synthXp.append(synthX[i] - synthX[j])
    synth_diff.append(synth_y[i] - synth_y[j])
    synth_yp.append(np.sign(synth_diff[-1]))
    # output balanced classes
    if synth_yp[-1] != (-1) ** k:
        synth_yp[-1] *= -1
        synthXp[-1] *= -1
        synth_diff[-1] *= -1
    k += 1


print synthXp.__len__()
synthXp, synth_yp, synth_diff = map(np.asanyarray, (synthXp, synth_yp, synth_diff))




