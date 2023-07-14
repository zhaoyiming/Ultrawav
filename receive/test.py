import numpy as np

import scipy.io as scio



data = scio.loadmat("1.mat")
print(type(data['IQ']))