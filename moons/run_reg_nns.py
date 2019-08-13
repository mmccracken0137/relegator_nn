#!/usr/bon/env python

import sys
import os
import numpy as np

frac_vals = np.linspace(0.05, 0.95, num=19)

for f in frac_vals:
    cmd = 'python nn_moons.py 10000 0.25 0.0 100 ' + str(np.round(f, 4)) + ' noplot'
    print(cmd)
    os.system(cmd)
