#!/usr/bon/env python

import sys
import os
import numpy as np

for i in range(20):
    cmd = 'python master_moons.py relegator 1000 0.25 0.9 200 0.05 write_results noplot'
    print(cmd)
    os.system(cmd)
