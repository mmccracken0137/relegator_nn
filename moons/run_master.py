#!/usr/bon/env python
import os

for i in range(20):
    cmd = 'python master_moons.py regress 10000 0.25 0.9 500 0.05 write_results noplot'
    print(cmd)
    os.system(cmd)
