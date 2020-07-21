#!/usr/bin/env python
'''

'''

import os
import sys
import itertools

n_fits = 4 # 0
n_datasets = 4 # 0

angle_vals = [0.0, 0.6, 1.7]
sig_frac_vals = [0.001, 0.00316, 0.01, 0.0316, 0.1]
noise_vals = [0.2, 0.3, 0.4]

n_datasets = 5
n_fits = 5

n_train = 20000
n_weight = 100000

f = open('parallel_comms.txt', 'w')

for i, x in enumerate(itertools.product(angle_vals, sig_frac_vals, noise_vals)):
    print(i, x)
    cmd = 'python run_genfit.py'
    cmd += ' -a' + str(x[0])
    cmd += ' -s' + str(x[1])
    cmd += ' -n' + str(x[2])
    cmd += ' -f' + str(n_fits)
    cmd += ' -d' + str(n_datasets)
    cmd += ' -t' + str(n_train)
    cmd += ' -w' + str(n_weight) + '\n'
    print(cmd)
    f.write(cmd)

f.close()


'''
sig_frac, noise, angle = 0, 0, 0
# parse command line
for c in sys.argv:
    if '-s' in c:
        sig_frac = float(c[2:])
    elif '-n' in c:
        noise = float(c[2:])
    elif '-a' in c:
        angle = float(c[2:])

print('generating data with \nnoise = ' + str(noise))
print('sig_frac = ' + str(sig_frac))
print('noise = ' + str(noise))

for dno in range(n_datasets):
    choose = string.ascii_lowercase + '0123456789'
    ds_tag = ''.join(random.choice(choose) for i in range(8))

    train_file_name = './datasets/train_ds_' + ds_tag
    weight_file_name = './datasets/weighted_ds_' + ds_tag

    # print(sig_frac)
    json_str = "{ \n \
    \"data\": { \n \
    \"n_events\": " + str(n_train_events) + ", \n \
    \"noise\": " + str(noise) + ", \n \
    \"angle\": " + str(angle) + ", \n \
    \"sig_fraction\": "
    json_str += str(sig_frac)
    json_str += ", \n \
    \"test_fraction\": 0.25, \n \
    \"min_mass\": 0.0, \n \
    \"max_mass\": 1.0, \n \
    \"mean_mass\": 0.5, \n \
    \"width_mass\": 0.18, \n \
    \"n_sigmas\": 2.5, \n \
    \"bkgd_beta\": 1.0, \n \
    \"ttsplit_random_state\": 42, \n \
    \"weighted_n_events\": " + str(n_weighted_events) + ", \n \
    \"train_data_file\": \"" + train_file_name + "\", \n \
    \"weighted_data_file\": \"" + weight_file_name + "\" \n \
    } \n \
    }"
    with open("config_run.json", "w") as f:
        f.write(json_str)
    cmd = "python make_datasets_2.py config_run.json"
    print(sig_frac)
    print(cmd)
    if run:
        os.system(cmd)

    for j in range(n_fits):
        ## python test_clf.py write_results tag:noise=0.3 tag:angle=1.4 tag:foo=bar noplot
        # run relegator
        cmd = 'python test_relegator_clf.py write_results noplot tag:noise=' + str(noise)
        cmd += ' tag:angle=' + str(angle)
        cmd += ' tag:nomsigfrac=' + str(sig_frac)
        cmd += ' tag:dataset=' + ds_tag
        print(cmd)
        if run:
            os.system(cmd)

        # run relegator v2
        cmd = 'python test_relegator_v2.py write_results noplot tag:noise=' + str(noise)
        cmd += ' tag:angle=' + str(angle)
        cmd += ' tag:nomsigfrac=' + str(sig_frac)
        cmd += ' tag:dataset=' + ds_tag
        print(cmd)
        if run:
            os.system(cmd)

        # run regressor
        cmd = 'python test_regressor_clf.py write_results noplot tag:noise=' + str(noise)
        cmd += ' tag:angle=' + str(angle)
        cmd += ' tag:nomsigfrac=' + str(sig_frac)
        cmd += ' tag:dataset=' + ds_tag
        print(cmd)
        if run:
            os.system(cmd)

    cmd = 'rm -f ' + train_file_name + '.csv ' + weight_file_name + '.csv'
    print(cmd)
    if run:
        os.system(cmd)
'''
