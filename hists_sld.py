#!/usr/bin/env python
'''
'''

import sys
import random
import numpy as np
from ROOT import TVector3, TLorentzVector
import pandas as pd
import matplotlib.pyplot as plt

decay_types = ['sl_mu', 'sl_e', 'ppim', 'had_mu', 'hadt_mu', 'had_e', 'hadt_e', 'kL']
#decay_types = ['3_body1', '2body_1']
n_evts = 1e6  # number of events to process
n_files = len(sys.argv) - 1
decays = []
dfs = []
for i in range(len(sys.argv) - 1):
    decays.append(sys.argv[i+1].split("/")[-1].split(".")[0])
    print(decays[i])
    dfs.append(pd.read_csv(sys.argv[i+1]))
    ## drop all of the columns pertaining to undetected neutrinos
    neut_cols = [c for c in dfs[i].columns if '_miss' in c]
    dfs[i].drop(neut_cols, axis=1, inplace=True)
    dfs[i]['type_label'] = i

#print(dfs[0].head)
df = pd.concat(dfs)

## some mass variables GeV/c^2
m_gamma    = 0.0
m_target   = 0.938272
m_kplus    = 0.493677
m_lambda   = 1.115683
m_pion     = 0.13957
m_muon     = 0.105658
m_proton   = 0.938272
m_neutrino = 0.0
m_electron = 0.000511

## type key
k_plus_id   = 11
proton_id   = 14
mum_id      = 6
neutrino_id = 4
lambda_id   = 18
pim_id      = 9
gamma_id    = 0
electron_id = 3

## defining vectors
vert       = TVector3()
sec_vert   = TVector3()
sec_vertm  = TVector3()
p4_gamma   = TLorentzVector()
p4_target  = TLorentzVector()
p4_total   = TLorentzVector()
p4_lambda  = TLorentzVector()
p4_pim     = TLorentzVector()



#
#     #p4_gamma.SetPxPyPzE(0.0, 0.0, e_gamma, e_gamma)
#     #p4_target.SetPxPyPzE(0.0, 0.0, 0.0, m_target)
#
#     lam_pz = np.random.random()*(lam_mom_range[1] - lam_mom_range[0]) + lam_mom_range[0]
#     p4_total.SetXYZM(0.0, 0.0, lam_pz, m_lambda)
#
#     if evt == 0:
#         cols.append('X_id')
#         cols.append('X_mass')
#         cols.append('X_px_truth')
#         cols.append('X_py_truth')
#         cols.append('X_pz_truth')
#         cols.append('X_px_meas')
#         cols.append('X_py_meas')
#         cols.append('X_pz_meas')
#
#     event_data.append(primary_id[0])
#     event_data.append(p4_total.M())
#     event_data.append(p4_total.Px())
#     event_data.append(p4_total.Py())
#     event_data.append(p4_total.Pz())
#
#     event_data.append(comp_smear(p4_total.Px(), mom_sigma))
#     event_data.append(comp_smear(p4_total.Py(), mom_sigma))
#     event_data.append(comp_smear(p4_total.Pz(), mom_sigma))
#
#     if evt == 0:
#         cols.append('V1_x_truth')
#         cols.append('V1_y_truth')
#         cols.append('V1_z_truth')
#         cols.append('V1_x_meas')
#         cols.append('V1_y_meas')
#         cols.append('V1_z_meas')
#
#     event_data.append(vert.X())
#     event_data.append(vert.Y())
#     event_data.append(vert.Z())
#     event_data.append(add_smear(vert.X(), vert_sigma))
#     event_data.append(add_smear(vert.Y(), vert_sigma))
#     event_data.append(add_smear(vert.Z(), vert_sigma))
#
#     ### BEGIN Lambda decay ###
#     lambda_decay.SetDecay(p4_total, len(secondary_masses), secondary_masses)
#     lambda_decay.Generate()
#
#     t, r   = 0, 0
#     keep_t = 0
#
#     # generate rest-frame lifetime
#     while keep_t == 0:
#         t = random.uniform(0.0, 7.0*tave)
#         r = random.uniform(0,1/tave)
#         if r <= prob_dec(t, tave):
#             keep_t = 1
#
#     # dilate for lab frame
#     t_lab = t * p4_total.Gamma()
#     t_vert1 = t_vert0 + t_lab
#
#     vel  = p4_total.Beta() * c #cm/ns
#     d    = vel * t_lab  #cm
#
#     velv     = p4_total.Vect().Unit() * vel
#     lam_vert = velv * t_lab  + vert
#
#     #vert_seq_num += 1
#
#     if evt == 0:
#         cols.append('V2_nparts')
#         cols.append('V2_t_truth')
#         cols.append('V2_x_truth')
#         cols.append('V2_y_truth')
#         cols.append('V2_z_truth')
#         cols.append('V2_x_meas')
#         cols.append('V2_y_meas')
#         cols.append('V2_z_meas')
#
#     event_data.append(len(secondary_masses))
#     event_data.append(t_vert1)
#     event_data.append(lam_vert.X())
#     event_data.append(lam_vert.Y())
#     event_data.append(lam_vert.Z())
#     event_data.append(add_smear(lam_vert.X(), vert_sigma))
#     event_data.append(add_smear(lam_vert.Y(), vert_sigma))
#     event_data.append(add_smear(lam_vert.Z(), vert_sigma))
#
#     for part_iter in range(len(secondary_masses)):
#         tag = 'Y' + str(part_iter + 1)
#         if evt == 0:
#             cols.append(tag + '_id')
#             cols.append(tag + '_mass')
#             cols.append(tag + '_px_truth')
#             cols.append(tag + '_py_truth')
#             cols.append(tag + '_pz_truth')
#             cols.append(tag + '_px_meas')
#             cols.append(tag + '_py_meas')
#             cols.append(tag + '_pz_meas')
#
#         event_data.append(secondary_ids[part_iter])
#         event_data.append(secondary_masses[part_iter])
#
#         secondary_p4s.append(lambda_decay.GetDecay(part_iter))
#
#         event_data.append(secondary_p4s[part_iter].X())
#         event_data.append(secondary_p4s[part_iter].Y())
#         event_data.append(secondary_p4s[part_iter].Z())
#         event_data.append(comp_smear(secondary_p4s[part_iter].X(), mom_sigma))
#         event_data.append(comp_smear(secondary_p4s[part_iter].Y(), mom_sigma))
#         event_data.append(comp_smear(secondary_p4s[part_iter].Z(), mom_sigma))
#
#     ### END Lambda decay ###
#
#     data_arr.append(event_data)
#     # ### BEGIN pi- decay ###
#     # if decay in ['had_mu', 'hadt_mu', 'had_e', 'hadt_e']:
#     #     p4_pim = secondary_p4s[1]
#     #     pim_decay.SetDecay(p4_pim, len(tertiary_masses), tertiary_masses)
#     #     pim_decay.Generate()
#     #
#     #     t, r   = 0, 0
#     #     keep_t = 0
#     #
#     #     # generate rest-frame lifetime
#     #     while keep_t == 0:
#     #         if decay in ['hadt_mu', 'hadt_e']:
#     #             t = random.uniform(0.0, 1.0*pave)
#     #         elif decay in ['had_mu', 'had_e']:
#     #             t = random.uniform(0.0, 7.0*pave)
#     #
#     #         r = random.uniform(0,1/pave)
#     #         if r <= prob_dec(t, pave):
#     #             keep_t = 1
#     #
#     #     # dilate for lab frame
#     #     tlab = t * p4_pim.Gamma()
#     #     t_vert2 = t_vert1 + tlab
#     #
#     #     vel  = p4_pim.Beta() * c #cm/ns
#     #     d    = vel * tlab  #cm
#     #
#     #     velv     = p4_pim.Vect().Unit() * vel
#     #     pim_vert = velv * tlab  + lam_vert
#     #
#     #     vert_seq_num += 1
#     #     outputfile.write(str(vert_seq_num) + " ")
#     #     outputfile.write(str(len(tertiary_masses)) + " ")
#     #     outputfile.write(str(round(t_vert2, 8)) + " ")
#     #     outputfile.write(str(round(pim_vert.X(), 8)) + " ")
#     #     outputfile.write(str(round(pim_vert.Y(), 8)) + " ")
#     #     outputfile.write(str(round(pim_vert.Z(), 8)) + " \n")
#     #
#     #     for part_iter in range(len(tertiary_masses)):
#     #         outputfile.write(str(p_id) + " ")
#     #         outputfile.write(str(tertiary_ids[part_iter]) + " ")
#     #         outputfile.write(str(pim_parent_id) + " ")
#     #         outputfile.write(str(tertiary_prod_decay_verts[part_iter]) + " ")
#     #
#     #         tertiary_p4s.append(pim_decay.GetDecay(part_iter))
#     #         outputfile.write(str(round(tertiary_p4s[part_iter].E(), 8)) + " ")
#     #         outputfile.write(str(round(tertiary_p4s[part_iter].X(), 8)) + " ")
#     #         outputfile.write(str(round(tertiary_p4s[part_iter].Y(), 8)) + " ")
#     #         outputfile.write(str(round(tertiary_p4s[part_iter].Z(), 8)) + "\n")
#     #         p_id += 1
#     #
#     # ### END pi- decay ###
#
#     evt = evt + 1
#     #if evt % 1000 == 0:
#     #    print(evt)
#
#
# df = pd.DataFrame(data_arr, columns=cols)
# df = df.round(5)
# print(df.head())
#
# df.to_csv("raw_mc_" + decay + ".ascii", sep=',')
#
# print("Generator exited smoothly after writing " + str(evt) + " events.")
#
# df.hist(bins=50)
# plt.tight_layout
# plt.show()
