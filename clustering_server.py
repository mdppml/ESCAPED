import numpy as np
import pandas as pd
from scipy.io import savemat as sm
from numpy import concatenate as cn
from numpy import transpose as tr
import math, time, sys, pdb
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import connection as co
import common_functions as comm_func


run = int(sys.argv[1])
port_as_ = int(sys.argv[2])
port_bs_ = int(sys.argv[3])
port_cs_ = int(sys.argv[4])
# gamma_values = np.array([10**-6, 10**-3, 1, 10**3, 10**6])
# dataset = sys.argv[8]

client2s = co.Connection() # for the communication with clients

result_folder = "results/clustering/"

(sock2a, conn2a, addr2a) = client2s.startConnection('localhost', port_as_)
print("Server-Alice")
time.sleep(0.2)
(sock2b, conn2b, addr2b) = client2s.startConnection('localhost', port_bs_)
print("Server-Bob")
time.sleep(0.2)
(sock2c, conn2c, addr2c) = client2s.startConnection('localhost', port_cs_)
print("Server-Charlie")

start_t = time.time() # start time after the connections are ready

alice_data = client2s.receiveData(conn2a)
print("Alice's data is received!")
bob_data = client2s.receiveData(conn2b)
print("Bob's data is received!")
charlie_data = client2s.receiveData(conn2c)
print("Charlie's data is received!")

conn2a.close()
sock2a.close()
conn2b.close()
sock2b.close()
conn2c.close()
sock2c.close()

# Let the data of Alice, Bob and Charlie be represented by X,Y and Z respectively
# calculation of X^T * Y

print("Dot product calculations...")
# Alice-Bob
xy_ge = alice_data[1] + bob_data[1] + (1.0/alice_data[3] * bob_data[3])
xy_mirna = alice_data[5] + bob_data[6] + (1.0/alice_data[7] * bob_data[8])
xy_meth = alice_data[9] + bob_data[11] + (1.0/alice_data[11] * bob_data[13])
xy_cnv = alice_data[13] + bob_data[16] + (1.0/alice_data[15] * bob_data[18])

# Alice-Charlie
xz_ge = alice_data[2] + charlie_data[1] + (1.0/alice_data[3] * charlie_data[3])
xz_mirna = alice_data[6] + charlie_data[6] + (1.0/alice_data[7] * charlie_data[8])
xz_meth = alice_data[10] + charlie_data[11] + (1.0/alice_data[11] * charlie_data[13])
xz_cnv = alice_data[14] + charlie_data[16] + (1.0/alice_data[15] * charlie_data[18])

# Bob-Charlie
yz_ge = bob_data[2] + charlie_data[2] + (1.0/bob_data[4] * charlie_data[4])
yz_mirna = bob_data[7] + charlie_data[7] + (1.0/bob_data[9] * charlie_data[9])
yz_meth = bob_data[12] + charlie_data[12] + (1.0/bob_data[14] * charlie_data[14])
yz_cnv = bob_data[17] + charlie_data[17] + (1.0/bob_data[19] * charlie_data[19])
print("... are done!")


# gene expression
dp_ge = cn((cn((cn((cn((alice_data[0], xy_ge), axis=1), xz_ge), axis=1), cn((cn((tr(xy_ge), bob_data[0]), axis=1), yz_ge), axis=1)), axis=0),
	cn((cn((tr(xz_ge), tr(yz_ge)), axis=1), charlie_data[0]), axis=1)), axis=0)

# mirna
dp_mirna = cn((cn((cn((cn((alice_data[4], xy_mirna), axis=1), xz_mirna), axis=1), cn((cn((tr(xy_mirna), bob_data[5]), axis=1), yz_mirna), axis=1)), axis=0),
	cn((cn((tr(xz_mirna), tr(yz_mirna)), axis=1), charlie_data[5]), axis=1)), axis=0)

# methylation
dp_meth = cn((cn((cn((cn((alice_data[8], xy_meth), axis=1), xz_meth), axis=1), cn((cn((tr(xy_meth), bob_data[10]), axis=1), yz_meth), axis=1)), axis=0),
	cn((cn((tr(xz_meth), tr(yz_meth)), axis=1), charlie_data[10]), axis=1)), axis=0)

# copy number variation
dp_cnv = cn((cn((cn((cn((alice_data[12], xy_cnv), axis=1), xz_cnv), axis=1), cn((cn((tr(xy_cnv), bob_data[15]), axis=1), yz_cnv), axis=1)), axis=0),
	cn((cn((tr(xz_cnv), tr(yz_cnv)), axis=1), charlie_data[15]), axis=1)), axis=0)

# np.savetxt("dp_test_cnv.csv", dp_cnv, fmt='%.0f', delimiter=',')

# clinical data of the patients
clinical_data = pd.concat([pd.concat([alice_data[16], bob_data[20]], axis=0, sort=False), charlie_data[20]], axis=0, sort=False)

# number of training and test samples in both clinics
# training_ind = cn((cn((np.arange(0, alice_data[16][0], dtype=int), np.arange(np.sum(alice_data[16]),np.sum(alice_data[16])+bob_data[20][0], dtype=int))),
# 	np.arange(np.sum(alice_data[16])+np.sum(bob_data[20]),np.sum(alice_data[16])+np.sum(bob_data[20])+charlie_data[20][0], dtype=int)))
# test_ind = cn((cn((np.arange(alice_data[16][0], np.sum(alice_data[16]), dtype=int), np.arange(np.sum(alice_data[16])+bob_data[20][0],np.sum(alice_data[16])+np.sum(bob_data[20]), dtype=int))),
# 	np.arange(np.sum(alice_data[16])+np.sum(bob_data[20])+bob_data[20][0]+1,np.sum(alice_data[16])+np.sum(bob_data[20])+np.sum(charlie_data[20]), dtype=int)))

# delete data from input-parties
del alice_data
del bob_data
del charlie_data

### one-kernel-matrix-per-data-type
# rule_of_thumb_gamma = 1.0 / (2 * clinical_data.shape[0]**2) # the choice of gamma is done by following the rule of thumb

rule_of_thumb_gamma_ge = 1.0 / (2 * 19433**2) # the choice of gamma is done by following the rule of thumb
rule_of_thumb_gamma_mirna = 1.0 / (2 * 581**2) # the choice of gamma is done by following the rule of thumb
rule_of_thumb_gamma_meth = 1.0 / (2 * 57159**2) # the choice of gamma is done by following the rule of thumb
rule_of_thumb_gamma_cnv = 1.0 / (2 * 23817**2) # the choice of gamma is done by following the rule of thumb
print("The number of patients: {:.0f}".format(clinical_data.shape[0]))
print("The gamma of GE: {:.15f}".format(rule_of_thumb_gamma_ge))
print("The gamma of miRNA: {:.15f}".format(rule_of_thumb_gamma_mirna))
print("The gamma of METH: {:.15f}".format(rule_of_thumb_gamma_meth))
print("The gamma of CNV: {}".format(rule_of_thumb_gamma_cnv))
sm("kms/rbf_ge", {"rbf_ge": comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_ge, rule_of_thumb_gamma_ge))})
sm("kms/rbf_mirna", {"rbf_mirna": comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_mirna, rule_of_thumb_gamma_mirna))})
sm("kms/rbf_meth", {"rbf_meth": comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_meth, rule_of_thumb_gamma_meth))})
sm("kms/rbf_cnv", {"rbf_cnv": comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_cnv, rule_of_thumb_gamma_cnv))})

### five-kernel-matrix-per-data-type
# # calculate rbf kernel
# rbf_ge = np.zeros((len(gamma_values), clinical_data.shape[0], clinical_data.shape[0]))
# rbf_mirna = np.zeros((len(gamma_values), clinical_data.shape[0], clinical_data.shape[0]))
# rbf_meth = np.zeros((len(gamma_values), clinical_data.shape[0], clinical_data.shape[0]))
# rbf_cnv = np.zeros((len(gamma_values), clinical_data.shape[0], clinical_data.shape[0]))
#
# for i in range(len(gamma_values)):
# 	# rbf_ge[i,:,:] = comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_ge, gamma_values[i]))
# 	# rbf_mirna[i,:,:] = comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_mirna, gamma_values[i]))
# 	# rbf_meth[i,:,:] = comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_meth, gamma_values[i]))
# 	# rbf_cnv[i,:,:] = comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_cnv, gamma_values[i]))
# 	sm("kms/rbf_ge_{:.0f}".format(i), {"rbf_ge_{:.0f}".format(i): comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_ge, gamma_values[i]))})
# 	sm("kms/rbf_mirna_{:.0f}".format(i), {"rbf_mirna_{:.0f}".format(i): comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_mirna, gamma_values[i]))})
# 	sm("kms/rbf_meth_{:.0f}".format(i), {"rbf_meth_{:.0f}".format(i): comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_meth, gamma_values[i]))})
# 	sm("kms/rbf_cnv_{:.0f}".format(i), {"rbf_cnv_{:.0f}".format(i): comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp_cnv, gamma_values[i]))})

del dp_ge
del dp_mirna
del dp_meth
del dp_cnv

# print("*****************************************************")
# print("***Parameters:")
# print("gamma: {:.3f}".format(opt_gamma))
# print("***Results:")
# print("*****************************************************")

# save the kernel matrices for web rMKL
# sm("rbf_ge.mat", {"rbf_ge": rbf_ge})
# sm("rbf_mirna.mat", {"rbf_mirna": rbf_mirna})
# sm("rbf_meth.mat", {"rbf_meth": rbf_meth})
# sm("rbf_cnv.mat", {"rbf_cnv": rbf_cnv})

# save the ids of the patients
np.savetxt("kms/ids.txt", clinical_data.index.values, fmt="%s", delimiter="\n")

# pdb.set_trace()

# execution time
end_t = time.time()
whole_time = end_t - start_t
print("The server whole time: %s seconds" % whole_time)

f = open(result_folder + "time_server_run_{:.0f}.csv".format(run), 'w')
f.write("{:.5f}\n".format(whole_time))
f.close()

print("Server finished!")













	#atadam
