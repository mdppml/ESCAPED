import math, time, sys, pdb
sys.path.insert(0, "../esdp/")
import numpy as np
from numpy import concatenate as cn
from numpy import transpose as tr
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import connection as co
import common_functions as comm_func

run = int(sys.argv[1])
opt_gamma = float(sys.argv[2])
opt_c = float(sys.argv[3])
opt_w1 = int(sys.argv[4])
port_as_ = int(sys.argv[5])
port_bs_ = int(sys.argv[6])
port_cs_ = int(sys.argv[7])
dataset = sys.argv[8]
# check for the dataset size
if dataset != "full" and dataset != "half" and dataset != "quarter":
	sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

client2s = co.Connection() # for the communication with clients

result_folder = "results/" + dataset + "/"

label_encoding = {"CCR5":0, "OTHER":1}
weights = {label_encoding["CCR5"]:1, label_encoding["OTHER"]:opt_w1}

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

print("Dot product calculations...")
xy = np.sum(alice_data[0] * bob_data[3] + alice_data[1] + bob_data[4] + alice_data[2], axis=0)
xz = np.sum(charlie_data[0] * alice_data[3] + charlie_data[1] + alice_data[4] + charlie_data[2], axis=0)
yz = np.sum(bob_data[0] * charlie_data[3] + bob_data[1] + charlie_data[4] + bob_data[2], axis=0)

xy = xy.reshape(np.sum(alice_data[5]), np.sum(bob_data[5]))
xz = np.transpose(xz.reshape(np.sum(charlie_data[5]), np.sum(alice_data[5])))
yz = yz.reshape(np.sum(bob_data[5]), np.sum(charlie_data[5]))

dp = cn((cn((cn((cn((alice_data[6], xy), axis=1), xz), axis=1), cn((cn((tr(xy), bob_data[6]), axis=1), yz), axis=1)), axis=0), cn((cn((tr(xz), tr(yz)), axis=1), charlie_data[6]), axis=1)), axis=0)

# labels
label = np.concatenate((alice_data[7], bob_data[7], charlie_data[7]))

# number of training and test samples in both clinics
training_ind = cn((cn((np.arange(0, alice_data[5][0], dtype=int), np.arange(np.sum(alice_data[5]),np.sum(alice_data[5])+bob_data[5][0], dtype=int))), np.arange(np.sum(alice_data[5])+np.sum(bob_data[5]),np.sum(alice_data[5])+np.sum(bob_data[5])+charlie_data[5][0], dtype=int)))
test_ind = cn((cn((np.arange(alice_data[5][0], np.sum(alice_data[5]), dtype=int), np.arange(np.sum(alice_data[5])+bob_data[5][0],np.sum(alice_data[5])+np.sum(bob_data[5]), dtype=int))), np.arange(np.sum(alice_data[5])+np.sum(bob_data[5])+bob_data[5][0]+1,np.sum(alice_data[5])+np.sum(bob_data[5])+np.sum(charlie_data[5]), dtype=int)))

# delete data from input-parties
del alice_data
del bob_data
del charlie_data

# calculate rbf kernel
rbf_km = comm_func.normalizeKernelMatrix(comm_func.sdp_rbf(dp, opt_gamma))
del dp

# train svm model with optimal parameters
model = svm.SVC(C=opt_c, kernel='precomputed', class_weight=weights, probability=True)
model.fit(rbf_km[training_ind][:,training_ind], label[training_ind])

# predict with the model
pre = model.predict(rbf_km[test_ind][:,training_ind])
prob_pre = model.predict_proba(rbf_km[test_ind][:,training_ind])

###################################################################
### Different Measurement Techniques
# class based accuracy
# accs = comm_func.getClassBasedAccuracies(label[n_training:], pre)
conf_mat = confusion_matrix(label[test_ind], pre)
# np.savetxt(result_folder + "run{:.0f}/conf_mat.csv".format(run), conf_mat, delimiter=',', fmt='%i')

# F-measure
f_measure = f1_score(label[test_ind], pre, pos_label=1)
# f = open(result_folder + "run{:.0f}/f1.csv".format(run), 'w')
# f.write("{:.2f}\n".format(f_measure))
# f.close()

# ROC:
auroc = roc_auc_score(label[test_ind], prob_pre[:,label_encoding["OTHER"]]) # assumption: we are looking at the prediction probabilities of class 1
# f = open(result_folder + "run{:.0f}/auroc.csv".format(run), 'w')
# f.write("{:.2f}\n".format(auroc))
# f.close()
###################################################################

print("*****************************************************")
print("***Parameters:")
print("C: {:.3f}".format(opt_c))
print("gamma: {:.3f}".format(opt_gamma))
print("w1: {:.3f}".format(opt_w1))
print("Dataset size: {}".format(dataset))
print("***Results:")
print("Class-based accuracy: {}".format(conf_mat))
print("F-measure: {}".format(f_measure))
print("ROC AUC: {}".format(auroc))
print("*****************************************************")

# execution time
end_t = time.time()
whole_time = end_t - start_t
print("Server whole time: %s seconds" % whole_time)
f = open(result_folder + "run{:.0f}/time.csv".format(run), 'w')
f.write("{:.5f}\n".format(whole_time))
f.close()

print("Server finished!")
