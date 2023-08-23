import numpy as np
from numpy import concatenate as cn
from numpy import transpose as tr
import math, time, sys, pdb, os
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import connection as co
import common_functions as comm_func


run = int(sys.argv[1])
opt_gamma = float(sys.argv[2])
opt_c = float(sys.argv[3])
opt_w1 = int(sys.argv[4])
base_port = int(sys.argv[5])
num_of_parties = int(sys.argv[6])
exp_type = int(sys.argv[7]) # 1 indicates the varying number of input-parties -- 2 indicates the varying size of the dataset

# checks
if exp_type != 1 and exp_type != 2:
	sys.exit("Wrong experiment type is given. 1 for varying number of input-parties and 2 for varying dataset size.")

if exp_type == 2:
	if num_of_parties != 3:
		sys.exit("The varying dataset size experiment is available for only 3 input-parties.")
	dataset_size = sys.argv[8]
	if dataset_size != "full" and dataset_size != "half" and dataset_size != "quarter":
		sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

client2s = co.Connection()  # for the communication with clients

label_encoding = {"CCR5":0, "OTHER":1}
weights = {label_encoding["CCR5"]:1, label_encoding["OTHER"]:opt_w1}

partners = comm_func.generate_turns(num_of_parties)
conns = np.empty([num_of_parties + 1,], dtype=object)

# connect to the input-parties
base_port += num_of_parties**2
for i in range(num_of_parties):
	# print("Server is waiting for party {:.0f} -- port: {:.0f}".format(i + 1, base_port + i + 1))
	(tmp, conns[i], tmp) = client2s.startConnection('localhost', int(base_port + i + 1))
	# print("Server is connected to party {:.0f}.".format(i + 1))
	time.sleep(0.2)

start_t = time.time() # start time after the connections are ready

# receive the data from the input-parties
components = []
unmaskers = []
n_samples = []
self_dps = []
label_list = []
total_n_samples = 0
n_training_samples = 0
start_points = np.zeros((num_of_parties + 1,), dtype=int)
for i in range(num_of_parties):
	tmp_comp, tmp_unmasker, tmp_n_samples, tmp_dp, tmp_label = client2s.receiveData(conns[i])
	components.append(tmp_comp)
	unmaskers.append(tmp_unmasker)
	n_samples.append(tmp_n_samples)
	self_dps.append(tmp_dp)
	label_list.append(tmp_label)
	n_training_samples += tmp_n_samples[0]
	total_n_samples += np.sum(tmp_n_samples)
	start_points[i+1] = start_points[i] + np.sum(tmp_n_samples)
	# print("The data of party {:.0f} is received!".format(i+1))
time.sleep(0.3)
start_points[num_of_parties] = total_n_samples

# close the connections
for i in range(num_of_parties):
	conns[i].close()

start_ml = time.time()

# assemble the gram matrix and the label matrix
dp = np.zeros((total_n_samples, total_n_samples))
label = np.zeros((total_n_samples,))

# self dot product
for i in range(num_of_parties):
	dp[start_points[i]:start_points[i+1]][:,start_points[i]:start_points[i+1]] = self_dps[i]
	label[start_points[i]:start_points[i+1]] = label_list[i]

for i in range(num_of_parties + (num_of_parties % 2) - 1): # inter-input-party dot product
	utilized = np.zeros((num_of_parties,), dtype=int)
	for j in range(num_of_parties):
		if partners[j,i] != 0 and utilized[j] != 1 and utilized[partners[j,i]-1] != 1:
			tmp_dp = components[j][i] + components[partners[j,i]-1][i] + unmaskers[j][i] * unmaskers[partners[j,i]-1][i]
			dp[start_points[j]:start_points[j+1]][:,start_points[partners[j,i]-1]:start_points[partners[j,i]]] = tmp_dp
			dp[start_points[partners[j,i]-1]:start_points[partners[j,i]]][:,start_points[j]:start_points[j+1]] = np.transpose(tmp_dp)
			utilized[j] = 1
			utilized[partners[j,i]-1] = 1

training_ind = np.zeros((n_training_samples,), dtype=int)
test_ind = np.zeros((total_n_samples - n_training_samples,), dtype=int)
cur_tr = 0
cur_test = 0
for i in range(num_of_parties):
	tr_s = start_points[i]
	tr_e = tr_s + n_samples[i][0]
	test_s = tr_e
	test_e = start_points[i+1]
	training_ind[cur_tr:cur_tr+n_samples[i][0]] = np.arange(tr_s, tr_e, dtype=int)
	test_ind[cur_test:cur_test+n_samples[i][1]] = np.arange(test_s, test_e, dtype=int)
	cur_tr += n_samples[i][0]
	cur_test += n_samples[i][1]

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
# accs = OK.getClassBasedAccuracies(label[n_training:], pre)
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
print("***Results:")
print("Class-based accuracy: \n{}".format(conf_mat))
print("F-measure: {}".format(f_measure))
print("ROC AUC: {}".format(auroc))
print("*****************************************************")

# execution time
end_t = time.time()
whole_time = end_t - start_t
ml_time = end_t - start_ml
print(f'The server whole time: {whole_time:{1}.{5}} seconds')

if exp_type == 1:
	result_folder = f'results/hiv_coreceptor_prediction/num_of_input-parties/{num_of_parties}_parties/run{run}/'
else:
	result_folder = f'results/hiv_coreceptor_prediction/dataset_size/{dataset_size}/run{run}/'
if not os.path.exists(result_folder):
	os.makedirs(result_folder)
with open(f'{result_folder}time_server.csv', 'w') as f:
	f.write(f'{whole_time:.5f},{ml_time:.5f}\n')

print("Server finished!")
