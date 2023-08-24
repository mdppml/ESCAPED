import sys, time, pdb
sys.path.insert(0, "../esdp/")
import numpy as np
import connection as c
import random
from REgen import generate_random_encoding

run = int(sys.argv[1])
port_ac = int(sys.argv[2])
port_bc = int(sys.argv[3])
port_cs = int(sys.argv[4])
dataset = sys.argv[5]
# check for the dataset size
if dataset != "full" and dataset != "half" and dataset != "quarter":
	sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

a_max = 50
conn = c.Connection() # for the connection between parties

##################### Extended Secure Dot Product ###########################
data_folder = "../esdp/data/hiv_v3_loop_seq/dataset_size_exp_data/" # base folder address storing the sequence files
result_folder = "results/" + dataset + "/"

time.sleep(0.2)
(sock2a, conn2a, addr2a) = conn.startConnection('localhost', port_ac)
print("Charlie-Alice")
time.sleep(0.5)
(sock2b, conn2b, addr2b) = conn.startConnection('localhost', port_bc)
print("Charlie-Bob")
time.sleep(3)
conn2s = conn.connectHost('localhost', port_cs)
print("Charlie-Server")
time.sleep(1)

start_t1 = time.time() # start time after the connections are ready

# Charlie data preparation - DATA MUST BE FORMATTED AS SAMPLE PER ROW!
c_training_data = np.loadtxt(data_folder + "tr_data_3_{}.txt".format(dataset), delimiter=',')
c_test_data = np.loadtxt(data_folder + "test_data_3_{}.txt".format(dataset), delimiter=',')
c_label = np.loadtxt(data_folder + "tr_label_3_{}.txt".format(dataset))
c_test_label = np.loadtxt(data_folder + "test_label_3_{}.txt".format(dataset))

data = np.transpose(np.concatenate((c_training_data, c_test_data), axis=0))
label = np.concatenate((c_label, c_test_label), axis=0)

n_features = data.shape[0] # number of features

dp = np.transpose(data) @ data
print("Dot product of the data of Charlie is ready!")

charlie_n_samples = np.array([c_training_data.shape[0], c_test_data.shape[0]])
total_n_charlie_samples = np.sum(charlie_n_samples)

# sending and receiving the number of sample information
conn.sendData(conn2b, charlie_n_samples)
alice_n_samples = np.sum(conn.receiveData(conn2a))

# generate encoding
offline_indices = [[] for i in range(n_features)]
offline_sign = [[] for i in range(n_features)]
enc_x = np.zeros((n_features, 6), dtype=int) # the order is: 1, r2, r1, r1, r2, r3
enc_y = np.zeros((n_features, 4), dtype=int) # the order is: 1, r1, r2, r4

latest_rand = generate_random_encoding(enc_x, enc_y, offline_indices, offline_sign, 1)

# generate random values for each pair of samples
ac_random_values = np.random.randint(0, a_max, (latest_rand, (total_n_charlie_samples * alice_n_samples))) # the order is: 1, r2, r1, r1, r2, r3
ac_random_values[0] = 1

# receive the random values from Bob
bc_rv = conn.receiveData(conn2b)

# send the random values to Alice
tmp_ind = np.array([0,2,1,3], dtype=int)
conn.sendData(conn2a, ac_random_values[enc_y[:,tmp_ind],])

###############################################################################
# compute the components of randomized encoding for the communication between Bob and Charlie
ac_c1 = np.zeros((n_features, (total_n_charlie_samples * alice_n_samples)))
ac_c2 = np.zeros((n_features, (total_n_charlie_samples * alice_n_samples)))
for i in range(total_n_charlie_samples):
	ac_c1[:,i*alice_n_samples:(i+1)*alice_n_samples] = data[:,i].reshape(-1,1) * ac_random_values[enc_x[:,0]][:,i*alice_n_samples:(i+1)*alice_n_samples] - ac_random_values[enc_x[:,2]][:,i*alice_n_samples:(i+1)*alice_n_samples]
	ac_c2[:,i*alice_n_samples:(i+1)*alice_n_samples] = data[:,i].reshape(-1,1) * ac_random_values[enc_x[:,1]][:,i*alice_n_samples:(i+1)*alice_n_samples] - (ac_random_values[enc_x[:,3]][:,i*alice_n_samples:(i+1)*alice_n_samples] * ac_random_values[enc_x[:,4]][:,i*alice_n_samples:(i+1)*alice_n_samples]) + ac_random_values[enc_x[:,5]][:,i*alice_n_samples:(i+1)*alice_n_samples]

print("ac_c1 and ac_c2 are computed")

ac_c5 = np.zeros(ac_c1.shape)
for i in range(total_n_charlie_samples):
	# print(i)
	for j in range(alice_n_samples):
		for f in range(n_features):
			ac_c5[f,(i*alice_n_samples)+j] = np.sum(ac_random_values[np.asarray(offline_indices[f]), (i*alice_n_samples)+j] * np.asarray(offline_sign[f]))

###############################################################################
# compute the components of randomized encoding for the communication between Alice and Bob
bc_c3 = np.zeros((n_features, bc_rv.shape[2]))
bc_c4 = np.zeros((n_features, bc_rv.shape[2]))
# tmp = bc_rv.shape[1] / total_n_charlie_samples
for i in range(total_n_charlie_samples):
	tmp = np.arange(i, bc_rv.shape[2], total_n_charlie_samples, dtype=int)
	bc_c3[:,tmp] = data[:,i].reshape(-1,1) * bc_rv[:,0][:,tmp] - bc_rv[:,1][:,tmp]
	bc_c4[:,tmp] = data[:,i].reshape(-1,1) * bc_rv[:,2][:,tmp] + bc_rv[:,3][:,tmp]

del c_training_data
del c_test_data
del c_label
del c_test_label

print("Now server time!")
conn.sendData(conn2s, [ac_c1, ac_c2, ac_c5, bc_c3, bc_c4, charlie_n_samples, dp, label])

end_t1 = time.time()

# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Charlie whole time: %s seconds" % whole_time)
f = open(result_folder + "run{:.0f}/time_charlie.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Charlie finished!")
