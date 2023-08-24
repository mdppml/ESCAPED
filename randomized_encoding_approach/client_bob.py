import sys, time, pdb
sys.path.insert(0, "../esdp/")
import numpy as np
import connection as c
import random
from REgen import generate_random_encoding

run = int(sys.argv[1])
port_ab = int(sys.argv[2])
port_bc = int(sys.argv[3])
port_bs = int(sys.argv[4])
dataset = sys.argv[5]
# check for the dataset size
if dataset != "full" and dataset != "half" and dataset != "quarter":
	sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

a_max = 50
conn = c.Connection() # for the connection between parties

##################### Extended Secure Dot Product ###########################
data_folder = "../esdp/data/hiv_v3_loop_seq/dataset_size_exp_data/" # base folder address storing the sequence files
result_folder = "results/" + dataset + "/"

# time.sleep(0.2)
(sock2a, conn2a, addr2b) = conn.startConnection('localhost', port_ab)
print("Bob-Alice")
time.sleep(2)
conn2c = conn.connectHost('localhost', port_bc)
print("Bob-Charlie")
time.sleep(1)
conn2s = conn.connectHost('localhost', port_bs)
print("Bob-Server")

start_t1 = time.time() # start time after the connections are ready

# Bob data preparation - DATA MUST BE FORMATTED AS SAMPLE PER ROW!
b_training_data = np.loadtxt(data_folder + "tr_data_2_{}.txt".format(dataset), delimiter=',') #x2_bob_encoded.txt - a_encoded_seq.txt
b_test_data = np.loadtxt(data_folder + "test_data_2_{}.txt".format(dataset), delimiter=',') #x2_bob_encoded_test.txt - a_encoded_test_seq.txt
b_label = np.loadtxt(data_folder + "tr_label_2_{}.txt".format(dataset))
b_test_label = np.loadtxt(data_folder + "test_label_2_{}.txt".format(dataset))

data = np.transpose(np.concatenate((b_training_data, b_test_data), axis=0))
label = np.concatenate((b_label, b_test_label), axis=0)

n_features = data.shape[0] # number of features

dp = np.transpose(data) @ data
print("Dot product of the data of Bob is ready!")
bob_n_samples = np.array([b_training_data.shape[0], b_test_data.shape[0]])
total_n_bob_samples = np.sum(bob_n_samples)

# sending and receiving the number of sample information
conn.sendData(conn2a, bob_n_samples)
charlie_n_samples = np.sum(conn.receiveData(conn2c))

# generate encoding
offline_indices = [[] for i in range(n_features)]
offline_sign = [[] for i in range(n_features)]
enc_x = np.zeros((n_features, 6), dtype=int) # the order is: 1, r2, r1, r1, r2, r3
enc_y = np.zeros((n_features, 4), dtype=int) # the order is: 1, r1, r2, r4

latest_rand = generate_random_encoding(enc_x, enc_y, offline_indices, offline_sign, 1)

# generate random values for each pair of samples
bc_random_values = np.random.randint(0, a_max, (latest_rand, (total_n_bob_samples * charlie_n_samples))) # the order is: 1, r2, r1, r1, r2, r3
bc_random_values[0] = 1

# receive the random values from Alice
ab_rv = conn.receiveData(conn2a)

# send the random values to Charlie
tmp_ind = np.array([0,2,1,3], dtype=int)
conn.sendData(conn2c, bc_random_values[enc_y[:,tmp_ind],])

###############################################################################
# compute the components of randomized encoding for the communication between Bob and Charlie
bc_c1 = np.zeros((n_features, (total_n_bob_samples * charlie_n_samples)))
bc_c2 = np.zeros((n_features, (total_n_bob_samples * charlie_n_samples)))
for i in range(total_n_bob_samples):
	bc_c1[:,i*charlie_n_samples:(i+1)*charlie_n_samples] = data[:,i].reshape(-1,1) * bc_random_values[enc_x[:,0]][:,i*charlie_n_samples:(i+1)*charlie_n_samples] - bc_random_values[enc_x[:,2]][:,i*charlie_n_samples:(i+1)*charlie_n_samples]
	bc_c2[:,i*charlie_n_samples:(i+1)*charlie_n_samples] = data[:,i].reshape(-1,1) * bc_random_values[enc_x[:,1]][:,i*charlie_n_samples:(i+1)*charlie_n_samples] - (bc_random_values[enc_x[:,3]][:,i*charlie_n_samples:(i+1)*charlie_n_samples] * bc_random_values[enc_x[:,4]][:,i*charlie_n_samples:(i+1)*charlie_n_samples]) + bc_random_values[enc_x[:,5]][:,i*charlie_n_samples:(i+1)*charlie_n_samples]

print("bc_c1 and bc_c2 are computed")

bc_c5 = np.zeros(bc_c1.shape)
for i in range(total_n_bob_samples):
	# print(i)
	for j in range(charlie_n_samples):
		for f in range(n_features):
			bc_c5[f,(i*charlie_n_samples)+j] = np.sum(bc_random_values[np.asarray(offline_indices[f]),(i*charlie_n_samples)+j] * np.asarray(offline_sign[f]))

###############################################################################
# compute the components of randomized encoding for the communication between Alice and Bob
ab_c3 = np.zeros((n_features, ab_rv.shape[2]))
ab_c4 = np.zeros((n_features, ab_rv.shape[2]))
# tmp = ab_rv.shape[1] / total_n_bob_samples
for i in range(total_n_bob_samples):
	tmp = np.arange(i, ab_rv.shape[2], total_n_bob_samples, dtype=int)
	ab_c3[:,tmp] = data[:,i].reshape(-1,1) * ab_rv[:,0][:,tmp] - ab_rv[:,1][:,tmp]
	ab_c4[:,tmp] = data[:,i].reshape(-1,1) * ab_rv[:,2][:,tmp] + ab_rv[:,3][:,tmp]

del b_training_data
del b_test_data
del b_label
del b_test_label

print("Now server time!")
conn.sendData(conn2s, [bc_c1, bc_c2, bc_c5, ab_c3, ab_c4, bob_n_samples, dp, label])

end_t1 = time.time()

# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Bob whole time: %s seconds" % whole_time)
f = open(result_folder + "run{:.0f}/time_bob.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Bob finished!")
