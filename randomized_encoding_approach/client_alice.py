import sys, time, pdb
sys.path.insert(0, "../esdp/")
import numpy as np
import connection as c
from REgen import generate_random_encoding

run = int(sys.argv[1])
port_ab = int(sys.argv[2])
port_ac = int(sys.argv[3])
port_as = int(sys.argv[4])
dataset = sys.argv[5]
# check for the dataset size
if dataset != "full" and dataset != "half" and dataset != "quarter":
	sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

a_max = 50
conn = c.Connection() # for the communication between parties

##################### Extended Secure Dot Product ###########################
data_folder = "../esdp/data/hiv_v3_loop_seq/dataset_size_exp_data/" # base folder address storing the sequence files
result_folder = "results/" + dataset + "/"

time.sleep(0.5)
conn2b = conn.connectHost('localhost', port_ab)
print("Alice-Bob")
time.sleep(0.5)
conn2c = conn.connectHost('localhost', port_ac)
print("Alice-Charlie")
time.sleep(0.5)
conn2s = conn.connectHost('localhost', port_as) # connect to the server
print("Alice-Server")

start_t1 = time.time() # start time after the connections are ready

# Alice data preparation - DATA MUST BE FORMATTED AS SAMPLE PER ROW!
a_training_data = np.loadtxt(data_folder + "tr_data_1_{}.txt".format(dataset), delimiter=',') # x1_alice_encoded.txt - a_encoded_seq.txt
a_test_data = np.loadtxt(data_folder + "test_data_1_{}.txt".format(dataset), delimiter=',') # x1_alice_encoded_test.txt - a_encoded_test_seq.txt
a_label = np.loadtxt(data_folder + "tr_label_1_{}.txt".format(dataset))
a_test_label = np.loadtxt(data_folder + "test_label_1_{}.txt".format(dataset))

# NOTE: data should be n_features-by-n_samples
data = np.transpose(np.concatenate((a_training_data, a_test_data), axis=0))
label = np.concatenate((a_label, a_test_label), axis=0)

n_features = data.shape[0] # number of features

dp = np.transpose(data) @ data
print("Dot product of the data of Alice is ready!")

alice_n_samples = np.array([a_training_data.shape[0], a_test_data.shape[0]])

# receive the number of sample information from other input-parties
bob_n_samples = np.sum(conn.receiveData(conn2b))
charlie_n_samples = conn.sendData(conn2c, alice_n_samples)
total_n_alice_samples = np.sum(alice_n_samples)

# generate encoding
offline_indices = [[] for i in range(n_features)]
offline_sign = [[] for i in range(n_features)]
enc_x = np.zeros((n_features, 6), dtype=int) # the order is: 1, r2, r1, r1, r2, r3
enc_y = np.zeros((n_features, 4), dtype=int) # the order is: 1, r1, r2, r4

latest_rand = generate_random_encoding(enc_x, enc_y, offline_indices, offline_sign, 1)

# generate random values for each pair of samples
ab_random_values = np.random.randint(0, a_max, (latest_rand, (total_n_alice_samples * bob_n_samples))) # the order is: 1, r2, r1, r1, r2, r3
ab_random_values[0] = 1

# send the random values to Bob
tmp_ind = np.array([0,2,1,3], dtype=int)
conn.sendData(conn2b, ab_random_values[enc_y[:,tmp_ind],])

# receive the random values from Charlie
ac_rv = conn.receiveData(conn2c)

###############################################################################
# compute the components of randomized encoding
ab_c1 = np.zeros((n_features, (total_n_alice_samples * bob_n_samples)))
ab_c2 = np.zeros((n_features, (total_n_alice_samples * bob_n_samples)))
for i in range(total_n_alice_samples):
	tmp = np.arange(i*bob_n_samples,(i+1)*bob_n_samples,dtype=int)
	# print("-- ind: {}".format(tmp))
	ab_c1[:,i*bob_n_samples:(i+1)*bob_n_samples] = data[:,i].reshape(-1,1) * ab_random_values[enc_x[:,0]][:,i*bob_n_samples:(i+1)*bob_n_samples] - ab_random_values[enc_x[:,2]][:,i*bob_n_samples:(i+1)*bob_n_samples]
	ab_c2[:,i*bob_n_samples:(i+1)*bob_n_samples] = data[:,i].reshape(-1,1) * ab_random_values[enc_x[:,1]][:,i*bob_n_samples:(i+1)*bob_n_samples] - (ab_random_values[enc_x[:,3]][:,i*bob_n_samples:(i+1)*bob_n_samples] * ab_random_values[enc_x[:,4]][:,i*bob_n_samples:(i+1)*bob_n_samples]) + ab_random_values[enc_x[:,5],i*bob_n_samples:(i+1)*bob_n_samples]

print("ab_c1 and ab_c2 are computed")

ab_c5 = np.zeros(ab_c1.shape)
for i in range(total_n_alice_samples):
	# print(i)
	for j in range(bob_n_samples):
		for f in range(n_features):
			ab_c5[f,(i*bob_n_samples)+j] = np.sum(ab_random_values[np.asarray(offline_indices[f]), (i*bob_n_samples)+j] * np.asarray(offline_sign[f]))

print("ab_c3 is computed")

###############################################################################
# compute the components of randomized encoding for the communication between Alice and Bob
ac_c3 = np.zeros((n_features, ac_rv.shape[2]))
ac_c4 = np.zeros((n_features, ac_rv.shape[2]))
# tmp = ac_rv.shape[1] / total_n_alice_samples
for i in range(total_n_alice_samples):
	tmp = np.arange(i, ac_rv.shape[2], total_n_alice_samples, dtype=int)
	ac_c3[:,tmp] = data[:,i].reshape(-1,1) * ac_rv[:,0][:,tmp] - ac_rv[:,1][:,tmp]
	ac_c4[:,tmp] = data[:,i].reshape(-1,1) * ac_rv[:,2][:,tmp] + ac_rv[:,3][:,tmp]

del a_training_data
del a_test_data
del a_label
del a_test_label
del data

print("Now the server time!")
conn.sendData(conn2s, [ab_c1, ab_c2, ab_c5, ac_c3, ac_c4, alice_n_samples, dp, label])

end_t1 = time.time()

# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Alice whole time: %s seconds" % whole_time)
f = open(result_folder + "run{:.0f}/time_alice.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Alice finished!")
