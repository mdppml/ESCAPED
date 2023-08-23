import numpy as np
import pandas as pd
import sys, time, pdb
import connection as c

run = int(sys.argv[1])
port_ab = int(sys.argv[2])
port_ac = int(sys.argv[3])
port_as = int(sys.argv[4])
# dataset = sys.argv[5]

a_max = 50
conn = c.Connection() # for the communication between parties

################################################################################
data_folder = "data/" # base folder address storing the sequence files
result_folder = "results/clustering/"

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

################################################################################
'''
There will be different type of data - gene expression, methylation, mutation etc.
We need to apply the same procedure for each type separately
Note that we process the data in a format that each column is one sample!
'''

# gene expression data
ge_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/alice_geneExp.csv", delimiter=','))
dp_ge = np.transpose(ge_data) @ ge_data
ge_a = np.random.uniform(1, a_max, (ge_data.shape[0], ge_data.shape[1])) # mask
ge_data = ge_data - ge_a
ge_alpha = np.random.uniform(1, a_max, (1))

# miRNA data
mirna_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/alice_miRNA.csv", delimiter=','))
dp_mirna = np.transpose(mirna_data) @ mirna_data
mirna_a = np.random.uniform(1, a_max, (mirna_data.shape[0], mirna_data.shape[1])) # mask
mirna_data = mirna_data - mirna_a
mirna_alpha = np.random.uniform(1, a_max, (1))

# methylation data
meth_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/alice_meth.csv", delimiter=','))
dp_meth = np.transpose(meth_data) @ meth_data
meth_a = np.random.uniform(1, a_max, (meth_data.shape[0], meth_data.shape[1])) # mask
meth_data = meth_data - meth_a
meth_alpha = np.random.uniform(1, a_max, (1))

# copy number variation data
cnv_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/alice_cnv.csv", delimiter=','))
dp_cnv = np.transpose(cnv_data) @ cnv_data
cnv_a = np.random.uniform(1, a_max, (cnv_data.shape[0], cnv_data.shape[1])) # mask
cnv_data = cnv_data - cnv_a
cnv_alpha = np.random.uniform(1, a_max, (1))

# clinical data - each row represents a sample, that is it is not a column-wise representation of samples
clinical_data = pd.read_csv(data_folder + "cancer_dataset/hnsc/alice_processed_clinical_data.csv", index_col=0, header=0)

################################################################################
print("Alice - Dot products are ready!")

# communicate with Bob
conn.sendData(conn2b, [ge_data, ge_alpha * np.transpose(ge_a), mirna_data, mirna_alpha * np.transpose(mirna_a),
	meth_data, meth_alpha * np.transpose(meth_a), cnv_data, cnv_alpha * np.transpose(cnv_a)])
[masked_bob_ge_data, masked_bob_mirna_data, masked_bob_meth_data, masked_bob_cnv_data] = conn.receiveData(conn2b)
print("Alice - Masked data of Bob is received.")

# communicate with Charlie
conn.sendData(conn2c, [ge_data, ge_alpha * np.transpose(ge_a), mirna_data, mirna_alpha * np.transpose(mirna_a),
	meth_data, meth_alpha * np.transpose(meth_a), cnv_data, cnv_alpha * np.transpose(cnv_a)])
[masked_charlie_ge_data, masked_charlie_mirna_data, masked_charlie_meth_data, masked_charlie_cnv_data] = conn.receiveData(conn2c)
print("Alice - Masked data of Alice is received.")

# Alice-Bob
ab_ge = np.transpose(ge_a) @ masked_bob_ge_data
ab_mirna = np.transpose(mirna_a) @ masked_bob_mirna_data
ab_meth = np.transpose(meth_a) @ masked_bob_meth_data
ab_cnv = np.transpose(cnv_a) @ masked_bob_cnv_data

# Alice-Charlie
ac_ge = np.transpose(ge_a) @ masked_charlie_ge_data
ac_mirna = np.transpose(mirna_a) @ masked_charlie_mirna_data
ac_meth = np.transpose(meth_a) @ masked_charlie_meth_data
ac_cnv = np.transpose(cnv_a) @ masked_charlie_cnv_data

print("Now the server time!")
conn.sendData(conn2s, [dp_ge, ab_ge, ac_ge, ge_alpha,
					dp_mirna, ab_mirna, ac_mirna, mirna_alpha,
					dp_meth, ab_meth, ac_meth, meth_alpha,
					dp_cnv, ab_cnv, ac_cnv, cnv_alpha,
					clinical_data])

end_t1 = time.time()

# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Alice whole time: %s seconds" % whole_time)
f = open(result_folder + "time_alice_run_{:.0f}.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Alice finished!")











#atadam
