import numpy as np
import pandas as pd
import sys, time, pdb
import connection as c

run = int(sys.argv[1])
port_ab = int(sys.argv[2])
port_bc = int(sys.argv[3])
port_bs = int(sys.argv[4])
# dataset = sys.argv[5]

a_max = 50
conn = c.Connection() # for the connection between parties

################################################################################
data_folder = "data/" # base folder address storing the sequence files
result_folder = "results/clustering/"

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

################################################################################
'''
There will be different type of data - gene expression, methylation, mutation etc.
We need to apply the same procedure for each type separately
Note that we process the data in a format that each column is one sample!
'''

# gene expression data
ge_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/bob_geneExp.csv", delimiter=','))
dp_ge = np.transpose(ge_data) @ ge_data
ge_a = np.random.uniform(1, a_max, (ge_data.shape[0], ge_data.shape[1])) # mask
masked_ge_data = ge_data - ge_a
ge_alpha = np.random.uniform(1, a_max, (1))

# miRNA data
mirna_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/bob_miRNA.csv", delimiter=','))
dp_mirna = np.transpose(mirna_data) @ mirna_data
mirna_a = np.random.uniform(1, a_max, (mirna_data.shape[0], mirna_data.shape[1])) # mask
masked_mirna_data = mirna_data - mirna_a
mirna_alpha = np.random.uniform(1, a_max, (1))

# methylation data
meth_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/bob_meth.csv", delimiter=','))
dp_meth = np.transpose(meth_data) @ meth_data
meth_a = np.random.uniform(1, a_max, (meth_data.shape[0], meth_data.shape[1])) # mask
masked_meth_data = meth_data - meth_a
meth_alpha = np.random.uniform(1, a_max, (1))

# copy number variation data
cnv_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/bob_cnv.csv", delimiter=','))
dp_cnv = np.transpose(cnv_data) @ cnv_data
cnv_a = np.random.uniform(1, a_max, (cnv_data.shape[0], cnv_data.shape[1])) # mask
masked_cnv_data = cnv_data - cnv_a
cnv_alpha = np.random.uniform(1, a_max, (1))

# clinical data
clinical_data = pd.read_csv(data_folder + "cancer_dataset/hnsc/bob_processed_clinical_data.csv", index_col=0, header=0)

################################################################################
print("Bob - Dot products are ready!")

# communication with Alice
[masked_alice_ge_data, masked_alice_ge_a, masked_alice_mirna_data, masked_alice_mirna_a,
	masked_alice_meth_data, masked_alice_meth_a, masked_alice_cnv_data, masked_alice_cnv_a] = conn.receiveData(conn2a)
conn.sendData(conn2a, [masked_ge_data, masked_mirna_data, masked_meth_data, masked_cnv_data])

# communication with Charlie
conn.sendData(conn2c, [masked_ge_data, ge_alpha * np.transpose(ge_a), masked_mirna_data, mirna_alpha * np.transpose(mirna_a),
	masked_meth_data, meth_alpha * np.transpose(meth_a), masked_cnv_data, cnv_alpha * np.transpose(cnv_a)])
[masked_charlie_ge_data, masked_charlie_mirna_data, masked_charlie_meth_data, masked_charlie_cnv_data] = conn.receiveData(conn2c)

# Alice-Bob
ab_ge = np.transpose(masked_alice_ge_data) @ ge_data
ab_ge_unmasker = masked_alice_ge_a @ ge_a
ab_mirna = np.transpose(masked_alice_mirna_data) @ mirna_data
ab_mirna_unmasker = masked_alice_mirna_a @ mirna_a
ab_meth = np.transpose(masked_alice_meth_data) @ meth_data
ab_meth_unmasker = masked_alice_meth_a @ meth_a
ab_cnv = np.transpose(masked_alice_cnv_data) @ cnv_data
ab_cnv_unmasker = masked_alice_cnv_a @ cnv_a

# Bob-Charlie
bc_ge = np.transpose(ge_a) @ masked_charlie_ge_data
bc_mirna = np.transpose(mirna_a) @ masked_charlie_mirna_data
bc_meth = np.transpose(meth_a) @ masked_charlie_meth_data
bc_cnv = np.transpose(cnv_a) @ masked_charlie_cnv_data

print("Now server time!")
conn.sendData(conn2s, [dp_ge, ab_ge, bc_ge, ab_ge_unmasker, ge_alpha,
					dp_mirna, ab_mirna, bc_mirna, ab_mirna_unmasker, mirna_alpha,
					dp_meth, ab_meth, bc_meth, ab_meth_unmasker, meth_alpha,
					dp_cnv, ab_cnv, bc_cnv, ab_cnv_unmasker, cnv_alpha,
					clinical_data])

end_t1 = time.time()


# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Bob whole time: %s seconds" % whole_time)
f = open(result_folder + "time_bob_run_{:.0f}.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Bob finished!")














#atadam
