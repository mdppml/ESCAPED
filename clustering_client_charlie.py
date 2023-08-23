import numpy as np
import pandas as pd
import sys, time, pdb
import connection as c

run = int(sys.argv[1])
port_ac = int(sys.argv[2])
port_bc = int(sys.argv[3])
port_cs = int(sys.argv[4])
# dataset = sys.argv[5]

a_max = 50
conn = c.Connection() # for the connection between parties

################################################################################
data_folder = "data/" # base folder address storing the sequence files
result_folder = "results/clustering/"

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

################################################################################
'''
There will be different type of data - gene expression, methylation, mutation etc.
We need to apply the same procedure for each type separately
Note that we process the data in a format that each column is one sample!
'''

# gene expression data
ge_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/charlie_geneExp.csv", delimiter=','))
dp_ge = np.transpose(ge_data) @ ge_data
ge_a = np.random.uniform(1, a_max, (ge_data.shape[0], ge_data.shape[1])) # mask
masked_ge_data = ge_data - ge_a

# miRNA data
mirna_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/charlie_miRNA.csv", delimiter=','))
dp_mirna = np.transpose(mirna_data) @ mirna_data
mirna_a = np.random.uniform(1, a_max, (mirna_data.shape[0], mirna_data.shape[1])) # mask
masked_mirna_data = mirna_data - mirna_a

# methylation data
meth_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/charlie_meth.csv", delimiter=','))
dp_meth = np.transpose(meth_data) @ meth_data
meth_a = np.random.uniform(1, a_max, (meth_data.shape[0], meth_data.shape[1])) # mask
masked_meth_data = meth_data - meth_a

# copy number variation data
cnv_data = np.transpose(np.loadtxt(data_folder + "cancer_dataset/hnsc/charlie_cnv.csv", delimiter=','))
dp_cnv = np.transpose(cnv_data) @ cnv_data
cnv_a = np.random.uniform(1, a_max, (cnv_data.shape[0], cnv_data.shape[1])) # mask
masked_cnv_data = cnv_data - cnv_a

# clinical data
clinical_data = pd.read_csv(data_folder + "cancer_dataset/hnsc/charlie_processed_clinical_data.csv", index_col=0, header=0)

################################################################################
print("Charlie - Dot products are ready!")

# communication with Alice
[masked_alice_ge_data, masked_alice_ge_a, masked_alice_mirna_data, masked_alice_mirna_a,
	masked_alice_meth_data, masked_alice_meth_a, masked_alice_cnv_data, masked_alice_cnv_a] = conn.receiveData(conn2a)
conn.sendData(conn2a, [masked_ge_data, masked_mirna_data, masked_meth_data, masked_cnv_data])

# communication with Bob
[masked_bob_ge_data, masked_bob_ge_a, masked_bob_mirna_data, masked_bob_mirna_a,
	masked_bob_meth_data, masked_bob_meth_a, masked_bob_cnv_data, masked_bob_cnv_a] = conn.receiveData(conn2b)
conn.sendData(conn2b, [masked_ge_data, masked_mirna_data, masked_meth_data, masked_cnv_data])

# Alice-Bob
ac_ge = np.transpose(masked_alice_ge_data) @ ge_data
ac_ge_unmasker = masked_alice_ge_a @ ge_a
ac_mirna = np.transpose(masked_alice_mirna_data) @ mirna_data
ac_mirna_unmasker = masked_alice_mirna_a @ mirna_a
ac_meth = np.transpose(masked_alice_meth_data) @ meth_data
ac_meth_unmasker = masked_alice_meth_a @ meth_a
ac_cnv = np.transpose(masked_alice_cnv_data) @ cnv_data
ac_cnv_unmasker = masked_alice_cnv_a @ cnv_a

# Bob-Charlie
bc_ge = np.transpose(masked_bob_ge_data) @ ge_data
bc_ge_unmasker = masked_bob_ge_a @ ge_a
bc_mirna = np.transpose(masked_bob_mirna_data) @ mirna_data
bc_mirna_unmasker = masked_bob_mirna_a @ mirna_a
bc_meth = np.transpose(masked_bob_meth_data) @ meth_data
bc_meth_unmasker = masked_bob_meth_a @ meth_a
bc_cnv = np.transpose(masked_bob_cnv_data) @ cnv_data
bc_cnv_unmasker = masked_bob_cnv_a @ cnv_a

print("Now server time!")
conn.sendData(conn2s, [dp_ge, ac_ge, bc_ge, ac_ge_unmasker, bc_ge_unmasker,
					dp_mirna, ac_mirna, bc_mirna, ac_mirna_unmasker, bc_mirna_unmasker,
					dp_meth, ac_meth, bc_meth, ac_meth_unmasker, bc_meth_unmasker,
					dp_cnv, ac_cnv, bc_cnv, ac_cnv_unmasker, bc_cnv_unmasker,
					clinical_data])

end_t1 = time.time()



# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
print("Charlie whole time: %s seconds" % whole_time)
f = open(result_folder + "time_charlie_run_{:.0f}.csv".format(run), 'w')
f.write("{:.5f}".format(whole_time))
f.close()

print("Charlie finished!")























	#atadam
