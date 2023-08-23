import numpy as np
import sys, time, os
import connection as c
import common_functions as comm_func

run = int(sys.argv[1])
base_port = int(sys.argv[2])
num_of_parties = int(sys.argv[3])
party_id = int(sys.argv[4])
exp_type = int(sys.argv[5]) # 1 indicates the varying number of input-parties -- 2 indicates the varying size of the dataset

# checks
if exp_type != 1 and exp_type != 2:
	sys.exit("Wrong experiment type is given. 1 for varying number of input-parties and 2 for varying dataset size.")

if exp_type == 2:
	if num_of_parties != 3:
		sys.exit("The varying dataset size experiment is available for only 3 input-parties.")
	dataset_size = sys.argv[6]
	if dataset_size != "full" and dataset_size != "half" and dataset_size != "quarter":
		sys.exit("The dataset size can be \"full\", \"half\" or \"quarter\".")

# print(f'Party {party_id}: Setup:')
# print(f'Party {party_id}: Run: {run}')
# print(f'Party {party_id}: Base port: {base_port}')
# print(f'Party {party_id}: Number of parties: {num_of_parties}')
# print(f'Party {party_id}: Party ID: {party_id}')
# print(f'Party {party_id}: Experiment type: {exp_type}')
# print(f'Party {party_id}:- Dataset size: {dataset_size:{1}.{0}}')

# print(f'Party {party_id}: Initial step is done!')

a_max = 50
conn = c.Connection() # for the communication between parties

data_folder = "data/hiv_v3_loop_seq/" # base folder address storing the sequence files
result_folder = "results/hiv_coreceptor_prediction/"

partners = comm_func.generate_turns(num_of_parties)[party_id-1]
# print(f'Party {party_id}: Partners: {partners}')
conns = np.empty([num_of_parties + 1,], dtype=object)

# print(f'Party {party_id}: Starting connections...')

# The connection order follows the order of partners - first partner's connection is the first connection
for i in range(len(partners)):
	if partners[i] != 0:
		if partners[i] > party_id: # the party creates the communication channel
			# print(f'Party {party_id}: Open a connection for Party {partners[i]} to connect on port '
			# 	  f'{base_port + (num_of_parties * party_id + partners[i])} as a host')
			(tmp, conns[i], tmp) = conn.startConnection('localhost', (base_port + (num_of_parties * party_id + partners[i])))
		elif partners[i] < party_id:
			# print(f'Party {party_id}: Try to connect Party {partners[i]} on port '
			# 	  f'{base_port + (num_of_parties * partners[i] + party_id)} as a client...')
			conns[i] = conn.connectHost('localhost', (base_port + (num_of_parties * partners[i] + party_id)))

# print(f'Party {party_id}: Connections are established!')

start_t1 = time.time() # start time after the connections are ready

if exp_type == 1:
	a_training_data = np.loadtxt(f'{data_folder}number_of_input-parties_exp_data/tr_data_{party_id}.txt', delimiter=',')
	a_test_data = np.loadtxt(f'{data_folder}number_of_input-parties_exp_data/test_data_{party_id}.txt', delimiter=',')
	a_label = np.loadtxt(f'{data_folder}number_of_input-parties_exp_data/tr_label_{party_id}.txt')
	a_test_label = np.loadtxt(f'{data_folder}number_of_input-parties_exp_data/test_label_{party_id}.txt')
else:
	a_training_data = np.loadtxt(f'{data_folder}dataset_size_exp_data/tr_data_{party_id}_{dataset_size}.txt', delimiter=',')
	a_test_data = np.loadtxt(f'{data_folder}dataset_size_exp_data/test_data_{party_id}_{dataset_size}.txt', delimiter=',')
	a_label = np.loadtxt(f'{data_folder}dataset_size_exp_data/tr_label_{party_id}_{dataset_size}.txt')
	a_test_label = np.loadtxt(f'{data_folder}dataset_size_exp_data/test_label_{party_id}_{dataset_size}.txt')

# print(f'Party {party_id}: Data is loaded')
# NOTE: data should be n_features-by-n_samples
data = np.transpose(np.concatenate((a_training_data, a_test_data), axis=0))
label = np.concatenate((a_label, a_test_label), axis=0)

n_features = data.shape[0] # number of features
n_samples = np.array([a_training_data.shape[0], a_test_data.shape[0]])

dp = np.transpose(data) @ data

a = np.random.uniform(1, a_max, (n_features, np.sum(n_samples))) # mask
alpha = np.random.uniform(1, a_max, (1))

masked_data = data - a # masked the original data by subtracting randomly generated vector from each sample

del a_training_data
del a_test_data
del a_label
del a_test_label

components = [] # the components for the dot product computation
unmaskers = [] # the unmasker part in which we do not have anything related to the input data

# print(f'Party {party_id}: Ready for sending/receiving among input-parties!')

start_among_ip = time.time()
# send/receive data to/from other input-parties
for i in range(len(partners)):
	if partners[i] != 0:
		if partners[i] > party_id:
			conn.sendData(conns[i], [masked_data, alpha * np.transpose(a)])
			received_masked_data = conn.receiveData(conns[i])
			components.append(np.transpose(a) @ received_masked_data)
			unmaskers.append((1.0 / alpha))
		elif partners[i] < party_id:
			received_masked_data, partial_unmasker = conn.receiveData(conns[i])
			conn.sendData(conns[i], masked_data)
			components.append(np.transpose(received_masked_data) @ data)
			unmaskers.append(partial_unmasker @ a)
		# print("Party {:.0f}: The data transfer with party {:.0f} is completed".format(party_id, partners[i]))
	else:
		components.append(-1)
		unmaskers.append(-1)
end_among_ip = time.time()
del masked_data

# print(f'Party {party_id}: Ready for sending data to the server!')

base_port += num_of_parties**2
conn2s = conn.connectHost('localhost', (base_port + party_id))
start_with_server = time.time()
conn.sendData(conn2s, [components, unmaskers, n_samples, dp, label])
end_with_server = time.time()

# print(f'Party {party_id}: Masked data has been sent to the server!')

end_t1 = time.time()

# calculate the execution times and report/save it
whole_time = (end_t1 - start_t1)
among_input_parties = end_among_ip - start_among_ip
with_server = end_with_server - start_with_server
print(f'Party {party_id}: Input-party {party_id} whole time: {whole_time} seconds')

f_path = ''
if exp_type == 1:
	f_path = f'{result_folder}num_of_input-parties/{num_of_parties}_parties/run{run}/'
else:
	f_path = f'{result_folder}dataset_size/{dataset_size}/run{run}/'

if not os.path.exists(f_path):
	os.makedirs(f_path)

if exp_type == 1:
	f_path = f'{f_path}time_party_{party_id}.csv'
else:
	f_path = f'{f_path}time_party_{party_id}.csv'
with open(f_path, 'w') as f:
	f.write(f'{whole_time:{1}.{5}},{among_input_parties:{1}.{5}},{with_server:{1}.{5}}')

print(f'Party {party_id} finished!')
