import numpy as np
import pdb;

def generate_random_encoding(x, y, offline, sign, current_r):
	if (len(offline) != len(sign)) or (x.shape[0] != len(sign)) or (y.shape[0] != len(sign)):
		raise ValueError("generate_random_encoding: The size of encoding and sign vectors must be the same! {}-{}-{}-{}".format(x.shape[0],y.shape[0],len(offline),len(sign)))
	# pdb.set_trace()
	dim = len(offline)
	if dim == 1:
		# base case - multiplication node will be handled here
		x[0] = np.array([0, current_r+1, current_r, current_r, current_r+1, current_r+2])
		y[0] = np.array([0, current_r, current_r+1, current_r+3])
		offline[0].extend([current_r+2, current_r+3])
		sign[0].extend([-1,-1])
		current_r += 4
		return current_r

	# find the largest 2's power smaller than the length of current encoding lists/arrays
	max_bin = 2**(len("{0:b}".format(dim-1))-1)

	## addition nodes will be handled here
	# +r
	offline[0].append(current_r)
	sign[0].append(1)

	# -r
	offline[max_bin].append(current_r)
	sign[max_bin].append(-1)

	current_r += 1

	# recursive call
	updated_r = generate_random_encoding(x[0:max_bin,], y[0:max_bin,], offline[0:max_bin], sign[0:max_bin], current_r)
	updated_r = generate_random_encoding(x[max_bin:,], y[max_bin:,], offline[max_bin:], sign[max_bin:], updated_r)
	return updated_r

def main():
	dim = 924
	enc_x = np.zeros((dim, 6), dtype=int) # the order is: 1, r2, r1, r1, r2, r3
	enc_y = np.zeros((dim, 4), dtype=int) # the order is: 1, r1, r2, r4
	enc_offline = [[] for i in range(dim)]
	offline_sign = [[] for i in range(dim)]
	# pdb.set_trace()
	latest_r = generate_random_encoding(enc_x, enc_y, enc_offline, offline_sign, 1)
	# pdb.set_trace()

	x = np.random.randint(0, 10, (dim,))
	y = np.random.randint(0, 10, (dim,))

	print(x @ y)

	# generate random values for encoding - note that the first element of the random values is 1
	random_values = np.random.randint(0, 100, (latest_r,))
	random_values[0] = 1

	# encode
	print(enc_x[:,0])
	pdb.set_trace()
	c1 = x * random_values[enc_x[:,0]] - random_values[enc_x[:,2]]
	c2 = x * random_values[enc_x[:,1]] - (random_values[enc_x[:,3]] * random_values[enc_x[:,4]]) + random_values[enc_x[:,5]]
	c3 = y * random_values[enc_y[:,0]] - random_values[enc_y[:,2]]
	c4 = y * random_values[enc_y[:,1]] + random_values[enc_y[:,3]]
	c5 = np.zeros(c1.shape)
	for i in range(c5.shape[0]):
		c5[i] = np.sum(random_values[np.asarray(enc_offline[i])] * np.asarray(offline_sign[i]))

	# decode
	dp = np.sum(c1 * c3 + c2 + c4 + c5)
	pdb.set_trace()
	print(dp)
	# print(x)

	print("Latest r: {}".format(latest_r))

if __name__ == "__main__":
	main()
