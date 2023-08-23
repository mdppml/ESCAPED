import numpy as np
import math, random
import copy
from collections import defaultdict

def normalizeKernelMatrix(km):
	norm_km = np.zeros(km.shape)
	for i in range(len(km)):
		for j in range(i,len(km)):
			# print(str(km[i,j]) + "-" + str(km[i,i]) + "-" + str(km[j,j]))
			norm_km[i,j] = km[i,j] / math.sqrt(km[i,i]*km[j,j])
			norm_km[j,i] = norm_km[i,j]
	return norm_km

def sdp_rbf(km, gamma):
	rbf_km = np.empty([km.shape[0], km.shape[1]])
	for i in range(km.shape[0]):
		for j in range(km.shape[1]):
			rbf_km[i,j] = np.exp(-(km[i,i] - 2*km[i,j] + km[j,j]) * gamma)
	return rbf_km

def generate_turns(latest_id):
	real_latest_id = latest_id
	if latest_id % 2 != 0:
		latest_id += 1
	partners = np.zeros((latest_id,latest_id-1), dtype=int)

	front = 1
	tail = latest_id - 2
	n_non_dummy = latest_id - 1

	for i in range(n_non_dummy):
		partners[front-1,i] = latest_id
		partners[latest_id-1,i] = front
		for j in range(n_non_dummy//2):
			partners[(front+j) % n_non_dummy, i] = (tail-j+n_non_dummy) % n_non_dummy + 1
			partners[(tail-j+n_non_dummy) % n_non_dummy, i] = (front+j) % n_non_dummy + 1
		front = front + 1
		tail = (tail + 1) % n_non_dummy

	partners = partners % (real_latest_id + 1)
	return partners[0:real_latest_id,]
