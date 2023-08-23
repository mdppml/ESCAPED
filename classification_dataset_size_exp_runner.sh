#!/bin/bash

# Optimum parameters that we found by 5-fold cross-validation
gammas=(0.12 0.03 0.03 0.06 0.06 0.06 0.12 0.06 0.06 0.03)
cs=(0.50 0.25 1.00 0.50 1.00 0.25 0.25 0.50 0.50 1.00)
w1s=(4 4 2 4 2 4 4 2 4 2)

at=0
port_base=$((10000+at))
# The number of input-parties in the experiment specified by the user
num_of_parties=3
exp_type=2
dataset_size=$1 #it can be "full", "half" or "quarter"

# Here, we set the repetition of the experiment to 1 for simplicity.
for exp_no in {0..0}
do
	mkdir -p "results/hiv_coreceptor_prediction/dataset_size/$dataset_size/run$((exp_no+1))"
	python server.py $((exp_no+1)) ${gammas[${exp_no}]} ${cs[${exp_no}]} ${w1s[${exp_no}]} ${port_base} ${num_of_parties} ${exp_type} ${dataset_size} &
	echo python server.py $((exp_no+1)) ${gammas[${exp_no}]} ${cs[${exp_no}]} ${w1s[${exp_no}]} ${port_base} ${num_of_parties} ${exp_type} ${dataset_size} 
	sleep 2
	for (( id=1; id<$num_of_parties; id++ ))
	do
		python client.py $((exp_no+1)) ${port_base} ${num_of_parties} ${id} ${exp_type} ${dataset_size} &
		echo python client.py $((exp_no+1)) ${port_base} ${num_of_parties} ${id} ${exp_type} ${dataset_size}
		sleep 2
	done
	python client.py $((exp_no+1)) ${port_base} ${num_of_parties} ${num_of_parties} ${exp_type} ${dataset_size} &
	echo python client.py $((exp_no+1)) ${port_base} ${num_of_parties} ${num_of_parties} ${exp_type} ${dataset_size}
	wait
done
