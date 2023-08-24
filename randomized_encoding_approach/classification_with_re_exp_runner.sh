#!/bin/bash

# Optimum parameters that we found by 5-fold cross-validation
gammas=(0.12 0.03 0.03 0.06 0.06 0.06 0.12 0.06 0.06 0.03)
cs=(0.50 0.25 1.00 0.50 1.00 0.25 0.25 0.50 0.50 1.00)
w1s=(4 4 2 4 2 4 4 2 4 2)

base=0
at=50
port_as=$((10034+at))
port_bs=$((10035+at))
port_cs=$((10036+at))
port_ab=$((10037+at))
port_ac=$((10038+at))
port_bc=$((10039+at))
dataset_size=$1 #it can be "full", "half" or "quarter"

# Here, we set the repetition of the experiment to 1 for simplicity.
for exp_no in {0..0}
do
	mkdir -p "results/$dataset_size/run$((exp_no+1))"
	python server.py $((exp_no+1)) ${gammas[$((exp_no-$base))]} ${cs[$((exp_no-$base))]} ${w1s[$((exp_no-$base))]} ${port_as} ${port_bs} ${port_cs} ${dataset_size}&
	python client_charlie.py $((exp_no+1)) ${port_ac} ${port_bc} ${port_cs} ${dataset_size}&
	python client_bob.py $((exp_no+1)) ${port_ab} ${port_bc} ${port_bs} ${dataset_size}&
	python client_alice.py $((exp_no+1)) ${port_ab} ${port_ac} ${port_as} ${dataset_size}
	wait
done
