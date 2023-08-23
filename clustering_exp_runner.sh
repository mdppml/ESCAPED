#!/bin/bash
at=0
port_as=$((10034+at))
port_bs=$((10035+at))
port_cs=$((10036+at))
port_ab=$((10037+at))
port_ac=$((10038+at))
port_bc=$((10039+at))

echo ${port_as}
echo ${port_bs}
echo ${port_cs}
echo ${port_ab}
echo ${port_ac}
echo ${port_bc}
mkdir -p "results/clustering"
mkdir -p "kms"
# Here, we set the repetition of the experiment to 1 for simplicity.
for exp_no in {0..0}
do
	echo ${exp_no}
	python clustering_server.py $((exp_no+1)) ${port_as} ${port_bs} ${port_cs} &
	python clustering_client_charlie.py $((exp_no+1)) ${port_ac} ${port_bc} ${port_cs} &
	python clustering_client_bob.py $((exp_no+1)) ${port_ab} ${port_bc} ${port_bs} &
	python clustering_client_alice.py $((exp_no+1)) ${port_ab} ${port_ac} ${port_as}
	wait
done
