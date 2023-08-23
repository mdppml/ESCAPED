#!/bin/bash

# Alice
mv alice_enc_data_training_full.txt tr_data_1_full.txt
mv alice_enc_data_test_full.txt test_data_1_full.txt
mv alice_label_training_full.txt tr_label_1_full.txt
mv alice_label_test_full.txt test_label_1_full.txt

alice_tr_nl=$(wc -l < tr_data_1_full.txt)
alice_ts_nl=$(wc -l < test_data_1_full.txt)
alice_tr_lbl_nl=$(wc -l < tr_label_1_full.txt)
alice_ts_lbl_nl=$(wc -l < test_label_1_full.txt)

head -n $((alice_tr_nl/2)) tr_data_1_full.txt > tr_data_1_half.txt
head -n $((alice_ts_nl/2)) test_data_1_full.txt > test_data_1_half.txt
head -n $((alice_tr_lbl_nl/2)) tr_label_1_full.txt > tr_label_1_half.txt
head -n $((alice_ts_lbl_nl/2)) test_label_1_full.txt > test_label_1_half.txt

head -n $((alice_tr_nl/4)) tr_data_1_full.txt > tr_data_1_quarter.txt
head -n $((alice_ts_nl/4)) test_data_1_full.txt > test_data_1__quarter.txt
head -n $((alice_tr_lbl_nl/4)) tr_label_1_full.txt > tr_label_1_quarter.txt
head -n $((alice_ts_lbl_nl/4)) test_label_1_full.txt > test_label_1_quarter.txt

# Bob
mv bob_enc_data_training_full.txt tr_data_2_full.txt
mv bob_enc_data_test_full.txt test_data_2_full.txt
mv bob_label_training_full.txt tr_label_2_full.txt
mv bob_label_test_full.txt test_label_2_full.txt

bob_tr_nl=$(wc -l < tr_data_2_full.txt)
bob_ts_nl=$(wc -l < test_data_2_full.txt)
bob_tr_lbl_nl=$(wc -l < tr_label_2_full.txt)
bob_ts_lbl_nl=$(wc -l < test_label_2_full.txt)

head -n $((bob_tr_nl/2)) tr_data_2_full.txt > tr_data_2_half.txt
head -n $((bob_ts_nl/2)) test_data_2_full.txt > test_data_2_half.txt
head -n $((bob_tr_lbl_nl/2)) tr_label_2_full.txt > tr_label_2_half.txt
head -n $((bob_ts_lbl_nl/2)) test_label_2_full.txt > test_label_2_half.txt

head -n $((bob_tr_nl/4)) tr_data_2_full.txt > tr_data_2_quarter.txt
head -n $((bob_ts_nl/4)) test_data_2_full.txt > test_data_2_quarter.txt
head -n $((bob_tr_lbl_nl/4)) tr_label_2_full.txt > tr_label_2_quarter.txt
head -n $((bob_ts_lbl_nl/4)) test_label_2_full.txt > test_label_2_quarter.txt

# Charlie
mv charlie_enc_data_training_full.txt tr_data_3_full.txt
mv charlie_enc_data_test_full.txt test_data_3_full.txt
mv charlie_label_training_full.txt tr_label_3_full.txt
mv charlie_label_test_full.txt test_label_3_full.txt

charlie_tr_nl=$(wc -l < tr_data_3_full.txt)
charlie_ts_nl=$(wc -l < test_data_3_full.txt)
charlie_tr_lbl_nl=$(wc -l < tr_label_3_full.txt)
charlie_ts_lbl_nl=$(wc -l < test_label_3_full.txt)

head -n $((charlie_tr_nl/2)) tr_data_3_full.txt > tr_data_3_half.txt
head -n $((charlie_ts_nl/2)) test_data_3_full.txt > test_data_3_half.txt
head -n $((charlie_tr_lbl_nl/2)) tr_label_3_full.txt > tr_label_3_half.txt
head -n $((charlie_ts_lbl_nl/2)) test_label_3_full.txt > test_label_3_half.txt

head -n $((charlie_tr_nl/4)) tr_data_3_full.txt > tr_data_3_quarter.txt
head -n $((charlie_ts_nl/4)) test_data_3_full.txt > test_data_3_quarter.txt
head -n $((charlie_tr_lbl_nl/4)) tr_label_3_full.txt > tr_label_3_quarter.txt
head -n $((charlie_ts_lbl_nl/4)) test_label_3_full.txt > test_label_3_quarter.txt
