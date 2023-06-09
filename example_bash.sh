#!/bin/bash

./ReTrain_DKPN.py -d INSTANCE -s NANO -r 42 -e 35 -o DKPN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32

./ReTrain_PN.py -d INSTANCE -s NANO -r 42 -e 35 -o PN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32

# Indomain
./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32 \
                       -p PN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32 \
                       -d INSTANCE -s NANO -x 0.2 -y 0.2 -n 5000 -f 10 -o trained_results_INSTANCE

# CrossDomain
./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32 \
                       -p PN_TrainDataSet_INSTANCE_Size_NANO_Rnd_42_Epochs_35_LR_0.0010_Batch_32 \
                       -d ETHZ -s NANO -x 0.2 -y 0.2 -n 5000 -f 10 -o trained_results_ETHZ
