#!/bin/bash

SIZEME="NANO3"

./ReTrain_DKPN.py -d ETHZ -s ${SIZEME} -r 42 \
                  -e 50 -b 32 -l 0.0001 \
                  -o DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_42_Epochs_50_LR_0.0001_Batch_32

./ReTrain_PN.py -d ETHZ -s NANO3 -r 42 \
                -e 50 -b 32 -l 0.0001 \
                -o PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_42_Epochs_50_LR_0.0001_Batch_32

./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_42_Epochs_50_LR_0.0001_Batch_32 \
                       -p PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_42_Epochs_50_LR_0.0001_Batch_32 \
                       -d ETHZ -s ${SIZEME} -x 0.2 -y 0.2 -n 5000 -f 10 -o Results_ETHZ_${SIZEME}

