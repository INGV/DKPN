#!/bin/bash

RND="42"
EPOCHS="100"
BATCH="32"
LR="0.0001"

# =================================================================
# =================================================================
# =================================================================


SIZEME="NANO3"
echo ""
echo "... Working with --->  ${SIZEME}"
echo ""
./ReTrain_DKPN.py -d ETHZ -s ${SIZEME} -r ${RND} \
                  -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                  -o DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

./ReTrain_PN.py -d ETHZ -s ${SIZEME} -r ${RND} \
                -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                -o PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                       -p PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                       -d ETHZ -s ${SIZEME} \
                       -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                       -o Results_ETHZ_${SIZEME}

SIZEME="NANO2"
echo ""
echo "... Working with --->  ${SIZEME}"
echo ""
./ReTrain_DKPN.py -d ETHZ -s ${SIZEME} -r ${RND} \
                  -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                  -o DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

./ReTrain_PN.py -d ETHZ -s ${SIZEME} -r ${RND} \
                -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                -o PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                       -p PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                       -d ETHZ -s ${SIZEME} \
                       -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                       -o Results_ETHZ_${SIZEME}

# SIZEME="NANO1"
# echo ""
# echo "... Working with --->  ${SIZEME}"
# echo ""
# ./ReTrain_DKPN.py -d ETHZ -s ${SIZEME} -r ${RND} \
#                   -e ${EPOCHS} -b ${BATCH} -l ${LR} \
#                   -o DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

# ./ReTrain_PN.py -d ETHZ -s ${SIZEME} -r ${RND} \
#                 -e ${EPOCHS} -b ${BATCH} -l ${LR} \
#                 -o PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

# ./LoadEvaluate_DKPN.py -k DKPN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
#                        -p PN_TrainDataSet_ETHZ_Size_${SIZEME}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
#                        -d ETHZ -s ${SIZEME} \
#                        -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
#                        -o Results_ETHZ_${SIZEME}
