#!/bin/bash

TRAINDATA="INSTANCE"
TESTDATA_INDOMAIN="INSTANCE"
TESTDATA_CROSSDOMAIN="ETHZ"

RND="42"
BATCH="128"
LR="0.0100"   # Must be a %.04f FORMAT!

# =================================================================
# =================================================================
# =================================================================


source /opt/anaconda/etc/profile.d/conda.sh  # SERRA
conda activate sometools_SBupdated


# =================================================================
# =================================================================
# =================================================================


# SIZES=("NANO3" "NANO2" "NANO1" "NANO" "MICRO" "TINY" "SMALL" "MEDIUM" "LARGE")
# EPOCHS=("20" "20" "20" "20" "20" "20" "20" "20" "20")

SIZES=("NANO3" "NANO2" "NANO1" "NANO" "MICRO" "TINY")  # ETHZ
EPOCHS=("120" "85" "60" "23" "20" "12")                # ETHZ

# Loop over the array
length=${#SIZES[@]}  # Get the length of the array

for ((i=0; i<length; i++)); do
  DATASIZE=${SIZES[i]}
  EPOCHSNUM=${EPOCHS[i]}

  echo ""
  echo ""
  echo "... Working with ---> ${TRAINDATA}  /  ${DATASIZE} - ${EPOCHSNUM} Epochs"
  echo ""
  ./ReTrain_DKPN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                    -e ${EPOCHSNUM} -b ${BATCH} -l ${LR} \
                    -o DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

  ./ReTrain_PN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                  -e ${EPOCHSNUM} -b ${BATCH} -l ${LR} \
                  -o PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

  # --- In-Domain  TEST
  ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_INDOMAIN} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}

  # --- Cross-Domain  TEST
  ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_CROSSDOMAIN} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TESTDATA_CROSSDOMAIN}_${DATASIZE}

done
