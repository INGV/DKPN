#!/bin/bash

TRAINDATA="INSTANCE"
TESTDATA_CROSSDOMAIN="ETHZ"
RND="42"
EPOCHS="20"
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


mylist=("NANO3" "NANO2" "NANO1" "NANO" "MICRO" "TINY" "SMALL" "MEDIUM" "LARGE")

# Loop over the array
for DATASIZE in "${mylist[@]}"; do

  echo ""
  echo ""
  echo "... Working with ---> ${TRAINDATA}  /  ${DATASIZE}"
  echo ""
  ./ReTrain_DKPN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                    -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                    -o DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

  ./ReTrain_PN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
                  -e ${EPOCHS} -b ${BATCH} -l ${LR} \
                  -o PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH}

  # --- In-Domain  TEST
  ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -d ${TRAINDATA} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}

  # --- Cross-Domain  TEST
  ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_Epochs_${EPOCHS}_LR_${LR}_Batch_${BATCH} \
                         -d ${TESTDATA_CROSSDOMAIN} -s ${DATASIZE} \
                         -x 0.2 -y 0.2 -n 5000 -f 10 -a 10 -b 20 \
                         -o Results_${TRAINDATA}_${TESTDATA_CROSSDOMAIN}_${DATASIZE}

done
