#!/bin/bash

RND="17"
BATCH="64"
LR="0.0010"   # Must be a %.04f FORMAT!

PATIENCE="5"
IMPROVEMENT="0.0005"

# =================================================================
# =================================================================
# =================================================================


source /opt/anaconda/etc/profile.d/conda.sh  # SERRA
conda activate sometools_SBupdated


# =================================================================
# =================================================================
# =================================================================


# Create a pandas dataframe from a dictionary with keys == column name and values the list containing the row value

TRAINDATA="INSTANCE"
TESTDATA_INDOMAIN="INSTANCE"
TESTDATA_CROSSDOMAIN="ETHZ"

SIZES=("NANO2" "MICRO" "MEDIUM")  # INSTANCE
THRESHOLDS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

# Loop over the array
length=${#SIZES[@]}  # Get the length of the array
length_thr=${#THRESHOLDS[@]}

for ((i=0; i<length; i++)); do
  DATASIZE=${SIZES[i]}

  echo ""
  echo ""
  echo "... Working with ---> ${TRAINDATA}  /  ${DATASIZE}"

  for ((i=0; i<length_thr; i++)); do
    THR=${THRESHOLDS[i]}

    echo "...    Threshold ---> ${THR}"

    # --- In-Domain  TEST
    echo ""
    echo "... Test IN-DOMAIN"
    ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -d ${TESTDATA_INDOMAIN} -s ${DATASIZE} \
                           -x ${THR} -y ${THR} -n 5000 -f 10 -a 10 -b 20 \
                           -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}_${THR}

    # --- Cross-Domain  TEST
    echo ""
    echo "... Test CROSS-DOMAIN"
    ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -d ${TESTDATA_CROSSDOMAIN} -s ${DATASIZE} \
                           -x ${THR} -y ${THR} -n 5000 -f 10 -a 10 -b 20 \
                           -o Results_${TRAINDATA}_${TESTDATA_CROSSDOMAIN}_${DATASIZE}_${THR}
  done

done

echo ""
echo "DONE!"
