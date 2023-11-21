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

SIZES=("NANO2" "MICRO" "MEDIUM")  # INSTANCE
THRESHOLDS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
TESTDATACROSSDOMAIN=("ETHZ" "PNW" "AQUILA")

# Loop over the array
length=${#SIZES[@]}  # Get the length of the array
length_thr=${#THRESHOLDS[@]}
length_test_data=${#TESTDATACROSSDOMAIN[@]}

for ((a=0; a<length; a++)); do
  DATASIZE=${SIZES[a]}

  echo ""
  echo ""
  echo "... Working with ---> ${TRAINDATA}  /  ${DATASIZE}"
  echo ""
  # echo "... Training DKPN"
  # ./ReTrain_DKPN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
  #                   -e 50 -b ${BATCH} -l ${LR} \
  #                   -o DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
  #                   --early_stop -x ${PATIENCE} -y ${IMPROVEMENT}
  # echo ""
  # echo "... Training PhaseNet"
  # ./ReTrain_PN.py -d ${TRAINDATA} -s ${DATASIZE} -r ${RND} \
  #                 -e ${EPOCHSNUM} -b ${BATCH} -l ${LR} \
  #                 -o PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
  #                 --early_stop -x ${PATIENCE} -y ${IMPROVEMENT}

  for ((b=0; b<length_thr; b++)); do
    THR=${THRESHOLDS[b]}

    echo "...    Threshold ---> ${THR}"

    # --- In-Domain  TEST
    echo ""
    echo "... Test IN-DOMAIN"
    ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                           -d ${TRAINDATA} -s ${DATASIZE} \
                           -x ${THR} -y ${THR} -n 5000 -f 100 -a 10 -b 20 \
                           -o Results_${TRAINDATA}_${TRAINDATA}_${DATASIZE}_${THR}

    for ((c=0; c<length_test_data; c++)); do
      CROSS=${TESTDATACROSSDOMAIN[c]}

      # --- Cross-Domain  TEST
      echo ""
      echo "... Test CROSS-DOMAIN"
      ./LoadEvaluate_DKPN.py -k DKPN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                             -p PN_TrainDataset_${TRAINDATA}_Size_${DATASIZE}_Rnd_${RND}_LR_${LR}_Batch_${BATCH} \
                             -d ${CROSS} -s ${DATASIZE} \
                             -x ${THR} -y ${THR} -n 5000 -f 100 -a 10 -b 20 \
                             -o Results_${TRAINDATA}_${CROSS}_${DATASIZE}_${THR}
    done
  done
done

echo ""
echo "DONE!"
