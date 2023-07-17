#!/bin/bash

FOLDERS=(\
"FinalRetrain_DKPN_PN_PAPER___Rnd__17__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__36__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__50__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__142__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__234__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__777__TESTING-ULTIMATE_noDetrend" \
"FinalRetrain_DKPN_PN_PAPER___Rnd__987__TESTING-ULTIMATE_noDetrend"\
)


# =================================================================
# =================================================================
# =================================================================


source /opt/anaconda/etc/profile.d/conda.sh  # SERRA
conda activate sometools_SBupdated


# =================================================================
# =================================================================
# =================================================================

OUTFOLDER="IndividualRandom_Plots"
length=${#FOLDERS[@]}

mkdir ${OUTFOLDER}

for ((i=0; i<length; i++)); do
    folder=${FOLDERS[i]}
    echo "Working with ... ${folder}"

    cd ${folder}
    mkdir ../${OUTFOLDER}/${folder}
    #
    ./TrainDev_Curves.py
    ./PlotScores.py INSTANCE INSTANCE 02
    ./PlotScores.py INSTANCE INSTANCE 05
    ./PlotScores.py INSTANCE ETHZ 02
    ./PlotScores.py INSTANCE ETHZ 05
    cp RESULTS/*  ../${OUTFOLDER}/${folder}/
    #
    cd ..
    echo ""
done

echo "DONE"
