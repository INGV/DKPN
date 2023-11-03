#!/bin/bash

FOLDERS=("Rnd_17" "Rnd_36" "Rnd_50" "Rnd_142" "Rnd_234" "Rnd_777" "Rnd_987")

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

