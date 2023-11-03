#!/bin/bash

#FOLDERS=(\
#"FinalRetrain_DKPN_PN_PAPER___Rnd__17__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__36__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__50__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__142__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__234__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__777__TESTING-ULTIMATE_noDetrend" \
#"FinalRetrain_DKPN_PN_PAPER___Rnd__987__TESTING-ULTIMATE_noDetrend"\
#)

FOLDERS=("Rnd_17" "Rnd_36" "Rnd_50" "Rnd_142" "Rnd_234" "Rnd_987")


length=${#FOLDERS[@]}
for ((i=0; i<length; i++)); do
    folder=${FOLDERS[i]}
    echo ${folder}
    echo "Working with ... ${folder}"
    cd ${folder} ; ./TRAIN_TEST_loop_F1_THRESHOLD.sh > log_train_test_f1_opt.txt ; cd ..
done

echo "DONE"

