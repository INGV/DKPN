SO:

In TestSituation,  all experiments ETH/INSTANCE/PNW use randomseed=42
for the trace selection of the TEST-DATASET.

The random seed number specified indicates the model TRAINED with that
random selection --> i.e. Rnd_36 indicates the model has been trained selecting
randomly a TRAIN dataset using the seed=36. The testing - and relative
TEST dataset selection- of this model against ETH/INSTANCE/PNW is ALWAYS using
the seed=42.

This allow some certain stability in testing (using always the same traces)
