#!/usr/bin/env python

import os
import pickle
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from pathlib import Path

import obspy
import seisbench as sb
import seisbench.models as sbm

import dkpn.train as dktrain


print(" SB version:  %s" % sb.__version__)
print("OBS version:  %s" % obspy.__version__)
print("")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=(
                                "Script for training PHASE-NET using full SeisBench APIs and Augmentations. "
                                "It needs to have the 'dkpn' folder in the working path. "
                                "Requires Python >= 3.9"))

parser.add_argument('-d', '--dataset_name', type=str, default='ETHZ', help='Dataset name')
parser.add_argument('-s', '--dataset_size', type=str, default='Nano', help='Dataset size')
parser.add_argument('-r', '--random_seed', type=int, default=42, help='Random seed')
parser.add_argument('-o', '--store_folder', type=str, default=None, help='Store folder')
#
parser.add_argument('-e', '--epochs', type=int, default=25, help='Max. Num. Epochs for training')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning Rate for training')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch-Size for training')
#
parser.add_argument("--early_stop", action="store_true", help="Adopt early-stop regulation for epochs")
parser.add_argument('-x', '--patience', type=int, default=5, help='Num. Epochs to evaluate for early stop')
parser.add_argument('-y', '--delta', type=float, default=0.001, help='Mean dev_loss improvement over the latest patience epochs')
#
args = parser.parse_args()

print("---> Training: PhaseNet")
print("")
print(f"DATASET_NAME: {args.dataset_name}")
print(f"DATASET_SIZE: {args.dataset_size}")
print(f"RANDOM_SEED: {args.random_seed}")
print(f"STORE_FOLDER: {args.store_folder}")
print("")
print(f"MAX. EPOCHS: {args.epochs}")
print(f"LEARNING_RATE: {args.learning_rate}")
print(f"BATCH_SIZE: {args.batch_size}")
print("")
print(f"EARLY STOP: {args.early_stop}")
print(f"  PATIENCE: {args.patience}")
print(f"     DELTA: {args.delta}")
print("")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# SELECT DATASET and SIZE

dataset_train = dktrain.select_database_and_size(
                            args.dataset_name, args.dataset_size,
                            RANDOM_SEED=args.random_seed)

train, dev, test = dataset_train.train_dev_test()

print("TRAIN samples %s:  %d" % (args.dataset_name, len(train)))
print("  DEV samples %s:  %d" % (args.dataset_name, len(dev)))
print(" TEST samples %s:  %d" % (args.dataset_name, len(test)))

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# INITIALIZE PN

mypn = sbm.PhaseNet()
mypn.cuda();

TRAIN_CLASS = dktrain.TrainHelp_PhaseNet(
                mypn,  # It will contains the default args for StreamCF calculations!!!
                train,
                dev,
                test,
                augmentations_par={
                    "amp_norm_type": "std",
                    "window_strategy": "move",  # "pad"
                    "final_windowlength": 3001,
                    "sigma": 10,
                    "fp_stabilization": 400,
                    "phase_dict": {
                        "trace_p_arrival_sample": "P",
                        "trace_pP_arrival_sample": "P",
                        "trace_P_arrival_sample": "P",
                        "trace_P1_arrival_sample": "P",
                        "trace_Pg_arrival_sample": "P",
                        "trace_Pn_arrival_sample": "P",
                        "trace_PmP_arrival_sample": "P",
                        "trace_pwP_arrival_sample": "P",
                        "trace_pwPm_arrival_sample": "P",
                        "trace_s_arrival_sample": "S",
                        "trace_S_arrival_sample": "S",
                        "trace_S1_arrival_sample": "S",
                        "trace_Sg_arrival_sample": "S",
                        "trace_SmS_arrival_sample": "S",
                        "trace_Sn_arrival_sample": "S"
                    },
                },
                batch_size=args.batch_size,
                num_workers=24,
                random_seed=args.random_seed,
)


# ----------------------------------------------------------------------------
# --------------->    ACTUAL TRAINING  <---------------

if args.early_stop:
    (train_loss_epochs, train_loss_epochs_batches,
     dev_loss_epochs, dev_loss_epochs_batches) = TRAIN_CLASS.train_me_early_stop(
        epochs=args.epochs, optimizer_type="adam",
        learning_rate=args.learning_rate,
        patience=args.patience, delta=args.delta)

else:
    (train_loss_epochs, train_loss_epochs_batches,
     dev_loss_epochs, dev_loss_epochs_batches) = TRAIN_CLASS.train_me(
        epochs=args.epochs, optimizer_type="adam",
        learning_rate=args.learning_rate)

# ----------------------------------------------------------------------------
# --------------->    STORE   MODEL    <---------------

_actual_epochs = TRAIN_CLASS.__training_epochs__
MODEL_NAME = "PN_TrainDataset_%s_Size_%s_Rnd_%d_Epochs_%d_LR_%06.4f_Batch_%d" % (
                    args.dataset_name, args.dataset_size, args.random_seed,
                    _actual_epochs, args.learning_rate, args.batch_size)

if not args.store_folder:
    STORE_DIR_MODEL = Path(MODEL_NAME)
else:
    STORE_DIR_MODEL = Path(args.store_folder)
#
if not STORE_DIR_MODEL.is_dir():
    STORE_DIR_MODEL.mkdir(parents=True, exist_ok=True)

TRAIN_CLASS.store_weigths(STORE_DIR_MODEL, MODEL_NAME, MODEL_NAME, version="1")

# ----------------------------------------------------------------------------
# --------------->    STORE   LOSS  TABLE    <---------------

with open(str(STORE_DIR_MODEL / "TRAIN_TEST_loss.csv"), "w") as OUT:
    OUT.write("EPOCH, TRAIN_LOSS, TEST_LOSS"+os.linesep)
    for xx, (trn, tst) in enumerate(zip(train_loss_epochs, dev_loss_epochs)):
        OUT.write(("%d, %.4f, %.4f"+os.linesep) % (xx, trn, tst))

# ----------------------------------------------------------------------------
# --------------->    STORE   LOSS  PICKLE    <---------------

TRAIN_LOSSES = []
for xx, (av_loss, batch_loss) in enumerate(zip(train_loss_epochs, train_loss_epochs_batches)):
    TRAIN_LOSSES.append((av_loss, batch_loss))
with open(str(STORE_DIR_MODEL / 'TRAIN_loss_batches.pickle'), 'wb') as file:
    pickle.dump(TRAIN_LOSSES, file)

DEV_LOSSES = []
for xx, (av_loss, batch_loss) in enumerate(zip(dev_loss_epochs, dev_loss_epochs_batches)):
    DEV_LOSSES.append((av_loss, batch_loss))
with open(str(STORE_DIR_MODEL / 'DEV_loss_batches.pickle'), 'wb') as file:
    pickle.dump(DEV_LOSSES, file)

# ----------------------------------------------------------------------------
# --------------->    STORE   PARAMETERS    <---------------

with open(str(STORE_DIR_MODEL / "TRAIN_ARGS.py"), "w") as OUT:
    OUT.write("TRAINARGS=%s" % args)

# Store DATABASE INFO
with open(str(STORE_DIR_MODEL / "TRAIN_DATA_INFO.txt"), "w") as OUT:
    OUT.write(("TRAIN samples %s:  %d"+os.linesep) % (args.dataset_name,
                                                      len(train)))
    OUT.write(("  DEV samples %s:  %d"+os.linesep) % (args.dataset_name,
                                                      len(dev)))
    OUT.write((" TEST samples %s:  %d"+os.linesep) % (args.dataset_name,
                                                      len(test)))

# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
plt.plot(train_loss_epochs, label="TRAIN_Loss", color="red", lw=2)
plt.plot(dev_loss_epochs, label="DEV_Loss", color="teal", lw=2)
plt.xlabel("epochs")
plt.ylabel("cross-entropy loss")
plt.legend()
plt.tight_layout()
fig.savefig(str(STORE_DIR_MODEL / "TrainTest_LOSS.pdf"))
