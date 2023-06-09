#!/usr/bin/env python

import os
import argparse
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
parser.add_argument('-o', '--store_folder', type=str, default='trained_model', help='Store folder')
#
parser.add_argument('-e', '--epochs', type=int, default=25, help='Num. Epochs for training')
parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning Rate for training')
parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch-Size for training')
#
args = parser.parse_args()

# Your main function here
print(f"DATASET_NAME: {args.dataset_name}")
print(f"DATASET_SIZE: {args.dataset_size}")
print(f"RANDOM_SEED: {args.random_seed}")
print(f"STORE_FOLDER: {args.store_folder}")
print("")
print(f"EPOCHS: {args.epochs}")
print(f"LEARNING_RATE: {args.learning_rate}")
print(f"BATCH_SIZE: {args.batch_size}")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# ### SELECT DATASET and SIZE
# 
# Here you can decide to load the same dataset for _In-Domain_ tests, or a different one for _Cross-Domain_ testing.

# In[3]:


(dataset_train, dataset_test) = dktrain.select_database_and_size(
                                    args.dataset_name, args.dataset_name, args.dataset_size, RANDOM_SEED=args.random_seed)
train = dataset_train.train()
dev = dataset_train.dev()
test = dataset_test.test()

print("TRAIN samples %s:  %d" % (args.dataset_name, len(train)))
print("  DEV samples %s:  %d" % (args.dataset_name, len(dev)))
print(" TEST samples %s:  %d" % (args.dataset_name, len(test)))


MODEL_NAME = "PN_TrainDataSet_%s_Size_%s_Rnd_%d_Epochs_%d_LR_%06.4f_Batch_%d" % (
                    args.dataset_name, args.dataset_size, args.random_seed, args.epochs, args.learning_rate, args.batch_size)

STORE_DIR_MODEL = Path(MODEL_NAME)
if not STORE_DIR_MODEL.is_dir():
    STORE_DIR_MODEL.mkdir(parents=True, exist_ok=True)


# -----------------------------------
# 
# # INITIALIZE PN
# 
# In this slot we initialize the picker and prepare 

# In[4]:


mypn = sbm.PhaseNet()
mypn.cuda();


# In[5]:


TRAIN_CLASS = dktrain.TrainHelp_PhaseNet(
                mypn,  # It will contains the default args for StreamCF calculations!!!
                train,
                dev,
                test,
                augmentations_par={
                    "amp_norm_type": "std",
                    "window_strategy": "move",  # "pad"
                    "final_windowlength": 3001,
                    "fp_stabilization": 400,
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


# --------------->    ACTUAL TRAINING  <---------------

train_loss_epochs, dev_loss_epochs = TRAIN_CLASS.train_me(epochs=args.epochs, optimizer_type="adam", learning_rate=args.learning_rate)
TRAIN_CLASS.store_weigths(STORE_DIR_MODEL, MODEL_NAME, MODEL_NAME, version="1")

# Store TABLE
with open(str(STORE_DIR_MODEL / "TRAIN_TEST_loss.csv"), "w") as OUT:
    OUT.write("EPOCH, TRAIN_LOSS, TEST_LOSS"+os.linesep)
    for xx, (trn, tst) in enumerate(zip(train_loss_epochs, dev_loss_epochs)):
        OUT.write(("%d, %.4f, %.4f"+os.linesep) % (xx, trn, tst))


# In[6]:


fig = plt.figure(figsize=(10, 7))
plt.plot(train_loss_epochs, label="TRAIN_Loss", color="red", lw=2)
plt.plot(dev_loss_epochs, label="DEV_Loss", color="teal", lw=2)
plt.xlabel("epochs")
plt.ylabel("cross-entropy loss")
# plt.ylim([0,0.1])
# plt.yscale("log")
plt.legend()
fig.savefig(str(STORE_DIR_MODEL / "TrainTest_LOSS.pdf"))
