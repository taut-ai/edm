# %%

from mygenerate import *

# make and save images for each noise schedule and class

networks = [
    # "edm-cifar10-32x32-cond-ve.pkl",  # exponential noise schedule (ODE)
    # "edm-cifar10-32x32-cond-vp.pkl",  # "empircally optmized noise schedule" (SDE)
    "edm-cifar10-32x32-cond-ve.pkl",  # exponential noise schedule (ODE)
    "edm-cifar10-32x32-cond-vp.pkl",  # "empircally optmized noise schedule" (SDE)
]
nout = 64

for network in networks:
    for class_ in range(10):
        out = f"out_{class_}_{network}"
        print(out)

        try:
            driver(
                network_pkl=network,
                class_idx=class_,
                outdir=out,
                seeds=list(range(0, nout)),
                max_batch_size=nout,
                subdirs=False,
                device="cuda",
            )
        finally:
            torch.distributed.destroy_process_group()

# %%

from mygenerate import *

# visualize images generated and saved in cell above

import PIL.Image
import matplotlib.pyplot as plt
import glob

import torch
import numpy as np

from decipher.vis import vis

for class_ in range(10):
    for network in networks:
        out = f"out_{class_}_{network}"
        ims = torch.cat(
            [
                torch.tensor(np.array(PIL.Image.open(f))).permute(2, 0, 1).unsqueeze(0)
                for f in glob.glob(f"{out}/*.png")
            ],
            dim=0,
        )
        plt.figure()
        vis(ims, title="")
        plt.suptitle(out)
        print(out)

# %%

from mygenerate import *


# test to make sure non-deterministic for seed repeats

network = "edm-cifar10-32x32-cond-ve.pkl"
class_ = 5
nout = 10

net = load_network(network, "cuda")

ims1 = driver(
    network_pkl=net,
    class_idx=class_,
    seeds=[1 * x for x in range(1, 1 + nout)],
    max_batch_size=2,
)

ims2 = driver(
    network_pkl=net,
    class_idx=class_,
    seeds=[1 * x for x in range(1, 1 + nout)],
    max_batch_size=2,
)

print((ims1 - ims2).abs().sum())


# %%

from mygenerate import *

# must provide None as class index for unconditional models!
network = "edm-cifar10-32x32-cond-ve.pkl"
class_ = 5
nout = 10

net = load_network(network, "cuda")

ims1 = driver(
    network_pkl=net,
    class_idx=None,
    seeds=[1 * x for x in range(1, 1 + nout)],
    max_batch_size=nout,
)
vis(ims1)


# %%
import time

network = "edm-cifar10-32x32-cond-ve.pkl"
class_ = 5
nout = 1000

net = load_network(network, "cpu")

t0 = time.time()
ims1 = driver(
    network_pkl=net,
    class_idx=class_,
    seeds=torch.randperm(nout),
    max_batch_size=500,
    device="cpu",
)
print(nout / (time.time() - t0), "im/s")

# cuda:
# bs=200, nout=1000 - 14 im/s -- uses about 50% of memory, 90% of cpu
# bs=500, nout=1000 - 12 im/s -- uses about 90% of memory, 100% of cpu
#
# cpu:
# bs=500, nout=1000 - 0.5 im/s -- uses about 90% of memory, 100% of cpu
