
# download pre-trained model weights

> % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
> % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl
> % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl
> % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vf.pkl
> % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl


# generate images locally

This will generate 1M images (about 9h on storm) into `./cifar10edm`.

> % python mygenerate.py

Settings are (currently):

> network_pkl="edm-cifar10-32x32-cond-ve.pkl"   # downloaded conditional exponential diffusion model
> nimages=1_000_000                             # generate 1m images
> max_batch_size=700                            # batch size for tensors
> max_images_per_file=5_000                     # number of images per chunk file
> outdir="cifar10edm"                           # where to store chunk files (locally)
> device="cuda"                                 # use cuda
