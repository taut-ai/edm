 %%

# generate 64 images for each class and save as pngs

import os
import torch
import pickle
import tqdm
import PIL

from generate import dist, dnnlib, StackedRandomGenerator, ablation_sampler, edm_sampler

DEBUG = False


def load_network(
    network_pkl: str,
    device: torch.device = "cuda",
) -> torch.nn.Module:
    """
    Load EDM network/model from pkl save file or URL pointing to the
    pkl save file.

    Weight files can be download from nvidia, eg:
      % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
      % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl
      % wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl
      ...
    """
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)
    return net


def driver(
    *,
    network_pkl: str | torch.nn.Module,
    outdir: str | None = None,
    subdirs: bool = False,
    seeds: list[int],
    class_idx: list[int] | None,
    max_batch_size: int,
    device: torch.device = "cuda",
    **sampler_kwargs,
) -> torch.Tensor:
    """
    Stand alone version of `generate.main(). Note that this is set up
    for multiprocessing out of the box, but works fine on single GPU/CPU.

    On storm, performance is around 12 images/s or 0.082 s/image. So
    about 22h to generate 1M images on 4090 (batchsize=64)

    Parameters
    ----------
    network_pkl: str | torch.nn.Module
        Instance of the EDM model to use (save file/URL or actual model).
        This generally should be one of the conditional ("cond") models,
        since the unconditional models don't take `class_idx`.

    outdir: str | None = None
        Dir to save png files. None means don't save, just return

    subdirs: bool = False
        if True, then create subdirs inside outdir for batches of 1000 images

    seeds: list[int]
        Random seeds -- this determines the number of images generated (ie,
        not `max_batch_size`. For identical seeds, the generation process is
        100% deterministic, so same seeds yield exactly the same images.

    class_idx: list[int] | None
        Desired class of generated image (usual 10 cifar classes). For the
        unconditional models, this must be `None`.

    max_batch_size: int
        Batch size for generation (breaks up len(seeds) into batches)

    device: torch.device = "cuda",
        Processing device

    Result
    ------
    torch.Tensor: synthetic images (len(seeds), 3, 32, 32)

    """

    try:
        dist.init()
        num_batches = (
            (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
        ) * dist.get_world_size()
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        # Load network.
        if isinstance(network_pkl, torch.nn.Module):
            net = network_pkl
        else:
            net = load_network(network_pkl, device)

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')

        # there's fencepost error in here and it generates one extra chunk!
        for batch_seeds in tqdm.tqdm(
            rank_batches, unit="batch", disable=(dist.get_rank() != 0)
        ):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
                device=device,
            )
            class_labels = None
            if net.label_dim:
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[batch_size], device=device)
                ]
            if class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, class_idx] = 1

            # Generate images.
            # JM: there are two samplers available in the code base (and described in paper):
            #    - edm_sampler
            #    - ablation_sampler
            sampler_kwargs = {
                key: value for key, value in sampler_kwargs.items() if value is not None
            }
            have_ablation_kwargs = any(
                x in sampler_kwargs
                for x in ["solver", "discretization", "schedule", "scaling"]
            )
            sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
            images = sampler_fn(
                net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs
            )

            if outdir is not None:
                # Optionally save images as pngs
                images_np = (
                    (images * 127.5 + 128)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for seed, image_np in zip(batch_seeds, images_np):
                    image_dir = (
                        os.path.join(outdir, f"{seed - seed % 1000:06d}")
                        if subdirs
                        else outdir
                    )
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"{seed:06d}.png")
                    if image_np.shape[2] == 1:
                        PIL.Image.fromarray(image_np[:, :, 0], "L").save(image_path)
                    else:
                        PIL.Image.fromarray(image_np, "RGB").save(image_path)
        # Done.
        torch.distributed.barrier()
        dist.print0("Done.")
    finally:
        torch.distributed.destroy_process_group()

    return images


# torchvision cifar dataset is stored as a bunch of pickled numpy byte arrays in
# (N, H, w, CHN) order.ArithmeticError
# we are going to store as byte tensors in (N, CHN, H, W) order, where N is
# requested `max_batch_size`


def make_dataset(
    *,
    network_pkl: str | torch.nn.Module,
    nimages: int,
    max_batch_size: int,
    max_images_per_file: int,
    device: torch.device = "cuda",
    outdir: str | None = None,
):
    """
    Stand alone version of `generate.main(). This is setup to run on
    single CPU/GPU -- no longer uses `torch.distributed`

    On storm, performance is around 12 images/s or 0.082 s/image. So
    about 22h to generate 1M images on 4090 (batchsize=64)

    Parameters
    ----------
    network_pkl: str | torch.nn.Module
        Instance of the EDM model to use (save file/URL or actual model).
        This generally should be one of the conditional ("cond") models,
        since the unconditional models don't take `class_idx`.

    nimages: int
        Total number of images to generate. Classes will be roughly evenly
        spilt - random draws, all classes equally likely.

    max_batch_size: int
        Batch size for generation - diffusion model called with with this side

    max_images_per_file: int
        Number of batches to dump in each data file

    device: torch.device = "cuda",
        Processing device

    """
    if outdir is None:
        outdir = "."
    else:
        os.makedirs(outdir, exist_ok=True)

    seeds = torch.randperm(int(nimages)) * int(100 * torch.rand(1))

    num_batches = (len(seeds) - 1) // max_batch_size + 1
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    # Load network.
    if isinstance(network_pkl, torch.nn.Module):
        net = network_pkl
        net_name = "?unknown?"
    else:
        net = load_network(network_pkl, device)
        net_name = network_pkl

    # Loop over batches.
    print(f'Generating {len(seeds)} images to "{outdir}"...')
    images_acc = None
    file_no = 0
    for batch_seeds in tqdm.tqdm(all_batches, unit="batch"):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )
        class_labels = torch.eye(net.label_dim, device=device)[
            rnd.randint(net.label_dim, size=[batch_size], device=device)
        ]
        if DEBUG:
            images = torch.rand(len(class_labels), 3, 32, 32)
        else:
            images = edm_sampler(net, latents, class_labels, randn_like=rnd.randn_like)

        # save images and labels for this batch
        images_ = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu()
        class_labels_ = class_labels.argmax(1).cpu().to(torch.int).cpu()

        # images, labels and associated accumulators should be on cpu
        if images_acc is None:
            images_acc = images_
            class_labels_acc = class_labels_
            seeds_acc = batch_seeds.cpu()
        else:
            images_acc = torch.cat((images_acc, images_))
            class_labels_acc = torch.cat((class_labels_acc, class_labels_))
            seeds_acc = torch.cat((seeds_acc, batch_seeds.cpu()))

        if images_acc.shape[0] >= max_images_per_file:
            # have enough images to save -- write up to max_images_per_file
            # and clear written data from accumulators
            torch.save(
                dict(
                    images=images_acc[:max_images_per_file],
                    labels=class_labels_acc[:max_images_per_file],
                    seeds=seeds_acc[:max_images_per_file],
                    diffmodel=net_name,
                ),
                os.path.join(outdir, f"data_batch_{file_no}"),
            )
            images_acc = images_acc[max_images_per_file:]
            class_labels_acc = class_labels_acc[max_images_per_file:]
            seeds_acc = seeds_acc[max_images_per_file:]
            file_no += 1

    if len(images) > 0:
        # there are residual images generated after the last write, so
        # save partial last accumulated chunk
        torch.save(
            dict(images=images_acc, labels=class_labels_acc, seeds=seeds_acc, diffmodel=net_name),
            os.path.join(outdir, f"data_batch_{file_no}"),
        )


if __name__ == "__main__":
    make_dataset(
        network_pkl="edm-cifar10-32x32-cond-ve.pkl",
        nimages=1_000_000,
        max_batch_size=700,
        max_images_per_file=5_000,
        outdir="cifar10",
        device="cuda",
    )
