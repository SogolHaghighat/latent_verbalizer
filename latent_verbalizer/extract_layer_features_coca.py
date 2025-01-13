import argparse
import logging
import os
import time
from typing import Callable

import h5py
import open_clip
import torch
import webdataset as wds

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# source: https://github.com/webdataset/webdataset/blob/main/examples/train-resnet50-wds.ipynb
def make_sample(sample):
    """Take a decoded sample dictionary, augment it, and return an (image, label, key_name)."""
    image = sample["jpg"]
    text = sample["json"]
    name = sample["__key__"]
    return transform(image), text, name


# Source: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/ & https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
def get_activation(layer_id: str) -> Callable:
    def fn(_, __, output):
        features[layer_id] = output.detach().to("cpu")

    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extract features from images at different layers of CoCa model.",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="../data/test/{000000..000004}.tar",
        help="The url path to the dataset that is used for feature extraction in webdataset format.",
    )
    parser.add_argument(
        "-s",
        "--sample-per-shards",
        type=int,
        default=1000,
        help="The number of samples in a shard of webdataset format.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=500,
        help="The batch size for performing feature extraction from the frames. limited to the size of the gpu.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="The epoch for iterating through the infinite webdataset data. calculated based on the size of the dataset and batch size.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="coca_ViT-L-14",
        help="The name of the coca model used for interpretation.",
    )
    parser.add_argument(
        "-pre",
        "--pretrained",
        type=str,
        default="mscoco_finetuned_laion2B-s13B-b90k",
        help="The name of the pretraining data for coca model.",
    )
    parser.add_argument(
        "-nl",
        "--num-layers",
        type=int,
        default=24,
        help="The number of layers for the vision model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../data/interpret",
        help="The output path for extracted features.",
    )

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration: {args}")

    # Load model
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=args.model, pretrained=args.pretrained, device=device
    )
    visual = model.visual
    logger.info("Model loaded successfully.")

    # Attach hooks to relevant layers
    features = {}
    layers = []

    # This decoposition is only for the open-coca implementation from : https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/coca_model.py
    for i in range(args.num_layers):
        layers.append(
            f"transformer.resblocks.{i}"
        )  # output of each transformer layer into the residual stream
        # layers.append(f"transformer.resblocks.{i}.ln_1") # input to the self-attention
        # layers.append(f'transformer.resblocks.{i}.ls_1') # output of the self-attention

    for layer_id in layers:
        layer = dict([*visual.named_modules()])[layer_id]
        layer.register_forward_hook(get_activation(layer_id))

    # Create webdataset dataloader
    urls = args.dataset
    dataset = wds.WebDataset(urls, resampled=False, shardshuffle=False)
    dataset = dataset.shuffle(0).decode("pil").map(make_sample)  # no shuffling here
    dataset = dataset.batched(args.batch_size)
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=0)
    dataloader = dataloader.with_epoch(
        args.epochs
    )  # dataset is infinite, it needs an epoch
    logger.info("Dataset loaded successfully.")

    t1 = time.time()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels, names]
            test_data = os.path.join(
                args.output,
                f"layers_features_{args.model}_{args.pretrained}_batch{i}.h5",
            )
            features = {}
            inputs, _, names = data[0].to(device), data[1], data[2]
            _ = visual(inputs)

            # save data directly to h5 file
            with h5py.File(test_data, "w") as f:
                for layer in layers:
                    f.create_dataset(layer, data=features[layer])
                f.create_dataset("names", data=names)

            logger.info(
                f"completed extracting layers features for batch {i} in {time.time() - t1} sec"
            )
            t1 = time.time()
