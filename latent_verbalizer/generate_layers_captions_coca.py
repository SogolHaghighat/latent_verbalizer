import argparse
import json
import os
from typing import Any, Dict

import h5py
import torch
from utility import (
    InterpretCfg,
    calculate_entropy,
    give_caption,
    interpret,
    load_model,
    logger,
)


def process_batch(
    config: InterpretCfg, model_components: Dict[str, Any], device: torch.device
):
    """
    Process a single batch: generate captions and probabilities.

    Args:
        config (InterpretCfg): Configuration for interpretation.
        model_components (Dict[str, Any]): Loaded model components.
        device (torch.device): Device to perform computations on.
    """
    input_path = config.input_path_template
    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} does not exist.")
        return

    logger.info(f"Loading data from {input_path}...")
    with h5py.File(input_path, "r") as feat:
        caps = dict()
        entropy_dict = dict()
        names = feat.get("names", [])[:]
        for layer_no in range(24):  # FIXME: this model has 24 layers!
            layer_key = f"transformer.resblocks.{layer_no}"
            if layer_key not in feat:
                logger.warning(
                    f"{layer_key} not found in {input_path}. Skipping layer {layer_no}."
                )
                continue

            logger.debug(f"Processing layer {layer_no} for data {input_path}...")
            curr_layer_features = torch.tensor(
                feat[layer_key][:], dtype=torch.float32
            ).to(device)
            features = curr_layer_features.permute(
                1, 0, 2
            )  # Adjust dimensions if necessary

            # Run interpretation
            batched_token_ids, _, probs = interpret(
                features=features,
                config=config,
                vision_encoder=model_components["att_pooler_ln"],
                text_encoder=model_components["text_encoder"],
                multimodal_decoder=model_components["multimodal_decoder"],
                vocab=model_components["coca_vocab"],
                text_decoder=model_components["text_decoder"],
            )

            batch_size = batched_token_ids.size(0)
            curr_layer_caption = []
            curr_layer_entropy = []

            for sample in range(batch_size):
                # generate captions
                caption = give_caption(
                    batched_token_ids[sample], model_components["text_decoder"]
                )
                curr_layer_caption.append(caption)

                if config.keep_probs and probs is not None:
                    tokens = []
                    for token in batched_token_ids[sample]:
                        if token.item() == 49407:
                            break
                        tokens.append(token.item())
                    entropies = 0
                    for i, t in enumerate(tokens):
                        e = calculate_entropy(probs[i][sample].cpu().numpy())
                        entropies += e
                    curr_layer_entropy.append(entropies / (len(tokens)))

            caps[f"layer_{layer_no}"] = curr_layer_caption
            if config.keep_probs:
                entropy_dict[f"layer_{layer_no}"] = curr_layer_entropy

            logger.info(f"Generated captions for all samples at layer {layer_no}.")

    # Save captions
    output_caps_path = config.output_caps_path_template
    os.makedirs(os.path.dirname(output_caps_path), exist_ok=True)
    with open(output_caps_path, "w") as f:
        json.dump(caps, f)
    logger.info(f"Saved captions to {output_caps_path}")

    # Save entropies if required
    if config.keep_probs and config.output_entropy_path_template:
        output_entropy_path = config.output_entropy_path_template
        os.makedirs(os.path.dirname(output_entropy_path), exist_ok=True)
        with open(output_entropy_path, "w") as f:
            json.dump(entropy_dict, f)
        logger.info(f"Saved entropies to {output_entropy_path}")


def main(config: InterpretCfg):
    """
    Main function to process all batches.

    Args:
        config (InterpretCfg): Configuration for interpretation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model_components = load_model(config, device)
    logger.info(f"Configuration: {config}")

    process_batch(config, model_components, device)


if __name__ == "__main__":
    """
    Entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description="COCA Model Batch Interpretation Script"
    )

    # Add arguments corresponding to fields in InterpretCfg
    parser.add_argument(
        "--config",
        type=str,
        default="interpret.yaml",
        help="Path to the config yaml file for interpretation.",
    )
    args = parser.parse_args()

    try:
        config = InterpretCfg.from_yaml(args.config)
        main(config)
    except Exception as e:
        logger.exception("An error occurred during processing.")
        exit(1)
