import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import open_clip
import torch
import torch.nn as nn
import yaml
from pydantic import BaseModel, Field
from torch.nn import functional as F

# Import transformers-related components if needed
try:
    from transformers import (
        EosTokenCriteria,
        LogitsProcessorList,
        MaxLengthCriteria,
        MinLengthLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        StoppingCriteriaList,
        StopStringCriteria,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    GENERATION_TYPES = {"top_k": TopKLogitsWarper, "top_p": TopPLogitsWarper}
    _has_transformers = True
except ImportError:
    GENERATION_TYPES = {"top_k": None, "top_p": None}
    _has_transformers = False

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class InterpretCfg(BaseModel):
    seq_len: int = Field(default=30, description="The generated sequence length")
    max_seq_len: int = Field(
        default=77, description="maximum length of the generated sequence"
    )
    temperature: float = Field(default=1.0, description="")
    generation_type: str = Field(
        default="top_k",
        description="generation type at the moment is only top_k or top_p",
    )
    top_k: int = Field(
        default=1,
        description="the number of top choices to sample when generating the next token",
    )
    top_p: float = Field(
        default=0.9,
        description="samples from top p% when generating the next token",  # FIXME: definition
    )
    repetition_penalty: float = Field(
        default=1.0, description="repetition penalty for generating the next token"
    )
    model_name: str = Field(
        default="coca_ViT-L-14",
        description="model to be interpreted. at the moment only coca",
    )
    pretrained: str = Field(
        default="mscoco_finetuned_laion2B-s13B-b90k",
        description="the pretraining dataset for coca model. refer to open_clip repo",
    )
    keep_probs: bool = Field(
        default=True,
        description="when true the probabilities when generating each token is also returned by the interpret function",
    )
    pad_token_id: Optional[int] = Field(default=0, description="padding token id")
    eos_token_id: Optional[int] = Field(
        default=49407, description="end of sentence token id"
    )
    sot_token_id: Optional[int] = Field(
        default=49406, description="start of text token id"
    )
    min_seq_len: int = Field(
        default=5, description="minimum length for the generated sequence"
    )
    fixed_output_length: bool = Field(
        default=False, description="if True, output.shape == (batch_size, seq_len)"
    )

    # File paths
    input_path_template: str = Field(
        default="/data/interpret/layers_features_coca_ViT-L-14_mscoco_finetuned_laion2B-s13B-b90k_batch0.h5",
        description="Input HDF5 file path template containing the extracted visual features from the layers of CoCa vision encoder.",
    )
    output_caps_path_template: str = Field(
        default="/data/interpret/caps_batch0.json",
        description="Output captions JSON file path template.",
    )
    output_entropy_path_template: Optional[str] = Field(
        default="/data/interpret/entropy_batch0.json",
        description="Output probabilities JSON file path template.",
    )

    @classmethod
    def from_yaml(cls, file_path: str) -> "InterpretCfg":
        with open(file_path, "r") as infile:
            return cls(**yaml.safe_load(infile))


def seperate_cls(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pooled, tokens = x[:, 0], x[:, 1:]
    return pooled, tokens


def calculate_entropy(token_probabilities):
    """
    Calculate the entropy for a sequence of tokens.

    Parameters:
    token_probabilities (list or numpy array): Probabilities of each token in the sequence.

    Returns:
    float: Entropy of the sequence.
    """
    # Ensure the probabilities are valid (sum to 1 and non-negative)
    token_probabilities = np.array(token_probabilities)
    token_probabilities = np.clip(token_probabilities, 1e-15, 1 - 1e-15)

    # Calculate the entropy
    entropy = -np.sum(token_probabilities * np.log2(token_probabilities))  # type: ignore

    return entropy


# this function is adaptation of generate method from coca model implementation of open_clip
# https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/coca_model.py#L194
# this performms batch interpretation
def interpret(
    image: Optional[torch.Tensor] = None,
    features: Optional[torch.Tensor] = None,
    text: Optional[torch.Tensor] = None,
    config: InterpretCfg = None,  # type: ignore
    stopping_criteria: Optional[Any] = None,
    vision_encoder: Optional[nn.Module] = None,
    text_encoder: Optional[nn.Module] = None,
    multimodal_decoder: Optional[nn.Module] = None,
    vocab: Optional[Any] = None,
    text_decoder: Optional[Callable[[torch.Tensor], str]] = None,
):
    """
    Generate token ids to interpret the visual features or images using the COCA model.

    Args:
        image (torch.Tensor, optional): Image tensor input.
        features (torch.Tensor, optional): Features tensor input.
        text (torch.Tensor, optional): Initial text tensor input.
        config (InterpretCfg): Configuration for interpretation.
        vision_encoder (nn.Module, optional): Vision encoder component.
        text_encoder (nn.Module, optional): Text encoder component.
        multimodal_decoder (nn.Module, optional): Multimodal decoder component.
        vocab (Any, optional): Vocabulary or tokenizer.
        text_decoder (Callable[[torch.Tensor], str], optional): Function to decode tokens to text.

    Returns:
        Tuple[torch.Tensor, int, list]: Generated token IDs, sequence length, and probabilities (if kept).
    """
    assert config is not None, "Configuration must be provided."
    assert vision_encoder is not None, "Vision encoder must be provided."
    assert text_encoder is not None, "Text encoder must be provided."
    assert multimodal_decoder is not None, "Multimodal decoder must be provided."
    assert vocab is not None, "Vocabulary must be provided."
    assert text_decoder is not None, "Text decoder function must be provided."

    assert (
        _has_transformers
    ), "Please install transformers for generate functionality. `pip install transformers`."
    assert (
        config.seq_len > config.min_seq_len
    ), "seq_len must be larger than min_seq_len"

    # Initialize tokens and parameters
    sot_token_id = config.sot_token_id or 49406
    eos_token_id = config.eos_token_id or 49407
    pad_token_id = config.pad_token_id or 0

    logit_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(config.min_seq_len, eos_token_id),
            RepetitionPenaltyLogitsProcessor(config.repetition_penalty),
        ]
    )

    stopping_criteria = (
        StoppingCriteriaList([MaxLengthCriteria(max_length=config.seq_len)])
        if stopping_criteria is None
        else StoppingCriteriaList(stopping_criteria)
    )

    # Define logit warper based on generation_type
    if config.generation_type == "top_p":
        logit_warper = GENERATION_TYPES[config.generation_type](config.top_p)  # type: ignore
    elif config.generation_type == "top_k":
        logit_warper = GENERATION_TYPES[config.generation_type](config.top_k, min_tokens_to_keep=1)  # type: ignore
    else:
        raise ValueError(
            f"generation_type must be one of {', '.join(GENERATION_TYPES.keys())}"
        )

    # Encode image or features
    if image is not None:
        _, img_embs = vision_encoder(image)
        device = image.device
        batch = image.size(0)
    elif features is not None:
        _, img_embs = seperate_cls(vision_encoder(features))
        device = features.device
        batch = features.size(0)
    else:
        raise ValueError(
            "Either image or features must be provided for interpretation."
        )

    # Initialize text input
    if text is None:
        text = torch.ones((batch, 1), device=device, dtype=torch.long) * sot_token_id

    num_dims = len(text.shape)

    if num_dims == 1:
        text = text[None, :]

    out = text
    data = []

    with torch.no_grad():
        while True:
            x = out[:, -config.max_seq_len :]
            cur_len = x.size(1)
            _, token_embs = text_encoder(x)
            logits = multimodal_decoder(image_embs=img_embs, text_embs=token_embs)[
                :, -1
            ]

            mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
            sample = (
                torch.ones((out.shape[0], 1), device=device, dtype=torch.long)
                * pad_token_id
            )

            if mask.all():
                if not config.fixed_output_length:
                    break
            else:
                all_probs = F.softmax(logits / config.temperature, dim=-1)
                logits = logits[~mask, :]
                filtered_logits = logit_processor(x[~mask, :], logits)  # type: ignore
                filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                probs = F.softmax(filtered_logits / config.temperature, dim=-1)

                if cur_len + 1 == config.seq_len:
                    sample[~mask, :] = (
                        torch.ones((sum(~mask), 1), device=device, dtype=torch.long)  # type: ignore
                        * eos_token_id
                    )
                else:
                    sample[~mask, :] = torch.multinomial(probs, 1)

                if config.keep_probs:
                    data.append(all_probs.cpu())

            out = torch.cat((out, sample), dim=-1)
            cur_len += 1

            is_done = False
            if (
                EosTokenCriteria in stopping_criteria
                or StopStringCriteria in stopping_criteria
            ):
                is_done = stopping_criteria(out, None).all()  # type: ignore
            else:
                is_done = stopping_criteria(out, None).any()  # type: ignore
            if is_done:
                break
        if num_dims == 1:
            out = out.squeeze(0)

    return out, cur_len, data if config.keep_probs else None


def give_caption(
    token_ids: torch.Tensor, text_decoder: Callable[[torch.Tensor], str]
) -> str:
    """
    Convert token IDs to a human-readable caption.

    Args:
        token_ids (torch.Tensor): Tensor of token IDs.
        text_decoder (Callable[[torch.Tensor], str]): Function to decode tokens to text.

    Returns:
        str: Generated caption.
    """
    decoded_text = text_decoder(token_ids)
    caption = (
        decoded_text.split("<end_of_text>")[0].replace("<start_of_text>", "").strip()
    )
    return caption


def load_model(config: InterpretCfg, device: torch.device) -> Dict[str, Any]:
    """
    Load the COCA model and its components.

    Args:
        config (InterpretCfg): Configuration for the model.
        device (torch.device): Device to load the model on.

    Returns:
        Dict[str, Any]: Loaded model components.
    """
    logger.info(
        f"Loading model {config.model_name} with pretrained weights {config.pretrained} on {device}"
    )
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=config.model_name, pretrained=config.pretrained, device=device
    )

    model.to(device)
    model.eval()

    multimodal_decoder = model.text_decoder
    visual = model.visual
    text_encoder = model.text
    att_pooler = model.visual.attn_pool
    att_pooler_ln = nn.Sequential(att_pooler, model.visual.ln_post)  # type: ignore
    text_decoder_func = open_clip.decode  # Ensure this function exists in open_clip
    tokenizer = open_clip.get_tokenizer(config.model_name)
    coca_vocab = tokenizer.decoder  # type: ignore

    logger.info("Model loaded successfully.")

    return {
        "model": model,
        "multimodal_decoder": multimodal_decoder,
        "visual": visual,
        "text_encoder": text_encoder,
        "att_pooler_ln": att_pooler_ln,
        "text_decoder": text_decoder_func,
        "coca_vocab": coca_vocab,
        "transform": transform,
    }


# class adapted from: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
class FeatureExtractor:
    def __init__(self, model: nn.Module, layer_name: str):
        """
        Initialize the FeatureExtractor.

        Args:
        - model (nn.Module): The PyTorch model from which to extract features.
        - layer_name (str): The name of the layer where the hook should be attached.
        """
        self.model = model
        self.layer_name = layer_name
        self.features = None  # To store the features
        self._hook = None  # Store the hook handle to remove it later if needed

        # Find the layer by name and attach the hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                self._hook = module.register_forward_hook(self._forward_hook)
                break
        else:
            raise ValueError(f"Layer named {layer_name} not found in the model.")

    def _forward_hook(self, module, input, output):
        """
        Hook function to be registered with the layer. This function will
        be called every time forward is executed on the target layer.

        Args:
        - module: The module to which this hook is registered.
        - input: The input to the module.
        - output: The output from the module.
        """
        # Here we're just storing the output, but you could process it here if needed
        self.features = output.detach()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the specified layer for given input.

        Args:
        - x (torch.Tensor): Input tensor to pass through the network.

        Returns:
        - torch.Tensor: Features extracted from the specified layer.
        """
        # Clear previous features
        self.features = None

        # Run the forward pass, this will trigger the hook
        with torch.no_grad():  # Assuming we don't need gradients for feature extraction
            self.model(x)

        if self.features is None:
            raise RuntimeError(
                "Features were not captured. Check if the layer name is correct or if the model was run."
            )

        return self.features

    def remove_hook(self):
        """
        Remove the hook from the model to prevent it from affecting future forward passes.
        """
        if self._hook:
            self._hook.remove()
            self._hook = None
