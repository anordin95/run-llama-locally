import json
import time
from pathlib import Path
import math

from llama_models.llama3.reference_impl.model import Transformer as Llama3Model
from llama_models.llama3.api.args import ModelArgs
from llama_models.llama3.api.tokenizer import Tokenizer
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized
)
import torch.distributed

MODEL_NAME = "Llama3.2-1B-Instruct" 
LLAMA_MODELS_DIR = Path.home() / ".llama" / "checkpoints"
INPUT_STRING = "Write me a short story."
DEVICE = torch.device("cpu")

# Initialize torch.distributed & fairscale.
torch.distributed.init_process_group(world_size=1, rank=0, store=torch.distributed.HashStore())
initialize_model_parallel(model_parallel_size_=1)

# Load the model architecture.
model_hyperparams_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/params.json"
with open(model_hyperparams_path, "r") as fh:
    model_hyperparams_dict = json.load(fh)
model_hyperparams = ModelArgs(**model_hyperparams_dict)
llama_model = Llama3Model(model_hyperparams, DEVICE)

# Load the model weights
model_weights_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/consolidated.00.pth"
tensor_name_to_tensor_weights = torch.load(model_weights_path, weights_only=True, map_location=DEVICE)
llama_model.load_state_dict(tensor_name_to_tensor_weights)
llama_model = llama_model.to(DEVICE)

# Setup the tokenizer & input-sequence.
beginning_of_sequence_token = 128_000
token_dictionary_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/tokenizer.model"
tokenizer = Tokenizer(model_path=str(token_dictionary_path))
input_tokens = tokenizer.encode(s=INPUT_STRING, bos=True, eos=True)
input_batch = torch.LongTensor([input_tokens + [beginning_of_sequence_token]]).to(DEVICE)

output_batch = llama_model(input_batch, start_pos=0)
# Take the first batch-entry & final token-index.
final_token_activations = output_batch[0][-1]

