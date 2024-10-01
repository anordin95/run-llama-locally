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


# Select the model-weights you'd like to use; some options are: "Llama3.2-1B",
# "Llama3.2-1B-Instruct", "Llama3.2-3B", etc.
MODEL_NAME = "Llama3.2-1B-Instruct" 
print(f"Using model: {MODEL_NAME}.")

# If you've moved the downloaded model-weights elsewhere, update this variable.
# For example, I moved them to: Path.home() / "src" / "Llama-LLM" / "downloaded-weights"
LLAMA_MODELS_DIR = Path.home() / ".llama" / "checkpoints"

# This is the prompt/input you'd like to pass to the model.
INPUT_STRING = "Hi. Tell me about yoruself. Who are you? What do you do?"

# mps stands for Metal Performance Shaders, i.e. Apple GPU's.
# Something went awry when I tried using mps; the model output a tensor 
# full of nan's.
DEVICE = torch.device("cpu")


# ==========================================================================
# Both the torch.distributed and fairscale packages need to be initialized
# prior to being able to create an instance of the Llama Model class.

# Initialize the torch.distributed sub-package. 
# world_size: The number of processes to use.
# rank: the current processes' index.
# store: a distributed key-value store that all workers should have access to.
torch.distributed.init_process_group(world_size=1, rank=0, store=torch.distributed.HashStore())

# Initialize the fairscale package.
initialize_model_parallel(model_parallel_size_=1)

# ==========================================================================
# Load the model & model-weights.

# Initialize a model. That is, load the architecture, but not
# the weights.
model_hyperparams_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/params.json"
with open(model_hyperparams_path, "r") as fh:
    model_hyperparams_dict = json.load(fh)
model_hyperparams = ModelArgs(**model_hyperparams_dict)
llama_model = Llama3Model(model_hyperparams, DEVICE)

# Load saved weights into the model architecture.
model_weights_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/consolidated.00.pth"
tensor_name_to_tensor_weights = torch.load(model_weights_path, weights_only=True, map_location=DEVICE)
llama_model.load_state_dict(tensor_name_to_tensor_weights)

# ==========================================================================
# Setup and use the tokenizer.

# Prepare the tokenizer. This is the component responsible for transforming plain-English into
# a sequence of class-labels, i.e. strings to long-ints.
token_dictionary_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/tokenizer.model"
tokenizer = Tokenizer(model_path=str(token_dictionary_path))

# Convert the input-string to a series of tokens within a batch.
# bos and eos are booleans indicating whether a beginning-of-sequence 
# and end-of-sequence token should be prepended and appended, respectively,
# to the returned token-sequence.
print(f"Converting input-string: '{INPUT_STRING}' to tokens.")
input_tokens = tokenizer.encode(s=INPUT_STRING, bos=True, eos=True)
# The model expects int64's (i.e. LongTensor). The extra [] are there to add a batch-dimension.
input_batch = torch.LongTensor([input_tokens + [128000]]).to(DEVICE)

# ==========================================================================
# Run inference.

class PossibleTokenSequence:
    def __init__(self, starting_sequence: torch.LongTensor):
        self.token_sequence = starting_sequence
        self.log_p = 0.0
        self.initial_sequence_length = starting_sequence.shape[-1]

    def append_token(self, token: torch.LongTensor, p: float):
        # The [None, None] adds two empty dimensions to ensure the dimensions of the inputs
        # -- token_sequence and token -- are the same.
        self.token_sequence = torch.cat([self.token_sequence, token[None, None]], dim=1)
        self.log_p += math.log(p)
    
    def get_output_sequence(self):
        return self.token_sequence[:, self.initial_sequence_length:]

beam_width = 7
possible_token_sequences = [PossibleTokenSequence(input_batch) for _ in range(beam_width)]

next_most_likely_token = None
end_of_sequence_token = 128001
max_seq_len = 256
is_first_inference_pass = True

while next_most_likely_token != end_of_sequence_token:
        
    current_seq_len = possible_token_sequences[0].get_output_sequence().shape[-1]
    if current_seq_len >= max_seq_len:
        print(f"Reached maximum sequence length of: {max_seq_len}.")
        break
    
    for beam_idx in range(beam_width):

        possible_token_sequence = possible_token_sequences[beam_idx]

        # What does start_pos mean?
        output_batch = llama_model(possible_token_sequence.token_sequence, start_pos=0)

        # Take the first batch-output. Recall there was only one batch-input.
        output_activations = output_batch[0]

        # The output is now shape (num-input-tokens, vocabulary-size). Each row
        # represents the next-token prediction scores of the given index. For example
        # row 2 represents the 3rd token prediction scores given tokens 1 and 2.
        # We only care about the final next-token prediction, i.e. the next token
        # given our input-token sequence.
        next_token_activations = output_activations[-1]

        # Softmax the outputs, then select the most likely token at each step.
        # The next-word activation vector has the same dimensionality as the 
        # token-vocabulary. Take the softmax over the vector to obtain a probability
        # distribution over possible tokens.
        next_token_probabilities = next_token_activations.softmax(dim=0)
        
        if is_first_inference_pass:
            # On the first pass, we want to initialize the beam-sequences with the most likely {beam-width}
            # possible tokens.
            next_most_likely_token = torch.topk(next_token_probabilities, k=beam_width, dim=0).indices[beam_idx]
        else:
            # Select the index (i.e. the token) that has been assigned the highest probability value.
            next_most_likely_token = torch.argmax(next_token_probabilities, dim=0)

        next_most_likely_token_str = tokenizer.decode(t=[next_most_likely_token])

        # Find the probability associated with the token that was chosen.
        most_likely_token_probability = torch.max(next_token_probabilities, dim=0).values.item()
        
        # Keep track of the most-likely tokens. And, add the predicted token to the input.
        possible_token_sequence.append_token(token=next_most_likely_token, p=most_likely_token_probability)

        print(f"For beam: {beam_idx + 1}, the model thinks token {next_most_likely_token_str} is the most likely token to come next with p: {most_likely_token_probability:.3f}.")

    is_first_inference_pass = False

# Decode the output tokens.
for beam_idx in range(beam_width):
    possible_token_sequence = possible_token_sequences[beam_idx]
    decoded_tokens = tokenizer.decode(t=possible_token_sequence.get_output_sequence().squeeze().tolist())
    print(f"\n\nBeam: {beam_idx+1} predicted tokens correspond to this string: \n'{decoded_tokens}' with log-p: {possible_token_sequence.log_p:.4f}.")