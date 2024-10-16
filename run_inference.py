import json
import time
from pathlib import Path
import math
import time

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
print(f"\nUsing model: {MODEL_NAME}.\n")

# If you've moved the downloaded model-weights elsewhere, update this variable.
# For example, I moved them to: Path.home() / "src" / "Llama-LLM" / "downloaded-weights"
LLAMA_MODELS_DIR = Path.home() / ".llama" / "checkpoints"

# This is the prompt/input you'd like to pass to the model.
INPUT_STRING = "Write me a short story. It can be funny, emotional, intense, lovely, warm. It should be creative."

# mps stands for Metal Performance Shaders, i.e. Apple GPU's.
# cpu is central processing unit (i.e. the typical computational device).
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
print()

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
llama_model = llama_model.to(DEVICE)

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
print(f"Token representation is: {input_tokens}.")

# The model expects int64's (i.e. LongTensor). The extra [] are there to add a batch-dimension.
# Also, automatically add the beginning-of-sequence token.
beginning_of_sequence_token = 128_000
input_batch = torch.LongTensor([input_tokens + [beginning_of_sequence_token]]).to(DEVICE)

# ==========================================================================
# Run inference.

beam_width = 5
beam_idx_to_token_sequence = {}
beam_idx_to_sequence_log_probability = {}
beam_idx_to_time_elapsed = {}

# 128,001: <|end_of_text|> and 128,009: '<|eot_id|>'. I'm not sure what 'eot' 
# stands for in this context, though I commonly observe that token in places where
# there's a significant logical break in the text before and after.
end_of_sequence_tokens = [128_001, 128_009]
# Hm. May max out at 2,048. I need to test a bit more.
max_seq_len = 128

input_sequence_length = input_batch.shape[-1]


def append_token(tensor: torch.Tensor, token: int):
    """Helper function for appending a scalar token to a 2-D tensor.
    It's assumed the leading dimension of the tensor is 1, e.g. (1,N).
    """
    assert tensor.shape[0] == 1
    # The [None, None] adds two empty dimensions to ensure the dimensions 
    # of token and tensor match.
    token = torch.tensor(token, dtype=torch.int64, device=DEVICE)[None, None]
    return torch.cat([tensor, token], dim=1)

for beam_idx in range(beam_width):
    
    print(f"\nComputing beam: {beam_idx + 1}'s sequence.")
    
    beam_sequence_log_probability = 0.0
    beam_sequence = input_batch
    beam_start_time = time.time()
    is_beams_first_inference_pass = True
    next_most_likely_token = None
    
    while True:

        # Ignore the input-sequence/prompt.
        output_sequence_length = beam_sequence.shape[-1] - input_sequence_length
        
        if (
            next_most_likely_token in end_of_sequence_tokens or
            output_sequence_length >= max_seq_len
        ):
            
            if output_sequence_length >= max_seq_len:
                print(f"Reached maximum sequence length of: {max_seq_len}.")

            # Ignore the input/prompt tokens.
            beam_output_sequence = beam_sequence[:, input_sequence_length:]
            # Store the token-sequence & log-probability.
            beam_idx_to_token_sequence[beam_idx] = beam_output_sequence.squeeze().tolist()
            beam_idx_to_sequence_log_probability[beam_idx] = beam_sequence_log_probability
            beam_idx_to_time_elapsed[beam_idx] = time.time() - beam_start_time
            
            break

        else:
                
            output_batch = llama_model(beam_sequence, start_pos=0)
            
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

            if is_beams_first_inference_pass:
                
                # On the first pass, we want to initialize the beam-sequences with the most likely {beam-width}
                # possible tokens. That is, beam 1 begins with the most likely token, beam 2 begins with
                # the 2nd most likely token, etc.
                top_k_results = torch.topk(next_token_probabilities, k=beam_width, dim=0)
                next_most_likely_token = top_k_results.indices[beam_idx].item()
                most_likely_token_probability = top_k_results.values[beam_idx].item()
                
                is_beams_first_inference_pass = False

            else:
                # Select the index (i.e. the token) that has been assigned the highest probability value.
                next_most_likely_token = torch.argmax(next_token_probabilities, dim=0).item()
                most_likely_token_probability = torch.max(next_token_probabilities, dim=0).values.item()
            
            # Keep track of the most-likely tokens and the overall sequence log-probability.
            beam_sequence = append_token(beam_sequence, next_most_likely_token)
            beam_sequence_log_probability += math.log(most_likely_token_probability)
            
            next_most_likely_token_str = tokenizer.decode(t=[next_most_likely_token])
            if "\n" in next_most_likely_token_str:
                # escape the back-slash to prevent odd formatting in the later print statement.
                next_most_likely_token_str = next_most_likely_token_str.replace("\n", "\\n")
            
            # Pad the numeric representation so the ensuing print-statement's contents are aligned across lines.
            # For example, 126 -> "126  "; 1 -> "1    "; 52947 -> "52947". The longest token (128000) 
            # formatted with a comma has 7 characters: 6 digits and the comma.
            next_most_likely_token_padded = f"{next_most_likely_token:,}{' ' * (7 - len(f"{next_most_likely_token:,}"))}"
            print(f"next token {next_most_likely_token_padded}-> '{next_most_likely_token_str}' with p: {most_likely_token_probability:.3f}.")

# Decode each beam's output sequence.
for beam_idx in range(beam_width):
    
    beam_sequence = beam_idx_to_token_sequence[beam_idx]
    beam_log_probability = beam_idx_to_sequence_log_probability[beam_idx]
    decoded_beam_tokens = tokenizer.decode(t=beam_sequence)
    beam_time_elapsed = beam_idx_to_time_elapsed[beam_idx]

    print(f"\n\nBeam {beam_idx+1} ran for: {beam_time_elapsed:.0f}s. The average throughput was {len(beam_sequence) / beam_time_elapsed:.2f} tokens/second.")
    print(f"Beam: {beam_idx+1} predicted tokens with joint log-p: {beam_log_probability:.2f} that correspond to this string: \n{decoded_beam_tokens}")