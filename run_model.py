import json
import time
from pathlib import Path

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

# If, like me, you've moved the downloaded model-weights elsewhere, update this variable.
# LLAMA_MODELS_DIR = Path.home() / ".llama"
LLAMA_MODELS_DIR = Path.home() / "src" / "Llama-LLM" / "downloaded-weights"

# This is the prompt/input you'd like to pass to the model.
INPUT_STRING = "Hi. Who are you?"

# mps stands for Metal Performance Shaders, i.e. Apple GPU's.
# Something went awry when I tried using mps; the model output a tensor 
# full of nan's.
DEVICE = torch.device("cpu")


# Both the torch.distributed and fairscale packages need to be initialized
# prior to being able to create an instance of the Llama Model class.

"""
Initialize the torch.distributed sub-package. 
world_size: The number of processes to use.
rank: the current processes' index.
store: a distributed key-value store that all workers should have access to.
"""
torch.distributed.init_process_group(world_size=1, rank=0, store=torch.distributed.HashStore())

"""
Initialize the fairscale package.
"""
initialize_model_parallel(model_parallel_size_=1)

"""
Initialize a model. That is, load the architecture, but not
the weights.
"""
model_hyperparams_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/params.json"
with open(model_hyperparams_path, "r") as fh:
    model_hyperparams_dict = json.load(fh)
model_hyperparams = ModelArgs(**model_hyperparams_dict)
llama_model = Llama3Model(model_hyperparams, DEVICE)

"""
Load saved weights into the model architecture.
"""
model_weights_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/consolidated.00.pth"
tensor_name_to_tensor_weights = torch.load(model_weights_path, weights_only=True, map_location=DEVICE)
llama_model.load_state_dict(tensor_name_to_tensor_weights)

"""
Prepare the tokenizer. This is the component responsible for transforming plain-English into
a sequence of class-labels, i.e. strings to long-ints.
"""
token_dictionary_path = LLAMA_MODELS_DIR / f"{MODEL_NAME}/tokenizer.model"
tokenizer = Tokenizer(model_path=str(token_dictionary_path))

"""
Select the output-string and convert it to a series of tokens within a batch.
"""
# bos and eos are booleans indicating whether a beginning-of-sequence 
# and end-of-sequence token should be prepended and appended, respectively,
# to the returned token-sequence.
print(f"Converting input-string: '{INPUT_STRING}' to tokens.")
input_tokens = tokenizer.encode(s=INPUT_STRING, bos=True, eos=True)
# The model expects int64's (i.e. LongTensor). The extra [] are there to add a batch-dimension.
input_tokens_batch = torch.LongTensor([input_tokens]).to(DEVICE)

"""
Run inference, softmax the outputs, then select the most likely token at each step.
"""
print(f"Inputting the sequence of tokens: {input_tokens} to the model.")
start = time.time()
output_batch = llama_model(input_tokens_batch, start_pos=0)
end = time.time()
print(f"Inference finished. Time elapsed: {end-start:.2f}s.")
# Take the first batch-output. Recall there was only one batch-input.
output_activations = output_batch[0]
# Softmax across the possible output indices (i.e. tokens) to obtain a probability
# distribution over possible output tokens.
softmaxed_output = torch.nn.functional.softmax(output_activations, dim=1)
# Select the index (i.e. token number) that has been assigned the highest probability value.
output_tokens = torch.argmax(softmaxed_output, dim=1).tolist()
# Find the probability associated with each token that was chosen.
output_token_probabilities = torch.max(softmaxed_output, dim=1).values.tolist()

print(f"The model predicted this sequence of tokens: {output_tokens}.")
print(f"The probability assigned to each predicted token was: {[round(p, 3) for p in output_token_probabilities]}.")

"""
Decode the output tokens.
"""
decoded_tokens = tokenizer.decode(t=output_tokens)
print(f"Those predicted tokens correspond to this string: \n'{decoded_tokens}'.")