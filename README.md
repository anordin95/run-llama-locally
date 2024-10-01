### Running Llama locally with minimal dependencies

#### Motivation

This repository provides a bare-bones way of using the Llama language models. 

I was a bit surprised Meta didn't publish an example way to simply invoke one of these LLM's with only `torch` (or some minimal set of dependencies), though I am obviously grateful and so pleased with their contribution of the public weights! There are other popular ways to invoke these models, such as [Ollama](https://ollama.com/) and Hugging-Face's general API package: [transformers](https://pypi.org/project/transformers/), but those hide the interesting details behind an API. I want to peel back the layers, poke, prod, understand and gain insight into these models and help you do the same.

#### Setup steps

1. Download the relevant model weight(s) via https://www.llama.com/llama-downloads/

2. `pip install -r requirements.txt`

3. `cd llama-models; pip install -e .; cd ..`

4. `python run_inference.py`

#### Arguments

The three global variables in run_inference.py: `MODEL_NAME`, `LLAMA_MODELS_DIR` and `INPUT_STRING` take the values you'd expect (there are adjacent comments with examples and more details too). They should be modified as you see fit.

#### Technical Overview 

The minimal set of dependencies I found includes `torch` (perhaps, obviously), a lesser known library also published by Meta: `fairscale`, which implements a variety of highly scalable/parallelizable analogues of `torch` operators and `blobfile`, which implements a general file I/O mechanism that Meta's Tokenizer implementation uses.

Meta provides the language-model weights in a simple way, but a model-architecture to drop them into is still needed. This is provided, in a less obvious way, in the [llama_models](https://github.com/meta-llama/llama-models) repo. The model-architecture class therein relies on both `torch` and `fairscale` and expects each, specifically `torch.distributed` and `fairscale`, to be initialized appropriately. The use of CUDA is hard-coded in a few places in the official repo. I changed that and bundled that version here (as a git submodule).

With those initializations squared away, the model-architecture class can be instantiated. Though, that model is largely a blank slate until we then drop the weights in.

The tokenizer is similarly available in [llama_models](https://github.com/meta-llama/llama-models) and relies on a dictionary-like file distributed along with the model-weights. I'm not sure why, but that file's strings (which map to unique integers or indices) are base64 encoded. Technically, you don't need to know that to use the Tokenizer, but if you're curious to see the actual tokens the system uses, make sure to decode appropriately!

I believe most systems use beam-search rather than greedily taking the most-likely token at each time-step, so I implemented the same. Beam search takes the k (say 5) most likely
tokens at the first time-step and uses them as a seed for k distinct sequences. For all future time-steps, only the most likely token is appended to the sequence. At the end, the overall most likely sequence is selected.

I can pretty comfortably run the 1B model on my Mac M1 Air's cpu with 16GB ram averaging about 1 token per second. The 3B model struggles and gets about 1 token every 60 seconds. And the 8B model typically gets killed by the OS for using too much memory. I tried using `mps` (Metal Performance Shaders), i.e. Apple's GPU, but received all `nan`'s as model output -- not sure why. 




