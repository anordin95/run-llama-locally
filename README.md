## Running Llama locally with minimal dependencies

### Motivation

I want to peel back the layers of the onion and other gluey-mess to gain insight into these models. 

There are other popular ways to invoke these models, such as [Ollama](https://ollama.com/) and Hugging-Face's general API package: [transformers](https://pypi.org/project/transformers/), but those hide the interesting details behind an API. 
I was a bit surprised Meta didn't publish an example way to simply invoke one of these LLM's with only `torch` (or some minimal set of dependencies), though I am obviously grateful and so pleased with their contribution of the public weights! 

### Setup steps

1. Download the relevant model weight(s) via https://www.llama.com/llama-downloads/

2. `$ pip install -r requirements.txt`

3. `$ cd llama-models; pip install -e .; cd ..`

4. `$ python minimal_run_inference.py` or `$ python run_inference.py`

### Exploring the model & outputs

`run_inference.py` is more bloated than `minimal_run_inference.py`. It implements beam-search & features far more explanatory comments. 

`minimal_run_inference.py` is a simple, few lines of code way to run the Llama models. It's a great place to start hacking around or exploring on your own. If one of the steps in it doesn't make sense, peek over at `run_inference.py` where there are likely detailed comments.

### Script parameters

The global variables in the `run_inference.py` scripts: `MODEL_NAME`, `LLAMA_MODELS_DIR`, `INPUT_STRING` and `DEVICE` take the values you'd expect (there are adjacent comments with examples and more details too). They should be modified as you see fit.

### Technical Overview 

#### Dependencies

The minimal set of dependencies I found includes `torch` (perhaps, obviously), a lesser known library also published by Meta: `fairscale`, which implements a variety of highly scalable/parallelizable analogues of `torch` operators and `blobfile`, which implements a general file I/O mechanism that Meta's Tokenizer implementation uses.

Meta provides the language-model weights in a simple way, but a model-architecture to drop them into is still needed. This is provided, in a less obvious way, in the [llama_models](https://github.com/meta-llama/llama-models) repo. The model-architecture class therein relies on both `torch` and `fairscale` and expects each, specifically `torch.distributed` and `fairscale`, to be initialized appropriately. The use of CUDA is hard-coded in a few places in the official repo. I changed that and bundled that version here (as a git submodule).

With those initializations squared away, the model-architecture class can be instantiated. Though, that model is largely a blank slate until we then drop the weights in.

The tokenizer is similarly available in [llama_models](https://github.com/meta-llama/llama-models) and relies on a dictionary-like file distributed along with the model-weights. I'm not sure why, but that file's strings (which map to unique integers or indices) are base64 encoded. Technically, you don't need to know that to use the Tokenizer, but if you're curious to see the actual tokens the system uses, make sure to decode appropriately!

#### Beam-search

I believe most systems use beam-search rather than greedily taking the most-likely token at each time-step, so I implemented the same. Beam search takes the k (say 5) most likely
tokens at the first time-step and uses them as a seed for k distinct sequences. For all future time-steps, only the most likely token is appended to the sequence. At the end, the overall most likely sequence is selected.

#### Performance notes

Using CPU, I can pretty comfortably run the 1B model on my Mac M1 Air's that has 16GB of RAM averaging about 1 token per second. The 3B model struggles and gets about 1 token every 60 seconds. And the 8B model typically gets killed by the OS for using too much memory. 

Initially, using `mps` (Metal Performance Shaders), i.e. Apple's GPU, would produce all `nan`'s as model output. The issue is due to a known-bug in `torch.triu` which I implemented a workaround for in the `llama-models` git submoudle. 

With `mps`, the inference time of the first few tokens on the 1B model is notably faster, but the memory usage is much higher. It's not entirely clear to me why the memory usage differs so notably, particularly given Apple's unified memory layout (i.e. cpu & gpu share memory). Once the sequence is about 100 or 200 tokens, the throughput slows down notably -- about half of the cpu's throughput. I suspect that the relatively higher memory-load of the GPU (caused for unknown reasons) in conjunction with a growing sequence length starts to swamp my system's available memory to a degree which effects the computation speed.

Aside on GPU memory: I'm using a batch-size of 1, so there's no batch parallelism (i.e. presumably multiple full models in memory). And, the memory of each transformer layer should be relatively constant, unless perhaps each attention-heads' parameters are loaded into memory then discarded, whereas in the parallel (i.e. gpu) case all heads are simultaneously loaded AND that difference is enough to cause a notable change in memory-load. If you know why, drop a note!




