# Building RAG Agents with LLMs

Personal learning notes, notebooks, and experiments for the NVIDIA DLI **Building RAG Agents with LLMs** course.

The goal is to understand and build retrieval-augmented generation (RAG) workflows: retrieving relevant information from a document collection and using that context to support an LLM's response.

[Back to the repository README](../../README.md)

## Current status

The shared GPU-enabled dev container is configured. [`01_rag_environment_check.ipynb`](notebooks/01_rag_environment_check.ipynb) is a guided local-GPU adaptation of the course's LLM-services material, using **Qwen3-4B with Transformers** instead of NVIDIA-hosted endpoints. Setup is provided, with one learner-owned local generation operation and a checkpoint. Streaming, thinking-mode handling, LangChain integration, and a working RAG pipeline are later milestones, not completed implementations.

## Personal learning plan

- [x] Initialize a guided local-GPU environment-check notebook.
- [ ] Confirm Python, PyTorch, and CUDA access, then complete the local generation checkpoint.
- [ ] Add local text streaming and distinguish reasoning, final responses, and local metadata.
- [ ] Explore document loading, text splitting, and chunking choices.
- [ ] Generate embeddings and build a searchable document index.
- [ ] Retrieve relevant context and construct grounded prompts.
- [ ] Combine retrieval and generation in a RAG agent workflow.
- [ ] Evaluate answer quality, retrieval relevance, and failure cases.
- [ ] Record lessons learned and extend a course example with my own documents.

The conceptual workflow to explore is:

```text
Documents -> Chunking -> Embeddings -> Searchable index
Question -> Retrieval from the index -> Context + prompt -> LLM response
```

## Environment setup

Use the repository-level [dev container](../../.devcontainer/devcontainer.json) rather than a separate course container. Follow the [root setup instructions](../../README.md#getting-started) first.

The workspace inside the container is `/workspace/nvidia-dli-labs`, and this course lives at `/workspace/nvidia-dli-labs/courses/building-rag-agents-with-llms`.

Install the local-inference dependencies from the repository root in the container terminal:

```bash
python -m pip install -r courses/building-rag-agents-with-llms/requirements.txt
```

[`requirements.txt`](requirements.txt) pins a Qwen3-compatible Transformers release, includes `ipywidgets` for notebook progress bars, and leaves PyTorch/CUDA to the existing container image. It does not install an endpoint client or the packages for later LangChain milestones. Restart the notebook kernel after installing dependencies.

If `tqdm` reports `IProgress not found`, install these dependencies in the selected kernel's environment and restart that kernel. This warning concerns widget-based progress bars, not CUDA or model inference; upgrading the container's Jupyter or PyTorch installation is not required for this fix in VS Code.

### Run the first local-inference milestone

1. Open [`01_rag_environment_check.ipynb`](notebooks/01_rag_environment_check.ipynb) and select a Python kernel from the dev container, not a Windows or WSL-host interpreter.
2. Run the runtime cell to inspect Python, PyTorch, CUDA, available VRAM, and a small GPU operation.
3. Provide a complete local `Qwen/Qwen3-4B` snapshot under `models/Qwen3-4B/`, relative to this course directory, or change `MODEL_DIR` to its existing location. Alternatively, explicitly enable `DOWNLOAD_MODEL` in the notebook to download the public model once; this needs internet access and roughly 8 GB for weights, plus supporting files.
4. Run the local-only model loader and prompt preparation, complete the single `generate_local` TODO, and run its checkpoint. The placeholder deliberately raises `NotImplementedError` until implemented.

The target GPU is an NVIDIA GeForce RTX 4090 with 24 GB VRAM. The notebook explicitly uses BF16 weights, keeps model inputs on the same GPU, and does not silently fall back to CPU execution or a hosted service. Weight memory is only part of the requirement: allow room for the KV cache, temporary tensors, and host RAM while loading. If CUDA is unavailable, check the selected kernel and run `nvidia-smi` in the container terminal before investigating the WSL driver or NVIDIA Container Toolkit setup.

This first checkpoint uses non-thinking, non-streaming generation. Qwen3's chat-template thinking option and decoded text stream differ from `ChatNVIDIA`'s structured chunks; the notebook outlines those follow-on milestones without implementing them. See the [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B) for model-specific behavior and sampling guidance.

## Course organization

Keep lab notebooks in `notebooks/`, using the existing two-digit sequence and descriptive names, such as `02_<topic>.ipynb`. Keep explanations, observations, and small examples alongside the related notebook work.

Store local documents and datasets in this course's `data/` directory. From the repository root, create it with:

```bash
mkdir -p courses/building-rag-agents-with-llms/data
```

The directory is ignored by Git, so downloaded documents are not included in a clone. Record their source and any preparation steps in the notebook that uses them. Keep generated indexes, model caches, and outputs under ignored directories such as `cache/` or `outputs/`.

The first notebook needs no NVIDIA API key or hosted inference endpoint. Its optional model download is separate from inference; after files are available, loading is local-only and prompts are processed on the local GPU. Keep weights and generated outputs out of Git, and clear notebook outputs before committing. If a later lab explicitly uses hosted services, supply credentials at runtime rather than storing them in notebooks or tracked files.

## Study workflow

Work through the numbered notebooks in order, beginning with the environment check. Complete one coaching milestone and review its checkpoint before adding the next. For each experiment, record the data source, retrieval settings, model or service used, and observations about the results. Update the learning checklist as topics are completed.

Use the repository's `playground/` for experiments outside the course sequence, and promote reusable code to `shared_utils/` only when it is shared across labs.
