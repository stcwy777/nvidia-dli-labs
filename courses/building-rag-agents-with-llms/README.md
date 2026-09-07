# Building RAG Agents with LLMs

Personal learning notes, notebooks, and experiments for the NVIDIA DLI **Building RAG Agents with LLMs** course.

The goal is to understand and build retrieval-augmented generation (RAG) workflows: retrieving relevant information from a document collection and using that context to support an LLM's response.

[Back to the repository README](../../README.md)

## Current status

The shared GPU-enabled dev container is configured. The first notebook, [`01_rag_environment_check.ipynb`](notebooks/01_rag_environment_check.ipynb), is currently an empty placeholder, not a runnable Jupyter notebook. Course labs and a working RAG pipeline have not been implemented yet.

## Personal learning plan

- [ ] Initialize the environment-check notebook and confirm Python, PyTorch, and CUDA access.
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

There is no course-specific dependency manifest yet. As labs are added, record their required packages and versions here or in a course-local dependency file. The base PyTorch image alone should not be assumed to provide every RAG library used by a lab.

### Initialize the environment-check notebook

1. In the container's VS Code window, run **Jupyter: Create New Jupyter Notebook**.
2. Save it as `courses/building-rag-agents-with-llms/notebooks/01_rag_environment_check.ipynb`, replacing the empty placeholder.
3. Select a Python kernel from the dev container, not a Windows or WSL-host interpreter.
4. Run this cell to inspect the runtime and perform a small GPU operation:

```python
import sys

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)

assert torch.cuda.is_available(), "CUDA is not available in this notebook kernel."
print("GPU:", torch.cuda.get_device_name(0))

matrix = torch.ones((2, 2), device="cuda")
print("GPU matrix product:", matrix @ matrix)
```

On the target workstation, the GPU should be reported as an NVIDIA GeForce RTX 4090. If CUDA is unavailable, check the selected kernel and run `nvidia-smi` in the container terminal before investigating the WSL driver or NVIDIA Container Toolkit setup.

## Course organization

Keep lab notebooks in `notebooks/`, using the existing two-digit sequence and descriptive names, such as `02_<topic>.ipynb`. Keep explanations, observations, and small examples alongside the related notebook work.

Store local documents and datasets in this course's `data/` directory. From the repository root, create it with:

```bash
mkdir -p courses/building-rag-agents-with-llms/data
```

The directory is ignored by Git, so downloaded documents are not included in a clone. Record their source and any preparation steps in the notebook that uses them. Keep generated indexes, model caches, and outputs under ignored directories such as `cache/` or `outputs/`.

For labs that use hosted models or embedding services, follow the lab's authentication instructions and supply credentials at runtime rather than storing them in notebooks or tracked files.

## Study workflow

Work through the numbered notebooks in order, beginning with the environment check. For each experiment, record the data source, retrieval settings, model or service used, and observations about the results. Update the learning checklist as topics are completed.

Use the repository's `playground/` for experiments outside the course sequence, and promote reusable code to `shared_utils/` only when it is shared across labs.
