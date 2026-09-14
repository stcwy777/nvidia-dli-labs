# NVIDIA DLI Labs & RAG Development Environment

My self-study workspace for NVIDIA Deep Learning Institute (DLI) courses, hands-on labs, learning notes, and custom large language model (LLM) experiments.

The current focus is **[Building RAG Agents with LLMs](courses/building-rag-agents-with-llms/README.md)**. This repository starts with a GPU-enabled development environment; course notebooks and experiments will grow as I work through the material.

## Hardware and environment

The target setup is a local deep learning workstation:

| Component | Configuration |
| --- | --- |
| Host OS | Windows 11 |
| Linux environment | WSL 2 with Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB VRAM |
| Container runtime | Native Docker Engine with NVIDIA Container Toolkit |
| Development environment | VS Code Dev Containers |
| Base image | `nvcr.io/nvidia/pytorch:24.03-py3` |

The [dev container configuration](.devcontainer/devcontainer.json) enables GPU access with `--gpus=all` and uses `--network=host` to share the Linux host's network. The checkout is mounted at `/workspace/nvidia-dli-labs`.

VS Code extensions for Python, Jupyter, Pylance, and GitHub Copilot are included in the container configuration.

## Repository structure

```text
nvidia-dli-labs/
|-- .devcontainer/
|   `-- devcontainer.json              # Shared GPU-enabled development environment
|-- .github/
|   `-- copilot-instructions.md        # Repository-specific Copilot guidance
|-- courses/
|   `-- building-rag-agents-with-llms/
|       |-- README.md                 # Course setup, learning plan, and progress
|       |-- notebooks/                # Numbered Jupyter notebooks
|       `-- data/                     # Local course datasets; ignored by Git
|-- playground/                       # Reserved for scratch experiments and prototypes
|-- shared_utils/                     # Reserved for reusable helpers and templates
|-- .gitignore
`-- README.md
```

Git does not track empty directories. Create `data/`, `playground/`, and `shared_utils/` locally as needed; the latter two do not contain shared code or experiments yet.

## Getting started

Before opening the container, make sure the NVIDIA driver supports GPU access in WSL, Docker Engine is running with the NVIDIA Container Toolkit configured, and VS Code has the **WSL** and **Dev Containers** extensions installed.

1. Clone the repository from a WSL Ubuntu terminal and open it in VS Code:

   ```bash
   git clone https://github.com/stcwy777/nvidia-dli-labs.git
   cd nvidia-dli-labs
   code .
   ```

2. Run **Dev Containers: Reopen in Container** from the VS Code Command Palette. The first launch downloads the NVIDIA PyTorch image.
3. In the container terminal, run `nvidia-smi` to confirm that the GPU is visible.
4. Follow the [RAG course README](courses/building-rag-agents-with-llms/README.md) to install the course dependencies, select the container's Python kernel, and open the guided local-GPU notebook.

## Courses and progress

| Course | Status |
| --- | --- |
| [Building RAG Agents with LLMs](courses/building-rag-agents-with-llms/README.md) | Local-GPU notebook scaffolded; first generation exercise pending |

Course-specific dependencies, notes, and progress belong with the corresponding course. Keep scratch work in `playground/`; move genuinely reusable helpers into `shared_utils/` when there is code to share.

## Data and generated artifacts

Keep notebooks, source code, and personal study notes in Git. Store downloaded course data under the course's `data/` directory and keep model weights, caches, checkpoints, and generated outputs in ignored locations.

The [.gitignore](.gitignore) covers common artifact directories such as `data/`, `datasets/`, `models/`, `checkpoints/`, `outputs/`, and `cache/`, along with common model formats and notebook checkpoints. Obtain course materials through NVIDIA DLI and follow their usage terms.

This is a personal learning repository, not an official NVIDIA project.
