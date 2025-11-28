# Training VAE on ARC-AGI-2 Dataset

## How to Run

Sync the dependencies:

```bash
uv sync
```

Train and evaluate the VAE model on MNIST:

```bash
uv run ./mnist.py --model-type vae
```

Train and evaluate the CVAE model on MNIST:

```bash
uv run ./mnist.py --model-type cvae
```

Download the ARC-AGI-2 dataset:

```bash
uv run ./scripts/download_arc.py
```

Train and evaluate the VAE model on ARC-AGI-2:

```bash
uv run ./arc_vae.py --epochs 50 --batch-size 32
```

Train and evaluate the CVAE model on ARC-AGI-2:

```bash
uv run ./arc_cvae.py --epochs 50 --batch-size 32
```

Make documentation:

```bash
typst compile docs.typ
```
