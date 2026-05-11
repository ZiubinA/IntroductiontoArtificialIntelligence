# Kohonen SOM — Lecture Code Package

Companion code for the lecture *Kohonen Self-Organizing Maps*.

## Contents
- `som.py` — Self-contained NumPy implementation + image compression CLI.
- `som_lecture.ipynb` — Jupyter notebook walking through the lecture step by step.
- `sample.png` — Tiny test image.
- `README.md` — this file.

## Setup
```bash
pip install numpy pillow matplotlib jupyter
```

## Quick start
Toy color-SOM:
```bash
python som.py demo-colors --grid 30 --iters 5000 --out color_map.png
```

Compress an image:
```bash
python som.py compress sample.png --block 4 --codebook 64 \
    --iters 2000 --out compressed.npz --reconstruct reconstructed.png
```

Decompress:
```bash
python som.py decompress compressed.npz --out reconstructed.png
```

Open the notebook:
```bash
jupyter notebook som_lecture.ipynb
```

## License
MIT — use freely for teaching and learning.
