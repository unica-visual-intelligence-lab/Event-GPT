# Event-GPT

Event-GPT: Sequence-Aware Video Event Classification via LoRA-Tuned GPT

This repository provides a pipeline for video event classification (e.g., Fire/Smoke) that combines a video encoder (VideoMAE v2) with an autoregressive model (GPT-2) adapted with LoRA for multi-label classification.


## Features
- Extracts video embeddings with `OpenGVLab/VideoMAEv2-Base`.
- Projects embeddings and classifies with `openai-community/gpt2` (multi-label classification head) adapted via LoRA.
- Training with light augmentations (Albumentations) and evaluation with macro F1.
- Ready-to-run inference script with sample videos in `submission/foo_videos`.

## Repository structure
- `code/`
  - `model.py`: defines the `EventClassifier` model (VideoMAE → Linear → GPT‑2 LoRA).
  - `train_multi.py`: training + validation (macro F1, saves the best model).
  - `helpers.py`: utilities for parsing RTF labels and pairing videos↔labels.
  - `inference.ipynb`, `analyze_videos.ipynb`, `get_results.ipynb`: notebooks for inspection/inference.
- `submission/`
  - `model.py`: model copy for test/evaluation.
  - `model.pth`: example pre-trained weights.
  - `test.py`: inference script over a folder of videos (`--videos`) producing `.txt` files with event seconds.
  - `foo_videos/`, `foo_results/`: sample inputs/outputs.
- `requirements.txt`: Python dependencies.
- `LICENSE`: project license.


## Requirements
- Python 3.12 (recommended) — tested with CPython 3.12.
- NVIDIA GPU recommended. The PyTorch versions pinned in `requirements.txt` use CUDA 12.8 (suffix `+cu128`). If your environment differs, install a compatible PyTorch from the official site first and then the remaining dependencies.


## Setup (Windows, PowerShell)
1) Create and activate a virtual environment:
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2) Install PyTorch according to your GPU/CUDA (recommended):
- Visit https://pytorch.org/get-started/locally/ and follow the Windows instructions.
- Example (reference only, adjust to your CUDA/CPU):
```
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

3) Install the remaining dependencies:
```
pip install -r requirements.txt
```
If installation fails for `torch/torchvision/torchaudio` due to the `+cu128` pin, install PyTorch as in step 2 first and then either edit/remove those lines in `requirements.txt` or run:
```
pip install -r requirements.txt --no-deps
```
and manually install any missing dependencies.


## Dataset and annotations
The training script expects root folders for videos and labels, each containing subfolders with matching files by index/sort order:
- Videos: `.mp4` files organized in subfolders under `--videos`.
- Labels: `.rtf` files organized in subfolders under `--labels`.

The parser `helpers.get_rtf_text(path)` converts RTF to plain text and expects a format like `timestart,int,<class(es)>` (comma-separated). Examples:
```
12,Fire
```
or
```
8,Smoke
```
Notes:
- Currently supported classes: `No event`, `Fire`, `Smoke`.
- Multiple classes are treated as multi‑label during training.


## Training
Example run (adjust paths):
```
python code/train_multi.py --videos E:/datasets/Fire/videos --labels E:/datasets/Fire/labels --fps 1 --output output --seed 42 --use_augmentations True
```
Main arguments:
- `--videos`: root folder with videos (with subfolders).
- `--labels`: root folder with `.rtf` labels (with subfolders).
- `--fps`: output sampling rate (default 1). Frames are sampled relative to input FPS.
- `--output`: where to save the best model (`output/event_classifier.pth`).
- `--use_augmentations`: enables light augmentations on frames.

Training runs validation every epoch computing macro F1. When F1 improves, the model is saved to `output/event_classifier.pth`.

Hardware/Notes:
- `NUM_FRAMES_INPUT = 16`. Windows are built by sampling/padding to 16 frames.
- The video encoder is frozen by default; GPT‑2 is adapted with LoRA to reduce trainable parameters.


## Quick inference (submission script)
Sample videos and a ready script are provided.

1) Move into `submission/`:
```
cd submission
```

2) Run on sample videos:
```
python test.py --videos foo_videos --results foo_results
```
- Loads `submission/model.pth` and writes, for each video, a `.txt` file with the estimated event start second (if detected) into `--results` (default `foo_results/`).

To use your own videos, prepare a folder with `.mp4` files and pass its path to `--videos`. To use your trained model, replace `submission/model.pth` with your weights (same architecture as in `submission/model.py`).


## Inference from a trained model (training output)
If you trained with `code/train_multi.py`, the best weights are in `output/event_classifier.pth`. You can:
- Use `code/inference.ipynb` by setting `PATH_TO_MODEL` to your file and run the notebook to generate CSVs/analysis.
- Or adapt a small script based on `submission/test.py`, pointing `PATH_TO_MODEL` to your weights and using `code/model.py` architecture.


## Troubleshooting
- PyTorch/CUDA: the `+cu128` pins may not match your environment. Install PyTorch from the official site first, then other deps. Verify with:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
- GPU memory: reduce input resolution (frames are resized to 224×224) or lower output sampling `--fps`.
- Label format: ensure `.rtf` contains at least `timestart,int` and a class (`Fire`/`Smoke`). When parsing is empty, `helpers.get_rtf_text` returns `['No event']`.


## License
See `LICENSE`.


## Acknowledgements / Models
- Video encoder: OpenGVLab/VideoMAEv2-Base
- Language model: openai-community/gpt2
- LoRA: PEFT (https://github.com/huggingface/peft)

If you use this project in academic or industrial work, please cite the base models and this repository.
