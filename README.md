# Visual Question Answering (VQA) with Easy-VQA

This project implements a **Visual Question Answering (VQA)** model using the [Easy-VQA](https://pypi.org/project/easy-vqa/) dataset. The model combines a pre-trained EfficientNet-B0 CNN for image feature extraction and a DistilBERT transformer for question encoding, fusing the representations to predict one of 20 possible answers.

## Overview

The goal of VQA is to answer natural language questions about images. In this implementation:

- Images are passed through EfficientNet-B0 (pretrained on ImageNet) to obtain a visual feature vector.
- Questions are tokenized and encoded using DistilBERT (pretrained, kept frozen during training).
- The two feature vectors are concatenated and fed into a small multilayer perceptron (MLP) that outputs a probability distribution over the answer classes.

The model is trained on the Easy-VQA dataset, which contains simple synthetic images (colored shapes) and corresponding questions (e.g., "What color is the square?"). After 10 epochs, the model achieves **98.41% test accuracy**.

## Requirements

The notebook is designed to run in a **Google Colab** environment with a GPU (T4 is used). All dependencies can be installed via pip:

```bash
pip install easy-vqa transformers peft torch torchvision matplotlib
```

- `easy-vqa` – provides the dataset and helper functions.
- `transformers` – for DistilBERT tokenizer and model.
- `torch` and `torchvision` – for deep learning and image preprocessing.
- `matplotlib` – for visualization.
- `peft` – installed but not actively used in the current notebook (may be a leftover).

## Dataset: Easy-VQA

[Easy-VQA](https://pypi.org/project/easy-vqa/) is a lightweight synthetic dataset for VQA. It contains images with simple geometric shapes (squares, circles, etc.) in different colors. Each image is associated with several questions and ground‑truth answers.

- **Train set**: 4000 images, 20 possible answers.
- **Test set**: 1000 images.

The dataset is automatically downloaded when using the `easy_vqa` package.

## Model Architecture

The model consists of three main components:

1. **Image Encoder**  
   - EfficientNet-B0 (pretrained) → `features` layer → adaptive average pooling → 1280‑dim vector.

2. **Text Encoder**  
   - DistilBERT (pretrained, frozen) → tokenization with max length 15 → mean pooling of last hidden state → 768‑dim vector.

3. **Fusion & Classifier**  
   - Concatenate image and question features → 1280+768 = 2048 dim.  
   - Two fully‑connected layers (2048 → 512 → 20) with ReLU, dropout (0.3), and no activation on the final layer (raw logits).

All parameters of DistilBERT are frozen (`requires_grad = False`) to reduce training time and leverage pre‑trained language representations.

## Training Details

- **Optimizer**: Adam with learning rate 5e-4 (only trainable parameters).
- **Loss function**: Cross‑entropy.
- **Batch size**: 32.
- **Epochs**: 10 (with early stopping patience of 3, but no early stop occurred).
- **Hardware**: GPU (NVIDIA T4) in Google Colab.

The training logs show steady improvement:

```
Epoch 01 | Loss 0.5246 | Acc 86.18%
Epoch 02 | Loss 0.3595 | Acc 86.82%
Epoch 03 | Loss 0.3504 | Acc 87.27%
Epoch 04 | Loss 0.3164 | Acc 88.14%
Epoch 05 | Loss 0.2976 | Acc 91.01%
Epoch 06 | Loss 0.2431 | Acc 93.64%
Epoch 07 | Loss 0.2021 | Acc 94.32%
Epoch 08 | Loss 0.1553 | Acc 95.72%
Epoch 09 | Loss 0.1266 | Acc 98.27%
Epoch 10 | Loss 0.1068 | Acc 98.41%
```

The best model (highest test accuracy) is saved and used for inference.

## Results

On the test set, the model achieves **98.41% accuracy**. Below are a few example predictions from the notebook:

<img width="437" height="492" alt="Screenshot 2026-03-09 170355" src="https://github.com/user-attachments/assets/8cac052d-3b83-469c-994e-fe49df5b473b" />


<img width="510" height="492" alt="Screenshot 2026-03-09 170500" src="https://github.com/user-attachments/assets/d6075123-b85b-4cf2-9038-ec68ad10903a" />


All examples shown are correct, confirming the model's strong performance on this synthetic dataset.

## Usage

To run the notebook yourself:

1. Open [VQA.ipynb](VQA.ipynb) in Google Colab or a local Jupyter environment.
2. Install the required packages (first cell).
3. Execute all cells. The notebook will:
   - Download the Easy-VQA dataset.
   - Load pre‑trained models (EfficientNet, DistilBERT).
   - Train the VQA model for 10 epochs.
   - Display sample test predictions with images.

If you want to train for more epochs or modify hyperparameters, adjust the `range(10)` in the training loop or the learning rate.

