# Digit Recognition App — From Scratch to CNN

A hands-on deep learning project that teaches you to build a handwritten digit classifier from the ground up — starting with pure NumPy math and finishing with an interactive drawing widget powered by a trained CNN.

Built as a mini-project for the **Neural Networks & Deep Learning** module of the VectorBrainAI AI/ML Bootcamp.

---

## What You Will Build

By the end of this notebook you will have:

- A **two-layer MLP trained from scratch** in NumPy (no frameworks — just matrix math and calculus)
- A **PyTorch MLP and CNN** trained on the full 60,000-image MNIST dataset
- A **test-set evaluation** comparing both models on 10,000 unseen digits
- A **saved model** you can reload without retraining
- A **learning rate experiment** showing what happens when hyperparameters go wrong
- An **interactive drawing widget** where you draw a digit and watch the CNN predict it in real time

---

## Concepts Covered

| Topic | Where |
|-------|-------|
| Perceptrons & Multi-Layer Perceptrons | Section 2 |
| Forward propagation & activation functions (ReLU, Softmax) | Section 2 |
| Cross-entropy loss & loss gradients | Section 2 — Step 3 |
| Backpropagation & the chain rule (symbolic + code-mapped) | Section 2 — Step 3 |
| Mini-batch vs. full-batch gradient descent | Section 3 |
| Convolutional Neural Networks — filters, feature maps, pooling | Section 3 |
| Recurrent Neural Networks — hidden state, sequence prediction | Section 4 |
| Hyperparameter sensitivity (learning rate) | Experiment cell |
| Model persistence (save / load) | Save & Load cell |

---

## Project Structure

```
Digit_Recognition_APP_finalized.ipynb   ← Main notebook (all sections)
mnist_cnn.pth                          ← Saved CNN weights (generated on first run)
data/                                  ← MNIST dataset (auto-downloaded on first run)
```

---

## Requirements

### Python

Python 3.8 or higher is recommended.

### Libraries

Install all dependencies with:

```bash
pip install torch torchvision matplotlib numpy ipywidgets ipycanvas
```

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | PyTorch framework + MNIST download |
| `matplotlib` | Image visualization and training plots |
| `numpy` | From-scratch neural network math (Section 2) |
| `ipywidgets` | Interactive UI controls for the drawing widget |
| `ipycanvas` | Drawing canvas for the interactive demo |

> **Note:** `ipywidgets` and `ipycanvas` are both required for the interactive widget in the final cell. If either is missing, the cell will print clear installation instructions and halt gracefully.

---

## How to Run

1. Clone or download this repository.
2. Install dependencies (see above).
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open Digit_Recognition_APP_finalized.ipynb.
5. Run all cells top to bottom (**Kernel → Restart & Run All**).

The MNIST dataset (~11 MB) will download automatically on first run into a local `data/` folder.

---

## Notebook Sections at a Glance

### Section 1 — Data
Downloads both the training (60,000) and test (10,000) MNIST splits, normalises pixel values to [0, 1], and visualises the first 10 training images.

### Section 2 — NumPy MLP (From Scratch)
Builds a 784 → 128 → 10 MLP using only NumPy. Every operation — weight initialisation, forward propagation, cross-entropy loss, backpropagation via the chain rule, and gradient descent — is implemented explicitly, with each line of code tied to its corresponding calculus expression.

### Section 3 — PyTorch MLP → CNN
Rebuilds the same MLP in PyTorch to show how `nn.Linear`, `Autograd`, and `DataLoader` replace the manual math from Section 2. Then upgrades the architecture to a CNN (`Conv2d` + `MaxPool2d`) with a full explanation of filters, feature maps, stride, padding, and dimension calculations. Both models are evaluated on the held-out test set.

### Section 4 — RNN Introduction
Introduces Recurrent Neural Networks: the hidden state concept, the recurrence formula, and a toy next-digit sequence prediction task using PyTorch's `nn.RNN`. Includes a note on why LSTMs and GRUs are preferred over plain RNNs for longer sequences.

### Hyperparameter Experiment
Re-trains the NumPy MLP at three learning rates (`0.05`, `0.5`, `5.0`) from identical starting weights and plots all three accuracy curves on one chart, demonstrating underfitting, stable convergence, and divergence.

### Model Save / Load
Saves the trained CNN's `state_dict` to `mnist_cnn.pth`, reloads it into a fresh model instance, and verifies identical test accuracy — demonstrating how to persist and deploy a trained model.

### Interactive Demo
An `ipycanvas`-powered drawing widget. Draw any digit on the 280×280 canvas, click **Predict**, and see the CNN's predicted class and a bar chart of all 10 class probabilities.

---

## Expected Results

| Model | Expected Test Accuracy |
|-------|----------------------|
| NumPy MLP (100 epochs, lr=0.5) | ~92–94% (training accuracy) |
| PyTorch MLP (3 epochs, SGD lr=0.1) | ~97% |
| PyTorch CNN (3 epochs, Adam lr=0.001) | ~98–99% |

Results may vary slightly depending on hardware and library versions. The fixed random seed (`np.random.seed(42)`) ensures reproducible NumPy results.

---

## Learning Path

This notebook is designed to be read in order. Each section builds on the previous:

```
NumPy math  →  PyTorch abstraction  →  CNN upgrade  →  RNN concepts  →  Real app
```

If you are new to neural networks, work through Section 2 carefully — the symbolic chain rule derivations and line-by-line code comments are the core learning material.

---

## License

For educational use as part of the VectorBrainAI AI/ML Bootcamp curriculum.
