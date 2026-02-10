# Day 02 – Shortcut Learning & Spurious Correlations

## 🚀 99 Days of AI Prototypes

In this experiment, we demonstrate how neural networks can learn **shortcuts** instead of real patterns.

We train a model on MNIST digits, but we intentionally inject a **spurious colored patch** correlated with each label.

The model achieves high accuracy.

Then we remove the patch.

Performance collapses.

This shows the model was not learning digits — it was learning the shortcut.

---

## 🧠 Why This Matters

This experiment illustrates:

- Shortcut learning
- Dataset bias
- Spurious correlations
- Distribution shift
- Robustness failure
- Why causality matters in ML

Modern deep learning models often exploit the easiest signal available — not necessarily the correct one.

---

## 🏗 What This Code Does

1. Loads MNIST
2. Injects a label-correlated colored patch
3. Trains a CNN on biased data
4. Evaluates on:
   - Biased test set
   - Clean test set
5. Compares performance under distribution shift

---

## 📊 Expected Outcome

- High accuracy on biased data
- Significant drop on clean data
- Clear demonstration of shortcut learning

---

## 🛠 Installation

Create a virtual environment (recommended):

```bash
conda create -n ai99 python=3.10
conda activate ai99
