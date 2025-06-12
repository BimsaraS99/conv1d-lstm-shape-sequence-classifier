# Conv1D-LSTM Shape Classifier

This repository contains a deep learning-based shape classifier that uses a hybrid **1D Convolutional Neural Network (Conv1D)** followed by an **LSTM** to classify sequential shape data into two categories: **heart** and **circle**. Each sample is a sequence of 2D points representing a contour, and the first digit in the `.csv` file denotes the class label (0 for heart, 1 for circle).

https://github.com/user-attachments/assets/34b89c3f-3a90-4851-9473-e115b1e7e259

---

## 📌 Project Overview

The goal of this project is to classify geometric shapes represented by sequences of 2D coordinates using a hybrid Conv1D + LSTM model. The model captures **local spatial patterns** via convolution and **temporal dependencies** through LSTM.

This architecture is particularly suitable for:

- Shape contour classification  
- Sequential pattern recognition in geometric data  
- Edge-device inference with lightweight models  

---

## 🧠 Model Architecture

The model architecture is composed of the following layers:

1. **Conv1D Layer**
   - Input: 2D coordinates (shape: `[batch_size, sequence_length, 2]`)
   - Applies a 1D convolution across the coordinate sequence
   - Captures spatial structure in local regions

2. **ReLU Activation**

3. **MaxPool1D**
   - Reduces dimensionality and emphasizes prominent features

4. **LSTM Layer**
   - Processes the pooled feature sequence
   - Captures temporal relationships and dependencies

5. **Fully Connected (Linear) Layer**
   - Reduces to a single output logit for binary classification

6. **Sigmoid Activation**
   - Converts logit to probability (Not included in the model but in the loss function)

---

## 🧾 Data Format

Each `.csv` file contains a single sample.  
The first value in the file is the **class label** (0 or 1).  
The remaining values are flattened 2D points:  
Example:

0, x1, y1, x2, y2, ..., xn, yn

1, x1, y1, x2, y2, ..., xn, yn
