# Deployment-1-
First assesment for SYS-304

# Digit Recognizer — Deep Learning Baseline Model

This repository contains a complete, production-ready machine learning pipeline that builds, trains, and evaluates a deep vision model to recognize handwritten digits.

## 📌 Proposed Problem
The core challenge is **Handwritten Digit Classification** (or Handwritten Character Recognition). 

Human handwriting is naturally inconsistent, messy, and varies wildly from person to person. Teaching a computer to automatically read handwritten numbers bridges the gap between physical documents and digital databases. 

In the real world, solving this problem allows industries to automate massive manual processing workflows, such as:
* **Banking:** Automatically reading and processing the handwritten dollar amounts on paper check deposits.
* **Logistics:** Scanning and routing postal mail envelopes by reading handwritten zip codes.
* **Administration:** Digitizing historical records, tax forms, and medical questionnaires.

## 📊 Dataset Description
The model is trained on the **Kaggle Digit Recognizer dataset**, which is based on the historic, industry-standard **MNIST** benchmark dataset.

* **Training Set (`train.csv`):** 42,000 human-annotated and labeled examples used to teach the model.
* **Testing Set (`test.csv`):** 28,000 unlabeled examples used to perform final inference verification.
* **Data Format:** Instead of raw image files, the drawings are pre-processed into a spreadsheet format. Every handwritten digit is a **28x28 pixel** grayscale image. Because $28 \times 28 = 784$, each row in the CSV consists of **784 individual pixel values** ranging from `0` (complete black background) to `255` (pure white brush stroke).

## 🤖 Model & Architecture
* **Framework:** PyTorch & `timm` (Torch Image Models)
* **Architecture:** **ResNet18** (Residual Network with 18 layers)
* **Custom Modification:** Standard ResNet engines expect 3-channel color photos (Red, Green, Blue). Because handwriting data is strictly grayscale, the model's initial convolutional layer (`conv1`) was re-engineered to accept a **1-channel input array**.
* **Classifier:** The final layer features a dense 10-class output mapping directly to numerical digits `0` through `9`.

## ⚙️ Core Pipeline Steps
1. **Exploratory Data Analysis (EDA):** Profiles missing data blocks, plots target class balances, and visualizes row arrays back into human-readable pixel grids.
2. **Data Pipeline:** Normalizes 0–255 pixel bounds down to a clean `0.0` to `1.0` range and applies an 85/15 stratified train/validation split.
3. **Training Routine:** Utilizes an `AdamW` optimization loop backed by `CrossEntropyLoss` error scoring across 1 epoch for rapid proof-of-concept testing.
4. **Weights Serialization:** Automatically exports model parameters to a standalone `resnet18_mnist_baseline.pt` file.
5. **Saved-Weight Inference:** Safely skips the entire training path by fetching the saved `.pt` weights directly from the filesystem to classify unseen testing data instantly.
