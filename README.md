# Resource-Constrained CNNs for Edge Image Classification

This repository contains the code and evaluation report for EN3150 Assignment 03. The primary objective of this project is to develop and analyze image classification models optimized for low-resource embedded devices (e.g., microcontrollers, edge nodes) where strict memory and computational constraints apply.

## Project Scope
* **Custom CNN Development:** 
  * **Model A (Standard):** A baseline architecture using standard 2D convolutions and max-pooling layers.
  * **Model B (Lightweight):** A highly optimized network utilizing depthwise separable convolutions, strictly capped at under 100,000 trainable parameters.
* **Optimization Analysis:** Evaluating the convergence and performance impacts of different optimizers, including standard SGD, SGD with Momentum, and a custom-selected optimizer.
* **State-of-the-Art (SOTA) Comparison:** Fine-tuning industry-standard lightweight models (such as MobileNet, SqueezeNet, or EfficientNet-B0) to serve as performance benchmarks.
* **Trade-off Evaluation:** A comprehensive comparison of test accuracy, parameter count, disk size footprint, and computational efficiency across all trained models.

## Dataset Setup
To emulate a low-resource sensor environment, the selected dataset features images downscaled to a maximum resolution of 64x64 pixels. The data is partitioned into a 70% training, 15% validation, and 15% testing split.
