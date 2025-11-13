# 🐍 ∆t-Mamba3D: A Time-Aware Spatio-Temporal State-Space Model for Breast Cancer Risk Prediction 
[![arXiv](https://img.shields.io/badge/arXiv-2510.19003-b31b1b.svg)](https://arxiv.org/abs/2510.19003) •  
Full paper + Appendix available on arXiv  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Mamba-based spatiotemporal model that captures longitudinal changes in screening mammography for **early cancer risk prediction**.  
This repository provides official PyTorch implementation, pretrained weights, and reproducible experiments.

---

## 📌 Overview

Early cancer detection benefits from modeling *temporal evolution* of breast tissue across multiple prior exams.  
However, existing methods treat mammograms independently or poorly account for temporal dependencies.

**LongitudinalMamba** introduces:

- ✔️ **Δt-aware state space modeling**  
- ✔️ **3D spatial–temporal feature mixing**  
- ✔️ **Robust longitudinal exam alignment**  
- ✔️ **Superior performance over Time-aware and Spatial-temporal baselines**

---

## 📊 Method Summary

<p align="center">
  <img src="assets/method_overview.png" width="700"/>
</p>

**Figure:** Overall architecture of LongitudinalMamba.  
The model processes each exam as a spatial feature map, mixes temporal states using Mamba layers, then aggregates for cancer risk prediction.

---

## 🧪 Experimental Results

<p align="center">
  <img src="assets/main_results.png" width="650"/>
</p>

- Achieves **+X% improvement** over ResNet/ViT baselines  
- Demonstrates **strong temporal consistency**  
- Performs well even under imbalanced longitudinal sequence availability

---

## 📁 Repository Structure

