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
  <img src="imgs/AAAI-Figure1.png" width="700"/>
</p>

**Figure:** (a) Illustration of a patient’s sequential imaging data acquired with irregular inter-visit gaps ∆t (e.g., 2008 → 2012
→ 2015). (b) Different scanning strategies for spatio-temporal feature volumes. (c) The scanning mechanism in the proposed
method ∆t-Mamba3D: time-aware scan modulated by inter-visit gaps ∆t with learnable multi-scale 3D neighborhood fusion.

---

## 📁 Repository Structure

