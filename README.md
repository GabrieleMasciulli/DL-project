# Few-Shot Adaptation of Vision-Language Models

### Deep Learning 2025 Project Assignment

![Project Banner](https://img.shields.io/badge/Deep%20Learning-2025-blue?style=for-the-badge&logo=pytorch)
![Model](https://img.shields.io/badge/Model-CLIP%20ViT--B%2F16-orange?style=for-the-badge)

## 📌 Project Overview

This project addresses the challenge of **Few-Shot Adaptation** in Vision-Language Models (VLMs), specifically **CLIP**. The objective is to adapt a pre-trained model to a set of **Base classes** using limited annotated examples (10 shots) while preserving the model's zero-shot generalization capabilities on unseen **Novel classes**.

Standard approaches like Prompt Tuning (e.g., CoOp) or Visual Adapters (e.g., CLIP-Adapter) often suffer from overfitting to the few-shot data, causing "catastrophic forgetting" of the novel classes. This project introduces a **Teacher-Student Framework** with dual-modality regularization to solve this issue.

### 🏆 Key Results

This method significantly improves Base accuracy while maintaining Novel accuracy, outperforming the Zero-Shot baseline on the Oxford Flowers 102 dataset.

| Model                         | Base Accuracy (Seen) | Novel Accuracy (Unseen) | Harmonic Mean (HM) |
| :---------------------------- | :------------------: | :---------------------: | :----------------: |
| **CLIP Zero-Shot (ViT-B/16)** |        72.10%        |       **77.70%**        |       74.79%       |
| **Our Method**                |      **94.80%**      |         76.50%          |     **84.67%**     |

---

## 🧠 Motivation & Problem Analysis

Existing State-of-the-Art (SOTA) methods often treat vision and language modalities separately.

1.  **Prompt Learning (e.g., CoOp):** Optimizes text context but ignores visual feature adaptation.
2.  **Visual Adapters (e.g., CLIP-Adapter):** Optimizes visual features but lacks linguistic context.

**Empirical Analysis:**
Preliminary studies show that performance degradation on Novel classes is strongly correlated with how far the learned embeddings "drift" from the original frozen CLIP embeddings.

<div align="center">
<table>
  <tr>
    <td align="center">
      <img src="https://i.postimg.cc/28r653Yw/clip-analysis.png" alt="Text Embedding Drift Analysis" width="600"/>
      <br />
      <em>Fig 1: Drifting too far from the original prompts correlates with reduced generalization.</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://i.postimg.cc/L4z5RLcg/clip-adapter-analysis.png" alt="Visual Feature Drift Analysis" width="600"/>
      <br />
      <em>Fig 2: Divergence in visual feature space correlates with performance drops on novel classes.</em>
    </td>
  </tr>
</table>
</div>

---

## 🚀 Methodology

To address the drift issue, a **Teacher-Student Framework** is used:

- **Teacher:** Frozen CLIP (ViT-B/16).
- **Student:** Custom CLIP initialized with:
  1.  **Learnable Text Context (CoOp style):** Adapts the language modality.
  2.  **Visual Adapter (Bottleneck, CLIP-Adapter style):** Adapts the vision modality.

### Loss Function

The student is trained to minimize a composite loss function that enforces task learning while constraining the model to stay close to the Teacher's knowledge distribution:

$$
\mathcal{L}_{student} = \alpha \mathcal{L}_{CE} + (1-\alpha) \mathcal{L}_{vis} + \beta \mathcal{L}_{txt}
$$

Where:

- $\mathcal{L}_{CE}$: **Cross-Entropy Loss** on the few-shot Base classes (Learn the task).
- $\mathcal{L}_{vis}$: **Visual Distillation Loss** (KL Divergence). Forces the student's logits to mimic the teacher's distribution, preserving visual generalization (teacher uses the learned text context and the frozen visual features).
- $\mathcal{L}_{txt}$: **Text Feature Regularization**. A cosine similarity constraint ensuring learned prompts do not diverge too far from hand-crafted prompts.

---

## 🛠️ Implementation Details

### Dataset

- **Dataset:** Oxford Flowers 102
- **Split:** 50% Base Classes (Seen during training), 50% Novel Classes (Unseen).
- **Shots:** 10 samples per class ($k=10$).

### Hyperparameters

An exhaustive grid search was performed to find the optimal balance between task learning ($\alpha$) and regularization ($\beta$) using a **3-fold cross-validation strategy**. For each configuration, the Base classes were randomly shuffled and split into _meta-training_ (N=25) and _meta-validation_(N=26) sets.

<div align="center">
  <img src="https://i.postimg.cc/T2Vt7MzM/hyperparam-study-analysis-2.png" alt="Hyperparameter Heatmap" width="900"/>
  <br />
  <em>Fig 3: Hyperparameter search for Alpha and Beta. The sweet spot was found at α=0.8, β=4.</em>
</div>

- **Backbone:** ViT-B/16
- **Epochs:** 15
- **Learning Rate:** 0.002 (Cosine Annealing)
- **Alpha ($\alpha$):** 0.8 (Balances CE and Visual Distillation)
- **Beta ($\beta$):** 4 (Text regularization strength)

---
