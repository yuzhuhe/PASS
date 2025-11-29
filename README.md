# PASS: Personalized, Anomaly-aware Sampling and reconStruction for Fast MRI

## 🔍 Description

**PASS (Personalized, Anomaly-aware Sampling and reconStruction)** is an intelligent and **personalized** framework for **accelerated Magnetic Resonance Imaging (MRI)**. It addresses the critical limitation of traditional fast MRI methods—their lack of adaptability to patient-specific pathologies—by leveraging a **Vision-Language Model (VLM)** to guide the entire imaging pipeline.

PASS establishes a **closed-loop system** for fast MRI that achieves:

* **Personalized Acquisition:** An **Anomaly-aware Sampling Module** dynamically generates patient-specific k-space trajectories
* **Task-Oriented Reconstruction:** A physics-informed **Deep Unrolling Network** is guided by an anomaly-aware prior extracted from a VLM (fine-tuned **NetAD**) to selectively enhance pathological features.
* **Superior Clinical Utility:** It achieves superior image quality, especially within lesion regions, and directly translates to improvements in downstream diagnostic tasks, such as **fine-grained anomaly detection** and **diagnosis**.

Evaluated on the **fastMRI benchmark** (Brain T1w, FLAIR, and Knee PD) augmented with **fastMRI+** anomaly labels, PASS demonstrates robust performance across diverse anatomies, contrasts, and acceleration factors ($4\times$ and $8\times$).

---

## 🧭 Framework Overview

Overview of the **PASS** closed-loop pipeline for personalized, fast MRI. The Vision-Language Model (VLM) guides both the anomaly-aware sampling and the deep unrolling reconstruction to prioritize clinically relevant features. 


![framework of the PASS](assets/framework.pdf)

---
## 📦 Package Dependencies

The dependencies for the PASS implementation are listed below. Please ensure you install all required packages.

| **Package** | **Version** | **Package** | **Version** |
| :--- | :--- | :--- | :--- |
| Python | *(Specify version)* | PyTorch | *(Specify version)* |
| CUDA | 11.8 (or compatible) | Adam Optimizer | [cite_start]N/A (Used for optimization) [cite: 740] |
| fastMRI | [cite_start]N/A (Dataset source) [cite: 865] | fastMRI+ | [cite_start]N/A (Annotation source) [cite: 866] |
| CLIP | [cite_start]N/A (VLM backbone) [cite: 190, 598] | ESPIRIT | [cite_start]N/A (Coil sensitivity estimation) [cite: 841] |
| NumPy | *(Specify version)* | SciPy | *(Specify version)* |

---

## 🗂 Step 1. Data Preparation

### 🔹 fastMRI and fastMRI+ Datasets

The PASS framework is trained and evaluated using publicly available datasets:

* [cite_start]**Multi-coil Brain and Knee MRI Data:** Sourced from the **fastMRI dataset** ([https://fastmri.org/](https://fastmri.org/))[cite: 840, 865].
* [cite_start]**Pathological Annotations:** Sourced from the **fastMRI+ dataset** ([https://github.com/microsoft/fastmri-plus](https://github.com/microsoft/fastmri-plus)), which provides lesion types and bounding box coordinates[cite: 850, 866].

### 🔹 Preprocessing Pipeline

[cite_start]The following steps were used for data preparation [cite: 841-846]:

1.  [cite_start]**Coil Sensitivity Maps:** Estimated using the **ESPIRIT algorithm**[cite: 841].
2.  **Brain Scan Standardization:** T1-weighted (T1w) and FLAIR contrasts. [cite_start]Raw k-space data were transformed to the image domain, uniformly **cropped to $320 \times 320$ pixels**, and corresponding k-space and coil sensitivity maps were simulated from these cropped images [cite: 842-844].
3.  [cite_start]**Knee Scan Standardization:** Proton-density (PD) images were used, consistent with the original $320 \times 320$ acquisition protocol[cite: 842, 846].
4.  [cite_start]**Anomaly Labels:** For brain scans, image-level labels were derived from study-level fastMRI+ annotations to indicate the presence of lesions[cite: 851, 852].

---

## 🧠 Step 2. VLM Fine-tuning (NetAD)

[cite_start]The core semantic guidance in PASS comes from a fine-tuned VLM, **NetAD**, adapted from a CLIP-based architecture for anomaly detection[cite: 159, 190, 593, 598].

* [cite_start]**Backbone:** CLIP encoders are **frozen** to retain broad semantic knowledge[cite: 599].
* [cite_start]**Adapters:** Lightweight adapters are inserted after the image encoder's feature layers for medical domain adaptation[cite: 600].
    * [cite_start]**Pixel-Level Adapter (PLA):** Learns **fine-grained localization** (trained with bounding-box supervision)[cite: 604, 609, 610].
    * [cite_start]**Image-Level Adapter (ILA):** Captures **global pathology-relevant cues** (trained with image-level labels)[cite: 609, 610].
* [cite_start]**Input Data:** The VLM is fine-tuned on each dataset using **12 annotated cases** containing both pixel-level bounding boxes and image-level lesion labels[cite: 746].
* [cite_start]**Output:** The fine-tuned VLM (NetAD) generates an **Anomaly Attention Map** from intermediate image estimates, which serves as the **anomaly-aware prior** for sampling and reconstruction[cite: 593, 610, 747].

---

## 🚀 Step 3. Training PASS

The full PASS framework is trained end-to-end to jointly optimize the VLM-guided reconstruction and the adaptive sampling mask.

### 🔹 Training the Reconstruction Network

[cite_start]The reconstruction network $R_{\Theta}$ is a deep unrolling architecture that minimizes a loss balancing data fidelity, a global image prior, and the VLM-guided anomaly-aware regularizer[cite: 612, 614, 617, 682]:

$$
[cite_start]\mathcal{L}_{Rec}=\gamma_{1}||X_{global}^{(K)}-X^{(gt)}||_{2}^{2}+\gamma_{2}||X^{(K)}-X^{(gt)}||_{2}^{2}+\gamma_{3}||map*(X^{(K)}-X^{(gt)})||_{2}^{2} \quad \text{[cite: 690]}
$$

* [cite_start]**Total Stages:** The network is unrolled for **three iterations**, with parameters shared across stages[cite: 741].
* [cite_start]**Components:** Each stage includes a **Global Image Denoising Module** and a **Personalized Anomaly-Aware Module (PA)**, the latter integrating the VLM-derived attention map ($map=Net_{AD}(X_{global}^{(K)})$)[cite: 189, 642, 653, 691].

### 🔹 Training the Adaptive Sampling Module

[cite_start]The learnable sampling mask $M_{\Phi}$ is optimized in a two-stage process [cite: 694-695]:

1.  [cite_start]**Population-Level Prior:** A baseline probabilistic sampling mask $M_{\Phi}$ and the reconstructor $R_{\Theta}$ are learned using the LOUPE paradigm [cite: 716-717].
2.  [cite_start]**Personalized Sampling:** With $R_{\Theta}$ fixed, an **Anomaly-aware Sampling network (AS)** generates a high-frequency sampling component $M_{\Phi}^{2}$ based on a low-resolution prior (Auto-Calibrating Signal, ACS) and VLM-guided feedback [cite: 718-720, 722]. [cite_start]The loss function explicitly includes an anomaly-specific k-space consistency term to ensure high-frequency pathological information is preserved[cite: 724]:

$$
[cite_start]\mathcal{L}_{mask}=||X^{(K)}-X^{(gt)}||_{2}^{2}+\lambda||map\odot(\mathcal{F}^{H}(M_{\Phi}^{2}\odot\mathcal{F}(X^{(K)}))-\mathcal{F}^{H}(M_{\Phi^{\prime}}^{2}\odot\mathcal{F}(X^{(gt)})))||_{2}^{2} \quad \text{[cite: 725]}
$$

---

## 💾 Code and Data Availability

The custom-processed data and the full implementation of the PASS framework are publicly available.

* [cite_start]**Codebase:** [https://github.com/ladderlab-xjtu/PASS](https://github.com/ladderlab-xjtu/PASS) [cite: 875]
* [cite_start]**Data Archive:** [https://zenodo.org/records/PASS](https://zenodo.org/records/PASS) [cite: 869]

Would you like me to elaborate on the quantitative results (PSNR/SSIM/AUC) for the downstream diagnostic tasks?





