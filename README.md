# 🚀 CLIP-AUTT: Test-Time Personalization with Action Units for Fine-Grained Video Facial Expression Recognition

This repository contains the official implementation of CLIP-AU and CLIP-AUTT, our AU-guided training and test-time personalization frameworks for fine-grained and subject-specific facial expression recognition (FER).    


## 📘 Overview
CLIP-AU integrates Action Unit (AU) semantics into the CLIP pipeline by encoding 46 AU textual descriptions and learning a lightweight AU text adapter that aligns AU embeddings with facial video representations. A temporal module then captures the dynamic evolution of AU activations for robust expression recognition.

CLIP-AUTT extends this pipeline with test-time AU prompt tuning, an unsupervised personalization strategy that adapts AU embeddings per subject via entropy minimization, enabling subject-specific AU–video alignment without AU labels.

This repository includes:

* CLIP-AU — AU-guided temporal CLIP for fine-grained FER
* CLIP-AUTT — test-time AU prompt tuning for personalized FER
* Subject-wise evaluation protocol for BioVid, StressID, and BAH datasets

## Prerequisites

### Environment 
The code is tested on:
* Python 3.12.2
* PyTorch 2.2.0
* CUDA 12.8

```
pip install -r requirements.txt

or

conda env create --file environment.yml
```

### 📁 Datasets 

We evaluate on three FER datasets with subtle and fine-grained expressions:

* [BioVid Heat Pain Database](https://www.nit.ovgu.de/nit/en/BioVid-p-1358.html) 
* [StressID](https://project.inria.fr/stressid/)
* [BAH Dataset](https://github.com/sbelharbi/bah-dataset)

Subject Lists for Personalization

We follow the same 10-target-subject protocol as prior works.
Full lists are provided in Supplementary Material.


## Run CLIP-AU
```
bash scripts/train_clip_au.sh $1 $2
```

This trains:
* Visual encoder (frozen CLIP)
* AU text adapter
* Temporal module (1D-CNN + GLU)
* AU-to-expression classifier


## Evaluate CLIP-AU
```
bash scripts/eval_clip_au.sh $1 $2
```

## Run CLIP-AUTT (Test-Time Personalization)
CLIP-AUTT adapts AU embeddings per subject using entropy minimization.
```
bash scripts/run_clip_autt.sh $1 $2
```
CLIP-AUTT modifies only AU embeddings — no model weights are updated.



### Main Results

#### Fine-Grained FER Performance

CLIP-AU outperforms STA CLIP-based video FER methods: Emo-CLIP, X-CLIP, and Exp-CLIP across BioVid, StressID, and BAH.

#### Personalized FER (Subject-Wise)

CLIP-AUTT consistently improves recognition accuracy for each subject, balancing prediction confidence and adapting AU relevance via entropy-driven AU tuning. 
CLIP-AUTT outperforms STA test-time classificaitona and action recognition video methods: TPT, TDA, DPE, PromptAlign, ReTA, and T3AL across 10 target subjects on three datatsets.

