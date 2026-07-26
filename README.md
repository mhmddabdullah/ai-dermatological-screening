AI-Powered Dermatological Disease Detection and Screening from Clinical Images and Clinical Data

Graduation project — a dual-branch deep learning framework for dermatological disease screening, combining an image classification branch with an NLP symptom-classification branch, evaluated under a rigorous, reproducible protocol.

Overview:
Skin diseases affect over a billion people globally, and delayed or inaccurate diagnosis carries serious consequences — from missed early-stage melanoma to progressive inflammatory disease. This project builds and evaluates a dual-branch AI framework — image classification and NLP symptom classification — designed to support faster, more accessible dermatological screening.

Image Branch — ConvNeXt-Tiny on a Five-Class Clinical Dataset

Evaluated against ResNet50 and EfficientNet-B4 baselines, with a full ablation study (label smoothing, cosine annealing LR, SE blocks, full fine-tuning) on a 319-image dataset spanning Burns, Cancer, Lupus, Normal, and Urticaria.

Model	Test Accuracy	Macro F1	AUC-ROC
ResNet50 (baseline)	83.33%	82.45%	98.12%
EfficientNet-B4 (224px)	—	76.2–76.8%	—
ConvNeXt-Tiny (+ label smoothing + cosine annealing)	91.67%	90.66%	98.99%
NLP Branch — BioBERT on Patient-Reported Symptoms

Seven-class symptom classification (Psoriasis, Chicken Pox, Impetigo, Fungal Infection, Acne, Drug Reaction, Allergy) from the Symptom2Disease dataset, comparing a fully frozen backbone against partial fine-tuning (top 4 transformer layers unfrozen).

Configuration	Test Accuracy	Macro F1	AUC-ROC
Frozen backbone	77.36%	75.67%	94.22%
Partial fine-tuning (top 4 layers)	96.23%	96.01%	99.65%

Partial fine-tuning produced a large jump in performance, most notably on the Drug Reaction class, which recovered from 0.364 to 0.923 F1 — frozen representations couldn't distinguish drug reaction from allergy symptom language; fine-tuning the upper layers resolved it almost completely.

Explainability:
Grad-CAM class activation mapping confirms the image model attends to clinically relevant regions: the erythematous patch for lupus, the pigmented lesion core for cancer, and diffuse (non-focal) activation for normal skin — consistent with the absence of a localized pathological feature.

Why the Branches Aren't Fused (Yet):
The image branch's taxonomy (Burns, Cancer, Lupus, Normal, Urticaria) and the NLP branch's taxonomy (Psoriasis, Chicken Pox, Impetigo, Fungal Infection, Acne, Drug Reaction, Allergy) don't overlap. Meaningful fusion requires either a dataset with paired image+text data across a shared disease taxonomy, or a learned mapping between the two taxonomies — neither of which currently exists publicly. This is documented as the primary direction for future work.

Tech Stack:
Python, PyTorch, Hugging Face Transformers (BioBERT), torchvision (ConvNeXt-Tiny, ResNet50, EfficientNet-B4), scikit-learn, Grad-CAM, JupyterLab
