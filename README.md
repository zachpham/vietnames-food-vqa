# Vietnamese Specialized Visual Question Answering (VQA)

## 📌 Project Overview
This project focuses on building a **Vietnamese Visual Question Answering (VQA)** system for a specialized domain. The system is designed to take an image and a Vietnamese-language question as input to generate a natural language answer. The implementation combines various deep learning architectures, including **CNN, LSTM, Transformer**, and **Multimodal learning**.

---

## 📊 Dataset Specifications
The dataset is either collected or self-built based on the following criteria:
* **Specialized Domain**: The project focuses on a niche area such as Vietnamese cuisine, landmarks, traffic signs, or traditional costumes.
* **Scale**: The training set contains $\ge 2,000$ triplets (image, question, answer), with a minimum of 200 unique images and at least 3 questions per image.
* **Standardized Testing**: A manual test set of $\ge 50$ samples is prepared, ensuring no image overlap with the training set.
* **Question Diversity**: Questions cover multiple types, including yes/no, counting, identification, attributes, and spatial reasoning.
* **Answer Constraint**: Generated answers are restricted to a maximum of 10 words.
* **Data Split**: The dataset is divided into **Train/Val/Test** sets with an **80/10/10** ratio.
* **Data Augmentation**: Techniques used include image flipping, rotation, and cropping, as well as text paraphrasing and back-translation.

---

## 🏗️ Model Architectures
The project explores two mandatory implementation directions:

### Direction A: Modular Architecture (Kiến trúc rời)
* **Image Encoder**: Utilizes a pre-trained CNN (ResNet, VGG, or EfficientNet) or a Vision Transformer (ViT).
* **Text Encoder**: Utilizes LSTM/BiLSTM or PhoBERT.
* **Fusion Mechanism**: Employs concatenation, element-wise operations, or co-attention mechanisms.
* **Answer Decoder**: Conducts a mandatory comparison between **LSTM** and **Transformer** decoders while keeping encoders constant.

### Direction B: Multimodal Pre-trained Models
* **Models**: Fine-tuning of state-of-the-art models such as **BLIP/BLIP-2, ViLT, LLaVA, Qwen-VL, or PaliGemma**.
* **Strategies**: Application of **LoRA/PEFT** where necessary, with a clear strategy for handling Vietnamese (direct use or translation).

---

## 🚀 Advanced Optimization
Quality is further enhanced through **Reinforcement Learning (RL)**:
* **Training**: Additional training using **PPO, DPO, or RLHF**.
* **Reward Signal**: Metrics like VQA Accuracy or BERTScore are used as rewards.
* **Preference Data**: Includes $\ge 100$ preference pairs for training.
* **Analysis**: Evaluation of RL versus Supervised Fine-Tuning (SFT) using both automatic metrics and human evaluation.

---

## 🧪 Experiments and Evaluation
Four mandatory configurations are analyzed:

| Config ID | Description |
| :--- | :--- |
| **A1** | Modular Architecture with **LSTM decoder** |
| **A2** | Modular Architecture with **Transformer decoder** |
| **B1** | Multimodal Pre-trained (**Zero-shot**) |
| **B2** | Multimodal Pre-trained (**Fine-tuned**) |

**Evaluation Metrics**:
* **VQA Accuracy**: Exact match and soft accuracy based on VQA v2 standards.
* **Language Metrics**: BLEU, ROUGE-L, and METEOR for descriptive answers.
* **Semantic Scoring**: **BERTScore** for semantic meaning.
* **Judgment**: **LLM-as-a-judge** for qualitative assessment.

---

## 📂 Deliverables
The following artifacts are provided for this project:
* **GitHub Repository**: Contains full source code and a detailed README.
* **Technical Report**: A 15–20 page document including an evaluation chapter.
* **Presentation**: Slides and a 3–5 minute demo video.
* **Resources**: Dataset and model checkpoints hosted on HuggingFace Hub and Google Drive.

---
*This project is part of the Deep Learning Course Final Project.*
