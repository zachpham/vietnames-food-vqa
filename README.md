Vietnamese Specialized Visual Question Answering (VQA)
📌 Project Overview

This project focuses on developing a Vietnamese Visual Question Answering (VQA) system for a specialized domain (e.g., Vietnamese cuisine, landmarks, or traffic signs). The system accepts an image and a Vietnamese-language question as input to generate an appropriate natural language answer. It integrates core Deep Learning architectures, including CNNs, LSTMs, and Transformers, alongside modern multimodal learning techniques.  

--------------------

📊 Dataset Specifications

The dataset was constructed and processed with the following requirements:
- Domain: Focused on a specific, specialized field.  
- Training Data: $\ge 2,000$ triplets (image, question, answer), with at least 200 unique images and $\ge 3$ questions per image.  
- Test Data: $\ge 50$ manually prepared sets with images strictly separate from the training set.  
- Question Diversity: Includes Yes/No, counting, identification, attribute, and spatial reasoning questions.  
- Constraints: Answers are limited to $\le 10$ words.  
- Split: Data is divided into Train/Val/Test sets at an 80/10/10 ratio.  
- Augmentation: Applied image transformations (flip, rotate, crop) and text enhancements (paraphrasing, back-translation).  

--------------------

Here is a professional README.md for Task 1, written in English and structured according to your project requirements.  Vietnamese Specialized Visual Question Answering (VQA)📌 Project OverviewThis project focuses on developing a Vietnamese Visual Question Answering (VQA) system for a specialized domain (e.g., Vietnamese cuisine, landmarks, or traffic signs). The system accepts an image and a Vietnamese-language question as input to generate an appropriate natural language answer. It integrates core Deep Learning architectures including CNNs, LSTMs, and Transformers, alongside modern multimodal learning techniques.  📊 Dataset SpecificationsThe dataset was constructed and processed with the following requirements:Domain: Focused on a specific specialized field.  Training Data: $\ge 2,000$ triplets (image, question, answer), with at least 200 unique images and $\ge 3$ questions per image.  Test Data: $\ge 50$ manually prepared sets with images strictly separate from the training set.  Question Diversity: Includes Yes/No, counting, identification, attribute, and spatial reasoning questions.  Constraints: Answers are limited to $\le 10$ words.  Split: Data is divided into Train/Val/Test sets at an 80/10/10 ratio.  Augmentation: Applied image transformations (flip, rotate, crop) and text enhancements (paraphrasing, back-translation).  🏗️ Model ArchitecturesThe project explores two distinct implementation strategies:  Approach A: Modular Architecture   Image Encoder: Pre-trained CNNs (ResNet, VGG, or EfficientNet) or Vision Transformers (ViT).  Text Encoder: LSTM, BiLSTM, or PhoBERT.  Fusion Layer: Utilizes concatenation, element-wise multiplication, or co-attention mechanisms.  Answer Decoder: A comparative study between LSTM decoders and Transformer decoders.  Approach B: Multimodal Pre-trained Models   Fine-tuning state-of-the-art models such as BLIP/BLIP-2, ViLT, LLaVA, Qwen-VL, or PaliGemma.  Utilizes LoRA or PEFT strategies for efficient adaptation.  Clearly defines strategies for handling Vietnamese input (direct use or translation).  
