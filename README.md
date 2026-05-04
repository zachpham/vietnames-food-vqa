Vietnamese Specialized Visual Question Answering (VQA)
📌 Project Overview

This project focuses on developing a Vietnamese Visual Question Answering (VQA) system for a specialized domain (e.g., Vietnamese cuisine, landmarks, or traffic signs). The system accepts an image and a Vietnamese-language question as input to generate an appropriate natural language answer. It integrates core Deep Learning architectures including CNNs, LSTMs, and Transformers, alongside modern multimodal learning techniques.  
--------------------
📊 Dataset Specifications
The dataset was constructed and processed with the following requirements:
- Domain: Focused on a specific specialized field.  
- Training Data: $\ge 2,000$ triplets (image, question, answer), with at least 200 unique images and $\ge 3$ questions per image.  
- Test Data: $\ge 50$ manually prepared sets with images strictly separate from the training set.  
- Question Diversity: Includes Yes/No, counting, identification, attribute, and spatial reasoning questions.  
- Constraints: Answers are limited to $\le 10$ words.  
- Split: Data is divided into Train/Val/Test sets at an 80/10/10 ratio.  
- Augmentation: Applied image transformations (flip, rotate, crop) and text enhancements (paraphrasing, back-translation).  
