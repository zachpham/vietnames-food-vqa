import torch
from PIL import Image
import sys 
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.B1_model  import PaliGemmaVQA
from Data_preprocessing import val_transform, encode_question, MAX_ANSWER_LEN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_predict_B1(image_path, question):
    """Dự đoán 1 sample sử dụng cấu hình B1 zero-shot[cite: 5]"""
   
    model_b1 = PaliGemmaVQA()
    
    image = Image.open(image_path).convert("RGB")
    answer = model_b1.predict(image, question)
    
    return answer
