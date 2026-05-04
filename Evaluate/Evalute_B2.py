import os
import json
import torch
import numpy as np
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


# CẤU HÌNH THÔNG SỐ CỦA B2

BASE_MODEL_ID  = "google/paligemma-3b-pt-224"
LORA_ADAPTER_DIR = r"D:\Code\Deep Learning\Endterm\B2_finetuned_model"
JSON_TEST_PATH = r"D:\Code\Deep Learning\Endterm\data\test.json"
IMAGE_BASE_DIR = r"D:\Code\Deep Learning\Endterm\data_resized_224"


# HÀM TÍNH METRICS 

def compute_vqa_accuracy(preds, gts):
    correct = sum(p == g for p, g in zip(preds, gts))
    return correct / len(gts)

def compute_bleu(preds, gts):
    smoothie = SmoothingFunction().method1
    scores = [sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie) for pred, gt in zip(preds, gts)]
    return np.mean(scores)

def compute_rouge(preds, gts):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(gt, pred)["rougeL"].fmeasure for pred, gt in zip(preds, gts)]
    return np.mean(scores)

def compute_bertscore(preds, gts):
    P, R, F1 = bert_score(preds, gts, lang="vi", verbose=False)
    return F1.mean().item()


# CLASS WRAPPER CHO B2 

class PaliGemmaVQAB2:
    def __init__(self):
        print("Đang load Processor...")
        self.processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL_ID)
        
        print("Đang load Base Model (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True
        )
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="cuda"
        )
        
        print("Đang load và gộp LoRA Adapters của B2...")
      
        self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
        self.model.eval() 

    def predict(self, image, question):
        # Format prompt CHUẨN như lúc fen train B2
        image_token = self.processor.image_token
        prompt = f"{image_token}answer vi {question}"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10, # Sinh tối đa 10 từ
                do_sample=False    
            )
            
        generated_tokens = outputs[0][input_len:]
        answer = self.processor.decode(generated_tokens, skip_special_tokens=True)
        return answer


# HÀM MAIN THỰC THI

def main():
    # 1. Khởi tạo Model B2
    vqa_model = PaliGemmaVQAB2()

    # 2. Load dữ liệu test
    if not os.path.exists(JSON_TEST_PATH):
        print(f"Không tìm thấy file JSON tại: {JSON_TEST_PATH}")
        return

    with open(JSON_TEST_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    test_samples = []
    for item in raw_data:
        img_rel_path = item["image_path"]
        full_img_path = os.path.join(IMAGE_BASE_DIR, img_rel_path)
        
        for q_obj in item["questions"]:
            test_samples.append({
                "path": full_img_path,
                "question": q_obj["question"],
                "answer": q_obj["answer"].lower().strip()
            })

    print(f"Tổng cộng có {len(test_samples)} câu hỏi cần test.")

    predictions = []
    ground_truths = []
    results_to_save = []

    # 3. Vòng lặp Inference
    for i, item in enumerate(test_samples):
        try:
            image = Image.open(item["path"]).convert("RGB")
            
            # Gọi hàm predict từ class B2
            pred = vqa_model.predict(image, item["question"])
            pred_clean = pred.lower().strip()
            
            predictions.append(pred_clean)
            ground_truths.append(item["answer"])
            
            results_to_save.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": pred_clean
            })
            
            if (i + 1) % 100 == 0:
                print(f"Đã xử lý {i + 1}/{len(test_samples)} câu hỏi")
        
        except Exception as e:
            print(f"Lỗi tại ảnh {item['path']}: {e}")

    # 4. Tính toán kết quả cuối
    print(" KẾT QUẢ B2 — SUPERVISED FINE-TUNING (SFT)")

    vqa_acc = compute_vqa_accuracy(predictions, ground_truths)
    bleu    = compute_bleu(predictions, ground_truths)
    rouge_l = compute_rouge(predictions, ground_truths)
    bert_f1 = compute_bertscore(predictions, ground_truths)

    print(f"  VQA Accuracy : {vqa_acc*100:.2f}%")
    print(f"  BLEU-4       : {bleu:.4f}")
    print(f"  ROUGE-L      : {rouge_l:.4f}")
    print(f"  BERTScore F1 : {bert_f1:.4f}")
  

if __name__ == "__main__":
    main()