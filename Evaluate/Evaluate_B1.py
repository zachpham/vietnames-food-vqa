import os
import json
import numpy as np
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Import class model vừa tạo từ file kia
from Model.B1_model import PaliGemmaVQA

JSON_TEST_PATH = r"D:\Code\Deep Learning\Endterm\data\test.json"
IMAGE_BASE_DIR = r"D:\Code\Deep Learning\Endterm\data"


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

def main():
    # 1. Khởi tạo Model
    vqa_model = PaliGemmaVQA()

    # 2. Load dữ liệu test
    if not os.path.exists(JSON_TEST_PATH):
        print(f" Không tìm thấy file JSON tại: {JSON_TEST_PATH}")
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
            
            # Gọi hàm predict từ class
            pred = vqa_model.predict(image, item["question"])
            pred_clean = pred.lower().strip()
            
            predictions.append(pred_clean)
            ground_truths.append(item["answer"])
            
            results_to_save.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": pred
            })
            
            if (i + 1) % 100 == 0:
                print(f"Đã xử lý {i + 1}/{len(test_samples)} câu hỏi")
        
        except Exception as e:
            print(f"Lỗi tại ảnh {item['path']}: {e}")

    # 4. Tính toán kết quả cuối

    print(" KẾT QUẢ B1 — ZERO-SHOT (PaliGemma)")
  
    
    vqa_acc = compute_vqa_accuracy(predictions, ground_truths)
    bleu    = compute_bleu(predictions, ground_truths)
    rouge_l = compute_rouge(predictions, ground_truths)
    bert_f1 = compute_bertscore(predictions, ground_truths)

    print(f"  VQA Accuracy : {vqa_acc*100:.2f}%")
    print(f"  BLEU-4       : {bleu:.4f}")
    print(f"  ROUGE-L      : {rouge_l:.4f}")
    print(f"  BERTScore F1 : {bert_f1:.4f}")

    # LƯU Ý: Phần code lưu JSON tôi đã giữ lại, fen có thể uncomment ra nếu muốn dùng.

if __name__ == "__main__":
    main()