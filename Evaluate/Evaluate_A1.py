import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import google.generativeai as genai
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


GEMINI_API_KEY = "AIzaSyC8uhNQmbVzh0X_WeRKD8oYTLUm4CDZZ18"  
genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

from Model.A1_model import VQAModel as ModelA1
from Model.A2_model import VQAModel_A2 as ModelA2
from Data_preprocessing import load_data, build_vocab, val_transform, encode_question

DATA_DIR       = r"D:\Code\Deep Learning\Endterm\data"
TRAIN_JSON     = r"D:\Code\Deep Learning\Endterm\data\train.json"
TEST_JSON      = r"D:\Code\Deep Learning\Endterm\data\test.json"
IMAGE_DIR      = r"D:\Code\Deep Learning\Endterm\data"
CKPT_A1        = r"D:\Code\Deep Learning\Endterm\data\checkpoints\better_A1.pth"
CKPT_A2        = r"D:\Code\Deep Learning\Endterm\data\checkpoints\better_A2.pth"
MAX_ANSWER_LEN = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy đánh giá trên: {DEVICE}")


# DECODE & GENERATE


def decode_tokens(token_ids, idx2word):
    words = []
    for idx in token_ids:
        word = idx2word.get(idx, "<unk>")
        if word in ("<sos>", "<pad>"):
            continue
        if word == "<eos>":
            break
        words.append(word)
    return " ".join(words).lower().strip()


def generate_a1(model, image, q_ids, q_mask, vocab, idx2word, max_len=MAX_ANSWER_LEN):
    """Autoregressive decode cho LSTM decoder (A1)."""
    model.eval()
    with torch.no_grad():
        V = model.image_encoder(image)
        Q = model.question_encoder(q_ids, q_mask)
        v_hat, q_hat = model.co_attention(V, Q, q_mask)

        combined = torch.cat([v_hat, q_hat], dim=1)
        hidden   = torch.tanh(model.init_h(combined)).unsqueeze(0)
        cell     = torch.tanh(model.init_c(combined)).unsqueeze(0)

        ans_ids = [vocab["<sos>"]]

        for _ in range(max_len):
            current_word = torch.tensor([[ans_ids[-1]]], dtype=torch.long, device=DEVICE)
            output, hidden, cell = model.decoder(current_word, hidden, cell)

            # FIX: tường minh lấy batch=0, step=0, toàn bộ vocab
            pred_id = output[0, 0, :].argmax().item()
            ans_ids.append(pred_id)

            if pred_id == vocab["<eos>"]:
                break

    return decode_tokens(ans_ids, idx2word)


def generate_a2(model, image, q_ids, q_mask, vocab, idx2word, max_len=MAX_ANSWER_LEN):
    """Autoregressive decode cho Transformer decoder (A2)."""
    model.eval()
    with torch.no_grad():
        V = model.image_encoder(image)
        Q = model.question_encoder(q_ids, q_mask)
        v_hat, q_hat = model.co_attention(V, Q, q_mask)

        combined = torch.cat([v_hat, q_hat], dim=1)
        fused    = model.fusion(combined).unsqueeze(1)
      
        memory   = torch.cat([fused, V], dim=1)      

        ans_ids = [vocab["<sos>"]]

        for _ in range(max_len):
            current_seq = torch.tensor([ans_ids], dtype=torch.long, device=DEVICE)
            output      = model.decoder(current_seq, memory)
            pred_id     = output[:, -1, :].argmax(dim=-1).item()
            ans_ids.append(pred_id)

            if pred_id == vocab["<eos>"]:
                break

    return decode_tokens(ans_ids, idx2word)



# METRICS & LLM JUDGE


def compute_metrics(predictions, ground_truths):

    # VQA Accuracy
    vqa_acc = sum(p == g for p, g in zip(predictions, ground_truths)) / len(ground_truths)

    # BLEU 
    smoothie = SmoothingFunction().method1
    bleu = np.mean([
        sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothie)
        for pred, gt in zip(predictions, ground_truths)
    ])

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge = np.mean([
        scorer.score(gt, pred)["rougeL"].fmeasure
        for pred, gt in zip(predictions, ground_truths)
    ])

    # BERTScore F1
    _, _, F1 = bert_score(
        predictions, ground_truths,
        model_type="bert-base-multilingual-cased",
        verbose=False
    )

    return {
        "acc":   vqa_acc,
        "bleu":  float(bleu),
        "rouge": float(rouge),
        "bert":  F1.mean().item()
    }


def compute_llm_judge_score(questions, gts, preds, num_samples=50):
    """Dùng Gemini chấm điểm ngữ nghĩa ngẫu nhiên num_samples câu."""
    if not GEMINI_API_KEY:
        print("  Bỏ qua LLM Judge: chưa cấu hình GEMINI_API_KEY")
        return 0.0

    total_samples = len(gts)
    sample_size   = min(num_samples, total_samples)
    indices       = np.random.choice(total_samples, sample_size, replace=False)

    scores = []
    for idx in tqdm(indices, desc="LLM Judging"):
        prompt = f"""Bạn là giám khảo chấm điểm hệ thống Hỏi đáp trên ảnh (VQA).
            Câu hỏi: {questions[idx]}
            Đáp án chuẩn: {gts[idx]}
            Câu trả lời của AI: {preds[idx]}
            Hãy chấm điểm độ chính xác về mặt ngữ nghĩa trên thang từ 1 đến 5 (1: Sai hoàn toàn, 5: Đúng hoàn toàn).
            CHỈ trả về 1 con số duy nhất, không giải thích."""
        try:
            response = judge_model.generate_content(prompt)
            score    = int(response.text.strip())
            if 1 <= score <= 5:
                scores.append(score)
            time.sleep(6)
        except Exception as e:
            print(f"  LLM Judge lỗi tại idx={idx}: {e}")

    return float(np.mean(scores)) if scores else 0.0



# 4. LOAD MODEL


def load_weights_safe(model, filepath, device):

    checkpoint = torch.load(filepath, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Đã load checkpoint từ {filepath}")
        return checkpoint.get("a_vocab", None)
    else:
        model.load_state_dict(checkpoint)
        print(f"  Đã load weights từ {filepath}")
        return None



# MAIN


def main():
    # Build vocab
    print("Đang build vocab từ tập train...")
    train_data = load_data(TRAIN_JSON)
    vocab      = build_vocab(train_data, "answer")
    idx2word   = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    print(f"  Kích thước vocab: {vocab_size}")

    #  Load models 
    print("Đang load models...")
  
    model_a1 = ModelA1(answer_vocab_size=vocab_size).to(DEVICE)
    model_a2 = ModelA2(answer_vocab_size=vocab_size).to(DEVICE)

    load_weights_safe(model_a1, CKPT_A1, DEVICE)
    load_weights_safe(model_a2, CKPT_A2, DEVICE)

        
    # Load test data 
    test_data = load_data(TEST_JSON)
    print(f"Tập test: {len(test_data)} câu hỏi")

    questions = []
    gts       = []
    preds_a1  = []
    preds_a2  = []

    #  Evaluate 
    print("\nBắt đầu đánh giá trên tập test...")
    for item in tqdm(test_data):
        # Load ảnh
        img_path     = os.path.join(IMAGE_DIR, item["image_path"].replace("\\", os.sep))
        image        = Image.open(img_path).convert("RGB")
        image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

        # Encode câu hỏi
        q_ids, q_mask = encode_question(item["question"])
        q_ids  = q_ids.unsqueeze(0).to(DEVICE)
        q_mask = q_mask.unsqueeze(0).to(DEVICE)

        # Ground truth
        gt = item["answer"].lower().strip()
        gts.append(gt)
        questions.append(item["question"])

        # Sinh câu trả lời
        preds_a1.append(generate_a1(model_a1, image_tensor, q_ids, q_mask, vocab, idx2word))
        preds_a2.append(generate_a2(model_a2, image_tensor, q_ids, q_mask, vocab, idx2word))

    # Traditional metrics
    print("\nĐang tính Traditional Metrics...")
    metrics_a1 = compute_metrics(preds_a1, gts)
    metrics_a2 = compute_metrics(preds_a2, gts)

    # LLM Judge 
    print("\nĐang chấm điểm bằng LLM (Gemini)...")
    print("  Đánh giá A1:")
    metrics_a1["llm_score"] = compute_llm_judge_score(questions, gts, preds_a1)
    print("  Đánh giá A2:")
    metrics_a2["llm_score"] = compute_llm_judge_score(questions, gts, preds_a2)

    # ----- In kết quả -----
   
    print(f"{'METRIC':<18} | {'A1 (LSTM)':<15} | {'A2 (TRANSFORMER)'}")
    print(f"{'VQA Accuracy':<18} | {metrics_a1['acc']*100:>13.2f}%  | {metrics_a2['acc']*100:>13.2f}%")
    print(f"{'BLEU':<18} | {metrics_a1['bleu']:>15.4f} | {metrics_a2['bleu']:>15.4f}")
    print(f"{'ROUGE-L':<18} | {metrics_a1['rouge']:>15.4f} | {metrics_a2['rouge']:>15.4f}")
    print(f"{'BERTScore F1':<18} | {metrics_a1['bert']:>15.4f} | {metrics_a2['bert']:>15.4f}")
    print(f"{'LLM Judge (/5)':<18} | {metrics_a1['llm_score']:>15.2f} | {metrics_a2['llm_score']:>15.2f}")
    # ----- Lưu kết quả -----
    results_out = {
        "A1": {
            "metrics": metrics_a1,
            "predictions": [
                {"question": q, "ground_truth": g, "prediction": p}
                for q, g, p in zip(questions, gts, preds_a1)
            ]
        },
        "A2": {
            "metrics": metrics_a2,
            "predictions": [
                {"question": q, "ground_truth": g, "prediction": p}
                for q, g, p in zip(questions, gts, preds_a2)
            ]
        }
    }
    out_path = "huong_a_eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print(f"\nDa luu chi tiet vao {out_path}")


if __name__ == "__main__":
    main()