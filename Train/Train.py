import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from Data_preprocessing import load_datasets

from Model.A1_model import VQAModel as ModelA1
from Model.A2_model import VQAModel_A2 as ModelA2

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# CONFIG
MODEL_TYPE = "A2"   # "A1" hoặc "A2"

BATCH_SIZE = 16
EPOCHS     = 10
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3

TRAIN_JSON = "D:\Code\Deep Learning\Endterm\data\\train.json"
VAL_JSON   = "D:\Code\Deep Learning\Endterm\data\\val.json"
CHECKPOINT = f"D:\Code\Deep Learning\Endterm\data\checkpoints\\better_{MODEL_TYPE}.pth"


# BUILD MODEL
def build_model(a_vocab):
    if MODEL_TYPE == "A1":
        return ModelA1(len(a_vocab))
    else:
        return ModelA2(len(a_vocab))


# DECODE
def decode_ids(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    words = []
    for i in ids:
        i = int(i)  # FIX: an toàn với cả tensor lẫn Python int
        if i == vocab["<eos>"]:
            break
        if i not in (vocab["<pad>"], vocab["<sos>"]):
            words.append(inv_vocab.get(i, ""))
    return " ".join(words)


# GREEDY GENERATE 
def greedy_generate(model, image, question, attention_mask, a_vocab, max_len=12):
    model.eval()
    B = image.size(0)
    sos_id = a_vocab["<sos>"]
    eos_id = a_vocab["<eos>"]

    with torch.no_grad():
        # Encode một lần
        V = model.image_encoder(image)
        Q = model.question_encoder(question, attention_mask)
        v_hat, q_hat = model.co_attention(V, Q, attention_mask)

        if MODEL_TYPE == "A1":
            combined = torch.cat([v_hat, q_hat], dim=1)
            hidden   = torch.tanh(model.init_h(combined)).unsqueeze(0)
            cell     = torch.tanh(model.init_c(combined)).unsqueeze(0)

            token    = torch.full((B, 1), sos_id, dtype=torch.long, device=DEVICE)
            outputs  = []

            for _ in range(max_len):
                out, hidden, cell = model.decoder(token, hidden, cell)
                token = out.argmax(dim=-1)           
                outputs.append(token)

            outputs = torch.cat(outputs, dim=1)    

        else:  # A2 — Transformer
            combined = torch.cat([v_hat, q_hat], dim=1)
            fused    = model.fusion(combined).unsqueeze(1)
            memory   = torch.cat([fused, V], dim=1)   
          
            seq = torch.full((B, 1), sos_id, dtype=torch.long, device=DEVICE)

            for _ in range(max_len):
                out   = model.decoder(seq, memory)     
                token = out[:, -1:, :].argmax(dim=-1) 
                seq   = torch.cat([seq, token], dim=1)

            outputs = seq[:, 1:]                       
    return outputs 


# FORWARD PASS 
def forward_pass(model, batch):
    images, questions, attention_mask, answers, _ = batch
    images         = images.to(DEVICE)
    questions      = questions.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    answers        = answers.to(DEVICE)

    answer_input  = answers[:, :-1]
    answer_target = answers[:, 1:]

    outputs = model(images, questions, attention_mask, answer_input)
    return outputs, answer_target


# EVALUATE
def evaluate(model, loader, criterion, a_vocab):
    model.eval()

    total_loss  = 0
    pred_texts  = []
    gt_texts    = []
    smooth      = SmoothingFunction().method1 
    rouge       = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with torch.no_grad():
        for batch in loader:
            images, questions, attention_mask, answers, _ = batch
            images         = images.to(DEVICE)
            questions      = questions.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            answers        = answers.to(DEVICE)

        
            outputs, answer_target = forward_pass(model, batch)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                answer_target.reshape(-1)
            )
            total_loss += loss.item()

            # --- Metric tính với greedy decode thực sự ---
           
            preds = greedy_generate(model, images, questions, attention_mask, a_vocab)
            gts   = answers[:, 1:].cpu()   # bỏ <sos>

            for p, g in zip(preds.cpu(), gts):
                pred_texts.append(decode_ids(p, a_vocab))
                gt_texts.append(decode_ids(g, a_vocab))

    # VQA ACC
    correct = sum(p.strip() == g.strip() for p, g in zip(pred_texts, gt_texts))
    vqa_acc = correct / len(pred_texts)

    # BLEU
    bleu_scores = [
        sentence_bleu([g.split()], p.split(), smoothing_function=smooth)
        for p, g in zip(pred_texts, gt_texts)
    ]
    bleu = sum(bleu_scores) / len(bleu_scores)

    # ROUGE-L
    rouge_scores = [
        rouge.score(g, p)["rougeL"].fmeasure
        for p, g in zip(pred_texts, gt_texts)
    ]
    rougeL = sum(rouge_scores) / len(rouge_scores)

    return {
        "loss":    total_loss / len(loader),
        "vqa_acc": vqa_acc,
        "bleu":    bleu,
        "rougeL":  rougeL
    }


# PLOT
def plot_metrics(train_losses, val_losses, vqa_accs, bleu_scores, rouge_scores):
    epochs = range(1, len(train_losses) + 1)

    # 1. Biểu đồ Loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png", bbox_inches='tight') 
    plt.show()

    # 2. Biểu đồ Accuracy
    plt.figure()
    plt.plot(epochs, [x * 100 for x in vqa_accs])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("VQA Accuracy")
    plt.savefig("vqa_accuracy.png", bbox_inches='tight')
    plt.show()

    # 3. Biểu đồ BLEU Score
    plt.figure()
    plt.plot(epochs, bleu_scores)
    plt.xlabel("Epoch"); plt.ylabel("BLEU"); plt.title("BLEU Score")
    plt.savefig("bleu_score.png", bbox_inches='tight')
    plt.show()

    # 4. Biểu đồ ROUGE-L Score
    plt.figure()
    plt.plot(epochs, rouge_scores)
    plt.xlabel("Epoch"); plt.ylabel("ROUGE-L"); plt.title("ROUGE-L Score")
    plt.savefig("rouge_score.png", bbox_inches='tight')
    plt.show()

# TRAIN
def train():
    print("=" * 50)
    print(f"TRAIN MODEL {MODEL_TYPE}")
    print("=" * 50)

    # LOAD DATA
 
    train_dataset, val_dataset, a_vocab = load_datasets(TRAIN_JSON, VAL_JSON)
    patience_counter = 0
  
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        num_workers=0, pin_memory=True
    )

    # MODEL
    model = build_model(a_vocab).to(DEVICE)

    # LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss(ignore_index=a_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    best_loss = float("inf")
    train_losses = []
    val_losses   = []
    vqa_accs     = []
    bleu_list    = []
    rouge_list   = []

    # TRAIN LOOP
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            outputs, answer_target = forward_pass(model, batch)

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                answer_target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss  = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, a_vocab)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        vqa_accs.append(val_metrics["vqa_acc"])
        bleu_list.append(val_metrics["bleu"])
        rouge_list.append(val_metrics["rougeL"])

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"VQA Acc: {val_metrics['vqa_acc']*100:.2f}% | "
            f"BLEU: {val_metrics['bleu']:.4f} | "
            f"ROUGE-L: {val_metrics['rougeL']:.4f}"
        )

        # SAVE BEST
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "a_vocab": a_vocab,
                "model_type": MODEL_TYPE
            }, CHECKPOINT)

            print(f"  Saved best model (Val Loss = {best_loss:.4f})")

        else:
            patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    plot_metrics(train_losses, val_losses, vqa_accs, bleu_list, rouge_list)


# MAIN
if __name__ == "__main__":
    train()
    