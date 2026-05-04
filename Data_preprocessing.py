import os
import re
import json
import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pyvi import ViTokenizer
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# =========================
# CONFIG
# =========================

DATA_DIR =  r"D:\Code\Deep Learning\Endterm\data"
MAX_ANSWER_LEN = 12
PHOBERT_MAX_LEN = 64

# =========================
# LOAD DATA
# =========================

def load_data(json_file, max_samples=None):
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = []
    for item in raw:
        image_path = item["image_path"]
        for q in item.get("questions", []):
            data.append({
                "image_path": image_path,
                "question":   q["question"],
                "answer":     q["answer"]
            })
            if max_samples and len(data) >= max_samples:
                return data
    return data


# =========================
# TOKENIZE
# =========================

def tokenize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


# =========================
# VOCAB
# =========================

def build_vocab(data, key):
    counter = Counter()
    for item in data:
        counter.update(tokenize_text(item[key]))

    vocab = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}

    for word, _ in counter.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


# =========================
# PHOBERT TOKENIZER (LAZY)
# =========================

_phobert_tokenizer = None

def get_phobert_tokenizer():
    global _phobert_tokenizer
    if _phobert_tokenizer is None:
        from transformers import AutoTokenizer
        print("Loading PhoBERT tokenizer...")
        _phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return _phobert_tokenizer

def encode_question(question):
    # 1. Tách từ bằng PyVi trước khi đưa vào Tokenizer
    segmented_question = ViTokenizer.tokenize(question)

    tokenizer = get_phobert_tokenizer()
    encoding = tokenizer(
        segmented_question, # Đổi biến truyền vào
        max_length=PHOBERT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return (
        encoding["input_ids"].squeeze(0),
        encoding["attention_mask"].squeeze(0)
    )

# =========================
# ANSWER ENCODING
# =========================

def encode_answer(answer, vocab):
    tokens = tokenize_text(answer)[:MAX_ANSWER_LEN - 2]
    ids = (
        [vocab["<sos>"]] +
        [vocab.get(t, vocab["<unk>"]) for t in tokens] +
        [vocab["<eos>"]]
    )
    if len(ids) < MAX_ANSWER_LEN:
        ids += [vocab["<pad>"]] * (MAX_ANSWER_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# =========================
# IMAGE TRANSFORM
# =========================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =========================
# DATASET
# =========================

class VQADataset(Dataset):
    def __init__(self, data, a_vocab, transform=None, image_root=DATA_DIR):
        self.data       = data
        self.a_vocab    = a_vocab
        self.transform  = transform
        self.image_root = image_root
        # Preload tokenizer khi khởi tạo dataset
        get_phobert_tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ===== IMAGE =====
        img_path = os.path.join(self.image_root, item["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        question_text = item["question"]
        # ===== QUESTION (PhoBERT cho cả A1 và A2) =====
        # FIX: bỏ rẽ nhánh model_type vì cả 2 model đều dùng PhoBERT encoder
        input_ids, attention_mask = encode_question(item["question"])

        # ===== ANSWER =====
        answer = encode_answer(item["answer"], self.a_vocab)

        return image, input_ids, attention_mask, answer, question_text


# =========================
# LOAD DATASETS
# =========================

def load_datasets(train_file, val_file):
    train_data = load_data(train_file)
    val_data   = load_data(val_file)

    a_vocab = build_vocab(train_data, "answer")

    train_dataset = VQADataset(
        train_data, a_vocab, transform=train_transform
    )
    val_dataset = VQADataset(
        val_data, a_vocab, transform=val_transform
    )

    return train_dataset, val_dataset, a_vocab