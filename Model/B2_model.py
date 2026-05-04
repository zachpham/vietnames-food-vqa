import os
import json
import torch
from datasets import Dataset
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
# 1. CẤU HÌNH THÔNG SỐ
MODEL_ID        = "google/paligemma-3b-pt-224"
TRAIN_JSON_PATH = r"D:\Code\Deep Learning\Endterm\data\train.json"
VAL_JSON_PATH   = r"D:\Code\Deep Learning\Endterm\data\val.json"
IMAGE_DIR       = r"D:\Code\Deep Learning\Endterm\data_resized_224"
OUTPUT_DIR      = r"./B2_finetuned_model"

# 2. LOAD PROCESSOR & MODEL 
print("Load Processor...")
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
# 3. FLATTEN DATASET
def flatten_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    flat = []
    for item in raw:
        img_path = os.path.join(IMAGE_DIR, item["image_path"])
        for q_obj in item["questions"][:5]: 
            flat.append({
                "image_path": img_path,
                "question":   q_obj["question"],
                "answer":     q_obj["answer"].strip()
            })
    return Dataset.from_list(flat)

# 4. DATA COLLATOR
def collate_fn(examples):
    texts  = [f"<image>answer vi {ex['question']}" for ex in examples]
    labels = [ex['answer'] for ex in examples]
    images = [Image.open(ex['image_path']).convert("RGB") for ex in examples]

    inputs = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest"
    )

    inputs = {
        k: v.to(torch.bfloat16) if v.dtype == torch.float32
        else v.to(torch.long)   if v.dtype == torch.int32
        else v
        for k, v in inputs.items()
    }
    return inputs

# 5. TRAINING ARGUMENTS
if __name__ == "__main__":
    
    print("Load Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_use_double_quant=True,
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
        
    )
    
        # Freeze vision encoder & projector
    for name, param in model.model.named_parameters():
        if "vision" in name or "projector" in name:
            param.requires_grad = False

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total  = sum(p.numel() for p in model.parameters())
    print(f"Frozen: {frozen/1e6:.1f}M / {total/1e6:.1f}M params")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Load datasets...")
    train_dataset = flatten_dataset(TRAIN_JSON_PATH)
    val_dataset   = flatten_dataset(VAL_JSON_PATH)
    print(f"Train: {len(train_dataset)} cap | Val: {len(val_dataset)} cap")


    # 5. TRAINING ARGUMENTS

    training_args = TrainingArguments(
        num_train_epochs=2, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        adam_beta2=0.999,
        logging_steps=20,
        optim="paged_adamw_8bit",
        save_total_limit=1,
        output_dir=OUTPUT_DIR,
        bf16=True,
        
        dataloader_pin_memory=False,
        dataloader_num_workers=0, 
        
        eval_accumulation_steps=10,
        eval_strategy="epoch",              
        save_strategy="epoch",  
        load_best_model_at_end=True, 
        remove_unused_columns=False
    )


    # 6. TRAIN

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Bat dau Fine-tune B2...")
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model da luu tai: {OUTPUT_DIR}")