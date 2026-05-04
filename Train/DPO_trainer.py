import os
import json
import torch

from datasets import Dataset
from PIL import Image
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


# 1. CẤU HÌNH THÔNG SỐ

BASE_MODEL_ID   = "google/paligemma-3b-pt-224"
SFT_ADAPTER_DIR = r"./B2_finetuned_model"
DPO_JSON_PATH   = r"D:\Code\Deep Learning\Endterm\dpo_data.json"
IMAGE_DIR       = r"D:\Code\Deep Learning\Endterm\data_resized_224"
OUTPUT_DIR      = r"./B3_DPO_model"


# 2. LOAD PROCESSOR

print("Load Processor...")
processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL_ID)

# Lấy image token đúng cách từ processor
# PaliGemma dùng "<image>" hoặc token đặc biệt — lấy từ tokenizer
image_token = processor.tokenizer.convert_ids_to_tokens(
    processor.tokenizer.convert_tokens_to_ids("<image>")
)
# Nếu không tồn tại, fallback về chuỗi literal
if image_token is None or "unk" in str(image_token).lower():
    image_token = "<image>"

print(f"Image token: {image_token}")


# 3. XỬ LÝ DỮ LIỆU DPO

def load_and_process_dpo_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_data = []
    skipped = 0
    print(f"Đang xử lý {len(raw_data)} mẫu...")

    for item in raw_data:
        img_path = os.path.join(IMAGE_DIR, item["image_path"])

        
        chosen_text  = item["chosen"].strip()   
        rejected_text = item["rejected"].strip()    
       
        prompt_text = f"<image><bos>answer vi {item['prompt']}\n"

        try:
            image = Image.open(img_path).convert("RGB")
            processed_data.append({
                "prompt":   prompt_text,
                "chosen":   chosen_text,
                "rejected": rejected_text,
                "images":   [image],
            })
        except Exception as e:
            print(f"  Bỏ qua ảnh lỗi: {img_path} — {e}")
            skipped += 1

    print(f"Hợp lệ: {len(processed_data)} | Bỏ qua: {skipped}")
    return Dataset.from_list(processed_data)



# 4. COLLATOR:


def make_labels_with_mask(full_ids: "torch.Tensor", prompt_ids: "torch.Tensor") -> "torch.Tensor":
    labels = full_ids.clone()
    for i in range(full_ids.size(0)):
        prompt_len = prompt_ids.size(1)
        mask_len = min(prompt_len, full_ids.size(1))
        labels[i, :mask_len] = -100
    return labels

def dpo_data_collator(features):
    prompts   = [f["prompt"]   for f in features]
    chosens   = [f["chosen"]   for f in features]
    rejecteds = [f["rejected"] for f in features]
    images    = [f["images"][0] for f in features]

  
    prompt_lengths = []
    for p, img in zip(prompts, images):
        p_enc = processor(text=p, images=img, return_tensors="pt")
        prompt_lengths.append(p_enc["input_ids"].size(1))


    all_texts = [p + c for p, c in zip(prompts, chosens)] + \
                [p + r for p, r in zip(prompts, rejecteds)]
    all_images = images + images  

    enc = processor(
        text=all_texts,
        images=all_images,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=768,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

  
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    
   
    full_prompt_lengths = prompt_lengths + prompt_lengths

    for i in range(len(full_prompt_lengths)):
        p_len = full_prompt_lengths[i]
        
      
        mask_start = min(p_len, input_ids.size(1))
        
      
        completion_mask[i, mask_start:] = True
        
       
        labels[i, :mask_start] = -100
        
       
        is_pad = (attention_mask[i] == 0)
        labels[i, is_pad] = -100
        completion_mask[i, is_pad] = False

  
    enc["completion_mask"] = completion_mask
    enc["labels"] = labels

    return enc


# 5. MAIN

if __name__ == "__main__":

  
    print("Load Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",   
        torch_dtype=torch.bfloat16,
    )

    # --- Đóng băng vision encoder & projector ---
    for name, param in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            param.requires_grad = False

    # --- Load SFT adapter ---
    print("Load SFT Adapter (B2)...")
   
    model = PeftModel.from_pretrained(model, SFT_ADAPTER_DIR, is_trainable=True)

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # In tóm tắt số param trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- Load dataset ---
    print("Load DPO Dataset...")
    dpo_dataset = load_and_process_dpo_dataset(DPO_JSON_PATH)
    print(f"Tổng số mẫu DPO hợp lệ: {len(dpo_dataset)}")

    import trl
    print(f"TRL version: {trl.__version__}")

    #DPO Config
 
    dpo_config = DPOConfig(
        beta=0.1,
        loss_type="sigmoid",        
        label_smoothing=0.0,
        disable_dropout=True,           
        max_length=768,
        truncation_mode="keep_start",
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,             
        learning_rate=5e-6,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        bf16=True,
        fp16=False,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,  
        seed=42,
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=processor,
        data_collator=dpo_data_collator,
    )

    print("Bắt đầu train DPO...")
    trainer.train()

    # Lưu model 
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Done! Model DPO (B3) đã lưu tại: {OUTPUT_DIR}")