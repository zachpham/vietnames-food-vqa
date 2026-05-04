from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch    
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_predict_B2(image_path, question, finetuned_dir=r"D:\Code\Deep Learning\Endterm\B2_finetuned_model"):
    """Dự đoán 1 sample sử dụng mô hình B2 đã fine-tune[cite: 4]"""
  
    processor = PaliGemmaProcessor.from_pretrained(finetuned_dir)

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, finetuned_dir)
    model.eval()
    
   
    image = Image.open(image_path).convert("RGB")
    prompt = f"<image>answer vi {question}" #[cite: 4]
    
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt"
    ).to(model.device)

    inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20,
            do_sample=False
        )
        
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = processor.decode(generated_ids, skip_special_tokens=True)
    
    return answer


