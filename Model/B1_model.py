import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from huggingface_hub import login


login(token="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")

class PaliGemmaVQA:
    def __init__(self, model_id="google/paligemma-3b-pt-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Đang khởi chạy PaliGemma trên: {self.device}")
        
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        ).eval()
        print(" Đã tải model PaliGemma thành công!")

    def predict(self, image_pil: Image.Image, question: str) -> str:
        """Thực hiện zero-shot inference."""
        prompt = (
            f"<image>Hãy trả lời câu hỏi sau bằng tiếng Việt, "
            f"ngắn gọn 10 từ.\n"
            f"Câu hỏi: {question}\nTrả lời:"
        )
        
        inputs = self.processor(text=prompt, images=image_pil, return_tensors="pt").to(self.device)
        
        if self.device == "cuda":
            inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_answer = self.processor.decode(generated_ids, skip_special_tokens=True)
        return raw_answer
