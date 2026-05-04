import os
import sys
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.A2_model import VQAModel_A2  #[cite: 9]
from Data_preprocessing import val_transform, encode_question, MAX_ANSWER_LEN 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_answer(token_ids, vocab):
    idx2word = {idx: word for word, idx in vocab.items()}
    words = []
    for token in token_ids:
        if token == vocab.get("<eos>"):
            break
        if token not in [vocab.get("<pad>"), vocab.get("<sos>"), vocab.get("<unk>")]:
            words.append(idx2word.get(token, ""))
    return " ".join(words)

def run_predict_A2(model, image_path, question, vocab):

    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    input_ids, attention_mask = encode_question(question)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        V = model.image_encoder(image_tensor)
        Q = model.question_encoder(input_ids, attention_mask)
        v_hat, q_hat = model.co_attention(V, Q, attention_mask)  
        
        combined = torch.cat([v_hat, q_hat], dim=1)
        fused = model.fusion(combined).unsqueeze(1)
        memory = torch.concat([fused, V], dim=1) 
        
        answer_tokens = [vocab["<sos>"]]
        
        for _ in range(MAX_ANSWER_LEN):
            ans_input = torch.tensor([answer_tokens], device=device)
            output = model.decoder(ans_input, memory)  #[cite: 9]
            pred_token = output[:, -1, :].argmax(dim=-1).item()
            
            if pred_token == vocab["<eos>"]:
                break
            answer_tokens.append(pred_token)
            
    return decode_answer(answer_tokens[1:], vocab)

if __name__ == "__main__":
    A2_CHECKPOINT = r"D:\Code\Deep Learning\Endterm\data\checkpoints\better_A2.pth"
    TEST_IMAGE = r"D:\Code\Deep Learning\Endterm\test.jpg"
    TEST_QUESTION = "Món ăn trong ảnh là gì?"

    print(f"Đang load checkpoint A2 từ: {A2_CHECKPOINT}")
    checkpoint = torch.load(A2_CHECKPOINT, map_location=device)
    

    vocab = checkpoint["a_vocab"]

    model_A2 = VQAModel_A2(answer_vocab_size=len(vocab)).to(device) 
    model_A2.load_state_dict(checkpoint["model_state_dict"])
    
    print("\n--- KẾT QUẢ A2 ---")
    ket_qua = run_predict_A2(model_A2, TEST_IMAGE, TEST_QUESTION, vocab)
    print(f"Câu trả lời: {ket_qua}")