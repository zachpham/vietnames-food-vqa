import os
import sys
import torch
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.A1_model import VQAModel  
from Data_preprocessing import val_transform, encode_question, MAX_ANSWER_LEN 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_answer(token_ids, vocab):
    """Chuyển đổi token IDs thành chuỗi văn bản"""
    idx2word = {idx: word for word, idx in vocab.items()}
    words = []
    for token in token_ids:
        if token == vocab.get("<eos>"):
            break
        if token not in [vocab.get("<pad>"), vocab.get("<sos>"), vocab.get("<unk>")]:
            words.append(idx2word.get(token, ""))
    return " ".join(words)

def run_predict_A1(model, image_path, question, vocab):
    
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
        hidden = torch.tanh(model.init_h(combined)).unsqueeze(0)
        cell   = torch.tanh(model.init_c(combined)).unsqueeze(0)
        
        answer_tokens = []
        current_token = vocab["<sos>"]
        
        for _ in range(MAX_ANSWER_LEN):  
            ans_input = torch.tensor([[current_token]], device=device)
            output, hidden, cell = model.decoder(ans_input, hidden, cell)  
            
            pred_token = output.argmax(dim=-1).item()
            if pred_token == vocab["<eos>"]:
                break
            answer_tokens.append(pred_token)
            current_token = pred_token
            
    return decode_answer(answer_tokens, vocab)

if __name__ == "__main__":
    A1_CHECKPOINT = r"D:\Code\Deep Learning\Endterm\data\checkpoints\better_A1.pth"
    TEST_IMAGE = r"D:\Code\Deep Learning\Endterm\data\test.jpg"
    TEST_QUESTION = "Món ăn trong ảnh là gì?"

    print(f"Đang load checkpoint A1 từ: {A1_CHECKPOINT}")
    checkpoint = torch.load(A1_CHECKPOINT, map_location=device)
    

    vocab = checkpoint["a_vocab"]
    print(f"-> Đã nạp vocab thành công (Size: {len(vocab)})")

    model_A1 = VQAModel(answer_vocab_size=len(vocab)).to(device) 
    model_A1.load_state_dict(checkpoint["model_state_dict"]) 
    
    print("\n--- KẾT QUẢ A1 ---")
    ket_qua = run_predict_A1(model_A1, TEST_IMAGE, TEST_QUESTION, vocab)
    print(f"Câu trả lời: {ket_qua}")