import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# 1. IMAGE ENCODER

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # Load EfficientNet-B3 pretrained trên ImageNet
        efficientnet = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-2])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
     
        self.conv = nn.Conv2d(1536, output_dim, kernel_size=1)
        
    def forward(self, x):
        x = self.backbone(x)           
        x = self.conv(x)             
        
       
        x = x.view(x.size(0), x.size(1), -1)  
        x = x.permute(0, 2, 1)                
        return x


# 2. QUESTION ENCODER 

class QuestionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        
        for param in self.phobert.parameters():
            param.requires_grad = False
            
        self.fc = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = outputs.last_hidden_state 
        x = self.fc(x)                
        return x


# 3. CO-ATTENTION BLOCK

class CoAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.W_v = nn.Linear(dim, dim)
        self.W_q = nn.Linear(dim, dim)

    def forward(self, V, Q, attention_mask=None):
        
        proj_v = self.W_v(V)
        proj_q = self.W_q(Q)
        
     
        # PHA 1: Ảnh đi hỏi Câu hỏi (Image attends to Question)
        
        v_mean = proj_v.mean(dim=1, keepdim=True)
        attn_q = torch.bmm(v_mean, proj_q.transpose(1, 2)) / (self.dim ** 0.5) 
        
       
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
          
            attn_q = attn_q.masked_fill(mask == 0, -1e9)
            
        alpha_q = F.softmax(attn_q, dim=-1) 
        q_hat = torch.bmm(alpha_q, Q).squeeze(1) 

        
        # PHA 2: Câu hỏi đi hỏi Ảnh (Question attends to Image)
     
      
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1) 
            q_sum = (proj_q * mask_expanded).sum(dim=1, keepdim=True) 
       
            valid_lengths = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            q_mean = q_sum / valid_lengths
        else:
            q_mean = proj_q.mean(dim=1, keepdim=True)
            
        attn_v = torch.bmm(q_mean, proj_v.transpose(1, 2)) / (self.dim ** 0.5)
      
        alpha_v = F.softmax(attn_v, dim=-1) 
        v_hat = torch.bmm(alpha_v, V).squeeze(1)

        return v_hat, q_hat


# 4. ANSWER DECODER (Giữ nguyên không thay đổi)

class AnswerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, answer_input, hidden, cell):
        embedded = self.embedding(answer_input)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden, cell


# 5. FULL VQA MODEL

class VQAModel(nn.Module):
    def __init__(self, answer_vocab_size, hidden_dim=512, embed_dim=300):
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim)
        
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)
        
        # Thêm block CoAttention
        self.co_attention = CoAttention(hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
        
        self.decoder = AnswerDecoder(
            answer_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim
        )

        # Project để đưa vào LSTM Decoder
        self.init_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.init_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, image, question, attention_mask, answer_input):
        V = self.image_encoder(image)
        Q = self.question_encoder(question, attention_mask)
        
   
        v_hat, q_hat = self.co_attention(V, Q, attention_mask) 
        
        # Concat 2 vector đã qua tinh chế
        combined = torch.cat([v_hat, q_hat], dim=1)  # (batch, 1024)

        combined = self.dropout(combined)
        
        # Khởi tạo trạng thái cho Decoder
        hidden = torch.tanh(self.init_h(combined)).unsqueeze(0)  
        cell   = torch.tanh(self.init_c(combined)).unsqueeze(0)  
        # Sinh câu trả lời
        output, _, _ = self.decoder(answer_input, hidden, cell)  

        return output