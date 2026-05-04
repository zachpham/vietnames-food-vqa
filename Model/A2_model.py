import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


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



# 2. QUESTION ENCODER (PhoBERT)

class QuestionEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")

        # freeze ban đầu
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



# 3. CO-ATTENTION 

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
            mask_exp = attention_mask.unsqueeze(-1)
            q_sum = (proj_q * mask_exp).sum(dim=1, keepdim=True)
            valid_len = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            q_mean = q_sum / valid_len
        else:
            q_mean = proj_q.mean(dim=1, keepdim=True)

        attn_v = torch.bmm(q_mean, proj_v.transpose(1, 2)) / (self.dim ** 0.5)
        alpha_v = F.softmax(attn_v, dim=-1)
        v_hat = torch.bmm(alpha_v, V).squeeze(1)

        return v_hat, q_hat



# 4. TRANSFORMER DECODER

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.3
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        """
        tgt: (B, seq_len)
        memory: (B, 1, 512)
        """

        B, T = tgt.size()
        positions = torch.arange(0, T, device=tgt.device).unsqueeze(0)

        x = self.embedding(tgt) + self.pos_embedding(positions)

        # mask future token
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # mask pad token
        tgt_key_padding_mask = (tgt == 0) 
        
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask 
        )

        out = self.fc(out)
        return out



# 5. FULL MODEL A2

class VQAModel_A2(nn.Module):
    def __init__(self, answer_vocab_size, hidden_dim=512):
        super().__init__()

        self.image_encoder = ImageEncoder(hidden_dim)
        self.question_encoder = QuestionEncoder(hidden_dim)
        self.co_attention = CoAttention(hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.decoder = TransformerDecoder(
            vocab_size=answer_vocab_size,
            embed_dim=hidden_dim
        )
        

    def forward(self, image, question, attention_mask, answer_input):

        V = self.image_encoder(image)
        Q = self.question_encoder(question, attention_mask)

        v_hat, q_hat = self.co_attention(V, Q, attention_mask)

        combined = torch.cat([v_hat, q_hat], dim=1)  # (B,1024)
        combined = self.dropout(combined)
        fused = self.fusion(combined).unsqueeze(1)  # (B,1,512)
        memory = torch.concat([fused, V], dim=1)  # (B,50,512)
        
        output = self.decoder(answer_input, memory)

        return output