import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from einops.layers.torch import Rearrange
from einops import rearrange, reduce
from timm.models.layers import DropPath
from fastdtw import fastdtw
from scipy.spatial.distance import pdist, squareform, cdist
import math

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):

        flat_inputs = inputs.view(-1, self.embedding_dim)

        # 计算输入和每个嵌入之间的距离
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

        # 选择最近的嵌入
        encoding_indices = torch.argmin(distances, dim=1)   # [224, 1024]
        # input.shape :  #[224, 12, 512]
        quantized = self.embedding(encoding_indices).view(inputs.shape)

        # 计算损失
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized, encoding_indices
       
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(-2)]
    

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        # self.vector_quantizer = VectorQuantizer(1000, d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # x_value_embedding = self.value_embedding(x)
        # Input encoding
        # quantized_x_enc = self.vector_quantizer(x_value_embedding)
        add_out = self.value_embedding(x) + self.position_embedding(x)
        # [224, 12, 512]
        patch_features = self.dropout(add_out)
        
        return patch_features, n_vars
    

class SelfAttentionDtwBias(nn.Module):
    def __init__(self, d_k, d_DtwCost, d_heads):
        super().__init__()
        self.d_k = d_k
        self.linear = nn.Linear(d_DtwCost, d_heads)
    def forward(self, Q, K, V, dtw_cost, mask=None): 
        """
        dtw_cost : [32,21,96,96]
        """

        if dtw_cost is None:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)    # [32, 8, 96, 96]
        else:
            dtw_cost = dtw_cost.permute(0,2,3,1)  # [32, 96, 96, 21]
            dtw_cost_h = self.linear(dtw_cost)   # [32, 96, 96, 8]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)    # [32, 8, 96, 96]
            scores = scores.permute(0, 2, 3, 1)
            scores = scores+dtw_cost_h 
            scores = scores.permute(0, 3, 1, 2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores , dim=-1)  # 32, 8, 96, 96    
        output = torch.matmul(attention, V)
        return output, attention


class MultiHeadAttention_var(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_DtwCost = 21
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  #  512， num_head=8 =  each head -> 64     512-> (8, 64)
      
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.gating = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
            )
        self.out_linear = nn.Linear(d_model, d_model)

    
        self.attention_bias = SelfAttentionDtwBias(self.d_k, self.d_DtwCost, self.num_heads)

    def forward(self, Q, K, V, G, dtw_cost, mask=None):
        bs = Q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(K).view(bs, -1, self.num_heads, self.d_k) 
        q = self.q_linear(Q).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(V).view(bs, -1, self.num_heads, self.d_k)
        # gate = self.gating(G).view(bs, -1, self.num_heads, self.d_k) 
        #[32, 96, 8, 64]
        
        # transpose to get dimensions bs * num_heads * seq_len * d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)       #[32, 8, 96, 64]
        # gate = gate.transpose(1,2)  #[32, 8, 96, 64]

        scores, attn_weight = self.attention_bias(q, k, v, dtw_cost, mask)
        # gated_scores = gate * scores   #[32, 8, 96, 64] -----> [32, 96, 512]
        # concatenate heads and put through final linear layer   #[32, 96, 8, 64]
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)  #[32, 96, 512]
        output = self.out_linear(concat)

        return output, attn_weight


class SelfAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, mask=None): 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)    # [32, 8, 7, 7]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores , dim=-1)  # 32, 8, 96, 96    
        output = torch.matmul(attention, V)
        return output, attention
    
   
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  #  512， num_head=8 =  each head -> 64     512-> (8, 64)
      
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        # self.gating = nn.Sequential(
        #     nn.Linear(d_model, d_model),
        #     nn.Sigmoid(),
        #     )
        
        self.out_linear = nn.Linear(d_model, d_model)

    
        self.attention = SelfAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        bs = Q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(K).view(bs, -1, self.num_heads, self.d_k) 
        q = self.q_linear(Q).view(bs, -1, self.num_heads, self.d_k)
        v = self.v_linear(V).view(bs, -1, self.num_heads, self.d_k)
        # gate = self.gating(G).view(bs, -1, self.num_heads, self.d_k)  #[32, 96, 8, 64]
        
        # transpose to get dimensions bs * num_heads * seq_len * d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)       #[32, 8, 96, 64]
        # gate = gate.transpose(1,2)  #[32, 8, 96, 64]

        scores, attn_weight = self.attention(q, k, v, mask)
        # scores = gate * scores   #[32, 8, 96, 64] -----> [32, 96, 512]
        # concatenate heads and put through final linear layer   #[32, 96, 8, 64]
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)  #[32, 96, 512]
        output = self.out_linear(concat)

        return output, attn_weight


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class AttnFFLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(AttnFFLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)    
        #[32, 96, 512] -> [32, 96, 8, 64]->[32, 8, 96, 64]
        # self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        # self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):  # [32, 96, 512]
        src2, _ = self.multi_head_attention(src, src, src)
        src_3 = self.layer_norm1(src + self.dropout(src2))  # [32, 96, 512]
        # src_4 = self.feed_forward(src_3)
        # output = self.layer_norm2(src_3 + self.dropout(src_4))  # [32, 96, 512]
        output = src_3
        return output    
          

class DataEmbedding_seq(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_seq, self).__init__()
        self.value_embedding = nn.Sequential(
        nn.Linear(c_in, d_model),
        # nn.GELU(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEmbedding(d_model=d_model)

    def forward(self, x, x_mark=None):
       # x = x.permute(0,2,1) # [32, 7, 96]
        
        x = x.unsqueeze(dim=3)  #[32,7,96,1]
        if x_mark is None:
            x = self.value_embedding(x) # [32, 96, 7, 512]   / [32,7,96,512]
        else:
            x_mark = x_mark.permute(0,2,1) 
            x_mark = x_mark.unsqueeze(dim=3)
            x = self.value_embedding(torch.cat([x, x_mark], dim=-2))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class MLP_Block(nn.Module):
    def __init__(self, d_in, d_out, d_ff=64, dropout=0.1):
        super(MLP_Block, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_out),
            DropPath(dropout))           
        
        if d_in == d_out:
            self.jump_net = nn.Identity()
        else:
            self.jump_net = nn.Linear(d_in, d_out)   #32, 96, 512

    def forward(self, x):
        return self.jump_net(x) + self.net(x)    # 32, 96, 7 
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.attn_seq = AttnFFLayer(d_model, num_heads, d_ff, dropout)

    def forward(self, x):
        return self.attn_seq(x)
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super().__init__()
        self.net = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self,x):
        for layer in self.net:
            x = layer(x)
        return self.norm(x)


class MyNet(nn.Module):
    def __init__(self, seq_len, pred_len, in_chn, ex_chn, out_chn, d_model, num_heads, d_ff, dropout, num_layers=1,  patch_len=1, stride=1):
        super(MyNet, self).__init__()
        self.out_chn = out_chn
  
        self.embedding = DataEmbedding_seq(c_in=1, d_model=d_model, dropout=dropout)
        # self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout)
        # self.encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.attn1 = AttnFFLayer(d_model, num_heads, d_ff, dropout)
        self.linear = nn.Linear(seq_len*d_model, d_model)

        self.attn2 = AttnFFLayer(d_model, num_heads, d_ff, dropout)

        self.projection = nn.Linear(seq_len*d_model, pred_len)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
     

    def forward(self, x, x_mark=None,  attn_mask=None):

        """
        x: [-1, 96, 7]
        """
        basz,seq_len,n_vars = x.shape   

        x = x.permute(0,2,1)

        enc_out = self.embedding(x)     # [32， 7， 96,  d_model]
        
        for i in range(10):
            enc_out = torch.reshape(enc_out, (enc_out.shape[0]* enc_out.shape[1], enc_out.shape[2],  enc_out.shape[3]))

            enc_out = self.attn1(enc_out)                  # [224, 96, d_model]

            enc_out = torch.reshape(enc_out, (basz, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

            enc_out = enc_out.permute(0,2,1,3)     # [32, 96, 7, d_model]
            enc_out = self.norm(enc_out)
            enc_out = torch.reshape(enc_out, (enc_out.shape[0]* enc_out.shape[1], enc_out.shape[2],  enc_out.shape[3]))
            enc_out = self.attn1(enc_out)
            enc_out = torch.reshape(enc_out, (basz, seq_len, n_vars, enc_out.shape[-1]))
            enc_out = enc_out.permute(0,2,1,3)   # [32, 7, 96, 512]

        # enc_out = torch.reshape(enc_out, (enc_out.shape[0] * enc_out.shape[1], enc_out.shape[2],  enc_out.shape[3]))

        
        # enc_out = self.norm(enc_out)

        # enc_out = self.attn2(enc_out)    # [32, 7, d_model]
            
        enc_out = torch.reshape(enc_out, (basz, n_vars, -1)) 
        deout = self.projection(enc_out)        # [32, 7, 120]
        out = deout.permute(0, 2, 1)[:,:,:self.out_chn]       
        return out
    
        

    # def forward(self, x, x_mark=None, dtw_cost=None, attn_mask=None):
    #     basz,_,_ = x.shape

    #     x_temp = self.val_emb(x, x_mark)    # [32, 96, 7, 512]

    #     for i in range(len(self.attn_layers)):
    #         x_temp = torch.reshape(x_temp, (x_temp.shape[0]*x_temp.shape[1], x_temp.shape[2], x_temp.shape[3])) 
    #         # [32*7, 96, 512]
            
    #         x_temp_out,_ = self.attn_layers[i](x_temp,  x_temp,  x_temp) 
    #         # mlp_out = self.mlp_blocks[i](x_temp_out)

    #         x_temp = self.norm[i](x_temp + self.dropout(x_temp_out))

    

    #         x_temp = torch.reshape(x_temp, (basz, -1, x_temp.shape[-2], x_temp.shape[-1]))  #[32,7,96,512]
    #         x_temp = x_temp.permute(0, 2, 1, 3)      # [32,96,7,512]  / [32,7,96,512]                                          #[32,96,7,512]
        
    #     x_var = x_temp.permute(0,2,1,3) #[32, 7, 96, 512]
    #     # x_var = x_temp
    #     x_var = torch.reshape(x_var, (basz, -1, x_var.shape[-2]*x_var.shape[-1]))

    #     out = self.projection(x_var)        # [32, 7, 120]
    #     dec_out = out.permute(0, 2, 1)[:,:,:self.out_chn]       
    #     return dec_out
    

'''
class DTWAttnV3(nn.Module):
    def __init__(self, seq_len, pred_len, in_chn, ex_chn, out_chn, d_model, num_heads, d_ff, dropout, num_layers=6, max_len=5000):
        super(DTWAttnV3, self).__init__()
        self.out_chn = out_chn
        # self.layers = num_layers
        self.val_emb = DataEmbedding_seq(1, d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.encod_layer_1 = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encod_layer_2 = EncoderLayer(d_model, num_heads, d_ff, dropout)

        self.projection = nn.Linear(seq_len*d_model, pred_len)

        self.multi_layers = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])
        
        self.layer_norm = nn.LayerNorm(d_model) 

        # self.layer_norm_2 = nn.LayerNorm(d_model) 

        self.dropout = nn.Dropout(dropout)
        # self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, x_mark=None, dtw_cost=None, attn_mask=None):
        basz,_,_ = x.shape
        x_temp = self.val_emb(x, x_mark)    # [32, 96, 7, 512]

        for layer in self.multi_layers:
            x_temp = torch.reshape(x_temp, (x_temp.shape[0]*x_temp.shape[1], x_temp.shape[2], x_temp.shape[3]))
            x_temp_out,_ = layer(x_temp,  x_temp, x_temp) 
            x_temp = self.layer_norm(x_temp + self.dropout(x_temp_out)) 

            x_temp = torch.reshape(x_temp, (basz, -1, x_temp.shape[-2], x_temp.shape[-1]))  #[32,7,96,512]
            x_temp = x_temp.permute(0, 2, 1, 3)                                                  #[32,96,7,512]
        
        x_var = x_temp.permute(0,2,1,3) #[32, 7, 96, 512]
        x_var = torch.reshape(x_var, (basz, -1, x_var.shape[-2]*x_var.shape[-1]))

        # x_temp = torch.reshape(x_temp, (x_temp.shape[0]*x_temp.shape[1], x_temp.shape[2], x_temp.shape[3]))

        # x_temp,_ = self.mha(x_temp,x_temp,x_temp)                    #[32*96, 7, 512]
        # x_temp = torch.reshape(x_temp, (basz, seq_len, x_temp.shape[-2], x_temp.shape[-1]))
        #                                                             # [32, 96, 7, 512]

        # x_var = x_temp.permute(0, 2, 1, 3)    # [32, 96, 7, 512] -> [32, 7, 96, 512]
        # x_var = torch.reshape(x_var, (x_var.shape[0]*x_var.shape[1], x_var.shape[2], x_var.shape[3]))

        # x_var,_ = self.mha(x_var,x_var,x_var)                      

        # x_var = self.encod_layer_1(x_var, attn_mask)     # [32*7, 96, 512]
        # x_var = torch.reshape(x_var, (basz, x_temp.shape[2], x_var.shape[-2]*x_var.shape[-1]))  # 


        out = self.projection(x_var)        # [32, 7, 120]
        dec_out = out.permute(0, 2, 1)[:,:,:self.out_chn]       
        return dec_out
'''


if __name__ == "__main__":

    data_sample = torch.randn(32,96,7)
    # # pair_dtw_cost = torch.randn(32,pair_chn,input_seq,input_seq)
    # model = DTWAttnV3(seq_len=96,pred_len=96,out_chn=7,d_model=512,num_heads=8,d_ff=2048, num_layers=2,dropout=0.1, )
    # output = model(data_sample)
    # print(output.shape)



