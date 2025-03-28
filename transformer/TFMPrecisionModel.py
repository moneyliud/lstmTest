import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim
from tqdm import tqdm


# 1. 位置编码（Positional Encoding）
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0))
#
#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]


# 2. 自注意力机制（Self-Attention）
class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 线性变换并分头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax和加权求和
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        # 合并多头输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(output)


# 3. Transformer编码器层
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 8 * d_model),
            nn.ReLU(),
            nn.Linear(8 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力和残差连接
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络和残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


# 4. 完整Transformer模型（仅解码器版本，类似GPT）
class TransformerLM(nn.Module):
    def __init__(self, seq_len, d_model, output_size, num_layers=4, num_heads=8):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x, mask=None):
        # x = self.embedding(x)
        # x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc(x)


# 使用示例
if __name__ == "__main__":
    # 超参数
    vocab_size = 10000  # 词汇表大小
    seq_length = 100  # 序列长度
    batch_size = 32

    # 初始化模型
    model = TransformerLM(vocab_size)

    # 创建虚拟输入数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    # 前向传播
    outputs = model(input_ids)
    print("输出形状:", outputs.shape)  # 应为 [batch_size, seq_length, vocab_size]


    # 生成文本示例
    def generate(prompt, max_length=20, temperature=0.7):
        model.eval()
        tokens = prompt
        for _ in range(max_length):
            # 创建注意力mask
            mask = torch.triu(torch.ones((len(tokens), len(tokens))), diagonal=1).bool()
            with torch.no_grad():
                logits = model(torch.tensor([tokens]))[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
        return tokens


    # 示例生成
    print("生成的文本索引:", generate([100, 200, 300]))  # 需要真实词汇表映射才能得到实际文本
