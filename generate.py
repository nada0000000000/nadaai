"""
학습된 모델(ckpt.pt)로 실제 텍스트를 생성해본다.

사용법:
    python generate.py
"""
import os
import pickle
import torch
from model import MiniGPT

here = os.path.dirname(__file__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(here, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']

block_size = 256
model = MiniGPT(vocab_size, block_size, n_layer=6, n_head=6, n_embd=384, dropout=0.0).to(device)
model.load_state_dict(torch.load(os.path.join(here, 'ckpt.pt'), map_location=device))
model.eval()


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join([itos[i] for i in ids])


# 시작 문장 (원하는 대로 바꿔보세요)
prompt = "\n"
start_ids = encode(prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=50)

print(decode(y[0].tolist()))
