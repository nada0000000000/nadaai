"""
미니 GPT를 처음부터 학습시키는 스크립트.

사용법:
    1. prepare_data.py 먼저 실행 (train.bin, val.bin, meta.pkl 생성)
    2. python train.py

CPU에서도 돌아가지만 GPU가 있으면 훨씬 빠르다.
(RunPod, Vast.ai 등에서 GPU 인스턴스 빌리면 몇천 원대로 몇 시간 안에 끝남)
"""
import os
import pickle
import time
import numpy as np
import torch
from model import MiniGPT

# ---- 설정값 (처음엔 이대로, 익숙해지면 바꿔보세요) ----
batch_size = 64          # 한 번에 몇 개 문장을 같이 학습할지
block_size = 256         # 한 번에 몇 글자까지 문맥으로 볼지
n_layer = 6              # 층 개수 (깊이)
n_head = 6               # attention head 개수
n_embd = 384             # 임베딩 차원 (모델 크기)
dropout = 0.2
learning_rate = 3e-4
max_iters = 5000         # 총 학습 스텝 수
eval_interval = 250      # 몇 스텝마다 검증할지
eval_iters = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -------------------------------------------------

print(f"device: {device}")
here = os.path.dirname(__file__)

with open(os.path.join(here, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
print(f"vocab_size: {vocab_size}")

train_data = np.memmap(os.path.join(here, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(here, 'val.bin'), dtype=np.uint16, mode='r')


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


model = MiniGPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"파라미터 수: {n_params:,} ({n_params/1e6:.1f}M)")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


print("학습 시작...")
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t0
        print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ({elapsed:.0f}s)")
        torch.save(model.state_dict(), os.path.join(here, 'ckpt.pt'))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("학습 완료! ckpt.pt 에 모델 저장됨.")
