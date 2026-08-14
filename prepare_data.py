"""
학습 데이터를 준비하는 스크립트.
input.txt 파일(학습시키고 싶은 텍스트)을 읽어서
문자 단위(character-level) 토큰화 후 train/val로 나눠 저장한다.

사용법:
    1. 이 폴더에 input.txt 파일을 넣는다 (학습시키고 싶은 텍스트, 클수록 좋음)
    2. python prepare_data.py
"""
import os
import pickle
import numpy as np

input_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_path):
    raise FileNotFoundError(
        f"{input_path} 가 없어요. 학습시키고 싶은 텍스트 파일을 input.txt 이름으로 이 폴더에 넣어주세요."
    )

with open(input_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"전체 글자 수: {len(data):,}")

# 등장하는 모든 고유 문자를 어휘집(vocab)으로 만든다
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"고유 문자 종류: {vocab_size}")
print(f"어휘 예시: {''.join(chars[:50])}")

# 문자 <-> 정수 매핑
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join([itos[i] for i in ids])


# 90%는 학습용, 10%는 검증용으로 분리
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"학습 토큰 수: {len(train_ids):,}")
print(f"검증 토큰 수: {len(val_ids):,}")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 나중에 모델이 다시 디코딩할 수 있도록 어휘집 정보 저장
meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("완료! train.bin, val.bin, meta.pkl 생성됨.")
