"""
학습된 미니 GPT(ckpt.pt)를 웹에서 쓸 수 있도록 API 서버로 감싼다.

사용법:
    pip install flask flask-cors torch numpy
    python serve.py

그 다음 chatbot.html 을 브라우저로 열면 이 서버한테 요청을 보낸다.
(서버와 html이 같은 컴퓨터에 있으면 그대로 되고, 서버를 클라우드에
 올리면 chatbot.html 안의 API_URL만 그 주소로 바꾸면 됨)
"""
import os
import pickle
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
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
    # 학습 데이터에 없던 문자가 들어오면 무시(모델이 모르는 글자라 에러 남 방지)
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return ''.join([itos[i] for i in ids])


app = Flask(__name__)
CORS(app)  # 브라우저에서 다른 주소로 요청 보낼 수 있게 허용


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_new_tokens = int(data.get('max_new_tokens', 300))

    if not prompt:
        prompt = "\n"

    ids = encode(prompt)
    if not ids:
        return jsonify({'error': '입력한 문자를 모델이 하나도 몰라요 (학습 데이터에 없는 글자예요).'}), 400

    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)

    full_text = decode(y[0].tolist())
    # 프롬프트 뒤에 새로 생성된 부분만 반환
    new_text = full_text[len(prompt):]

    return jsonify({'text': new_text})


if __name__ == '__main__':
    # Railway/Render 같은 호스팅은 PORT 환경변수로 포트를 지정해줌.
    # 로컬에서 그냥 돌릴 땐 없으니 5000번 기본값 사용.
    port = int(os.environ.get('PORT', 5000))
    print(f"모델 로드 완료 (device: {device}). 서버 시작... (port {port})")
    app.run(host='0.0.0.0', port=port)
