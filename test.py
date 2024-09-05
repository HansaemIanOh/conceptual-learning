import os
import argparse
import logging
import yaml
from flask import Flask, request, jsonify, send_from_directory
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from safetensors.torch import load_file
from models import SimpleTransformer
from textdataloader import TextDataModule

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='filename', default='configs/flowers_cnn.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 현재 스크립트의 디렉토리를 확인
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, 'static')

app = Flask(__name__, static_folder=static_folder, static_url_path='')

# 전역 변수로 model과 tokenizer 선언
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    
    # TensorBoard 로거 설정
    tb_logger = TensorBoardLogger(save_dir=config['log_config']['save_dir'], name=config['log_config']['name'])

    data_module = TextDataModule(**config['data_config'])
    tokenizer = data_module.tokenizer

    config['model_config']['tokenizer'] = tokenizer
    config['model_config']['max_length'] = config['data_config']['max_length']
    
    # 모델 로드
    logger.info("Loading model...")
    model = SimpleTransformer(config['model_config'])

    if config['log_config']['checkpoint']:
        try:
            current_version = int(tb_logger.version)
            prev_version_dir = os.path.join(os.path.dirname(tb_logger.log_dir), f'version_{current_version-1}')
            last_checkpoint = os.path.join(prev_version_dir, "checkpoints", "last.safetensors")
            if os.path.exists(last_checkpoint):
                model = model.__class__.load_from_checkpoint(last_checkpoint, config=config['model_config'])
                logger.info(f"Loaded checkpoint from {last_checkpoint}")
            else:
                logger.info(f"No checkpoint found at {last_checkpoint}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    else:
        logger.info("Checkpoint loading is disabled in config.")

    logger.info("Model loaded successfully")

def process_attention_mask(attention_mask, seq_length, device):
    # attention_mask를 bool 타입으로 변환
    bool_attention_mask = attention_mask.bool()

    # 인과관계(causal) 마스크 생성
    causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()

    # attention_mask와 causal_mask 결합
    combined_mask = bool_attention_mask.unsqueeze(1) & ~causal_mask.unsqueeze(0)

    return combined_mask.squeeze(0)

def generate_text(input_text, max_length=100, temperature=0.7):
    global model, tokenizer
    
    # 입력 텍스트 토큰화
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)
    
    # 특수 토큰 ID 설정
    eos_token_id = tokenizer.eos_token_id
    
    # 텍스트 생성
    generated_ids = input_ids[0].tolist()
    for _ in range(max_length):
        with torch.no_grad():
            inputs = torch.tensor([generated_ids]).to(model.device)
            pad_attn_mask = torch.ones_like(inputs).to(model.device)
            outputs = model(inputs, pad_attn_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).squeeze()
            generated_ids.append(next_token.item())
            if next_token.item() == eos_token_id:
                break
    
    # 생성된 텍스트 디코딩 (입력을 제외한 새로 생성된 부분만)
    generated_text = tokenizer.decode(generated_ids[len(input_ids[0]):], skip_special_tokens=True)
    return generated_text

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_text_api():
    data = request.json
    input_text = data['input']
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500
    
    generated_text = generate_text(input_text)
    
    return jsonify({'response': generated_text})

def create_index_html():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SimpleTransformer WebUI</title>
        <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    </head>
    <body>
        <h1>SimpleTransformer WebUI</h1>
        <textarea id="input" rows="4" cols="50"></textarea>
        <button onclick="generateText()">Generate</button>
        <div id="output"></div>

        <script>
            async function generateText() {
                const input = document.getElementById('input').value;
                const output = document.getElementById('output');
                
                try {
                    const response = await axios.post('/generate', { input: input });
                    output.innerText = response.data.response;
                } catch (error) {
                    console.error('Error:', error);
                    output.innerText = 'An error occurred while generating text.';
                }
            }
        </script>
    </body>
    </html>
    """
    os.makedirs(static_folder, exist_ok=True)
    with open(os.path.join(static_folder, 'index.html'), 'w') as f:
        f.write(html_content)
    logger.info(f"Created index.html in {static_folder}")

if __name__ == '__main__':
    load_model_and_tokenizer()
    create_index_html()
    logger.info("Starting server on http://localhost:5000")
    app.run(debug=False, port=5000)