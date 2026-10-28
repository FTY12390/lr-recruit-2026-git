from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import sys
import base64

# ==========================================
# 1. 工程架构：跨目录导入模块
# ==========================================
sys.path.append(r"D:\A\lr2\style_transfer")
from src.model import VGG19Features 

app = Flask(__name__)

# ==========================================
# 2. 初始化：加载模型与预处理
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = VGG19Features().to(device) 

preprocess = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

# ==========================================
# 3. 辅助函数：Gram 矩阵
# ==========================================
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (c * h * w)

# ==========================================
# 4. 核心推理逻辑
# ==========================================
def run_style_transfer(content_tensor, style_tensor, steps=50):
    with torch.no_grad():
        content_feats_true, _ = vgg(content_tensor)
        _, style_feats_true = vgg(style_tensor)
        style_grams_true = {k: gram_matrix(v) for k, v in style_feats_true.items()}

    target_img = content_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target_img], lr=0.01)
    
    content_weight = 1.0
    style_weight = 1e6  

    for step in range(steps):
        optimizer.zero_grad()
        target_content_feats, target_style_feats = vgg(target_img)
        
        content_loss = F.mse_loss(target_content_feats['conv4_2'], content_feats_true['conv4_2'])
        
        style_loss = 0.0
        for layer_name in style_grams_true:
            target_gram = gram_matrix(target_style_feats[layer_name])
            style_loss += F.mse_loss(target_gram, style_grams_true[layer_name])
            
        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        optimizer.step()

    return target_img.detach()

# ==========================================
# 5. 接口定义
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        content_img = Image.open(io.BytesIO(file.read())).convert('RGB')
        content_tensor = preprocess(content_img).unsqueeze(0).to(device)

        style_img_path = r"D:\A\lr2\style_transfer\data\images\style.jpg" 
        
        if not os.path.exists(style_img_path):
            return jsonify({"error": f"Style image not found at {style_img_path}"}), 500
            
        style_img = Image.open(style_img_path).convert('RGB')
        style_tensor = preprocess(style_img).unsqueeze(0).to(device)

        result_tensor = run_style_transfer(content_tensor, style_tensor, steps=50)

        result_tensor = denormalize(result_tensor).cpu().clamp(0, 1)
        result_img = transforms.ToPILImage()(result_tensor.squeeze(0))

        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "message": "Style transfer completed",
            "image_base64": f"data:image/jpeg;base64,{img_base64}"
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)