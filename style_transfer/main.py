import torch
from src.model import VGG19Features
from src.utils import load_image
from src.trainer import StyleTransferTrainer

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    content_path=r"D:\A\lr2\style_transfer\data\images\content.jpg"
    style_path=r"D:\A\lr2\style_transfer\data\images\style.jpg"

    content_img=load_image(content_path,max_size=512).to(device)
    style_img=load_image(style_path,max_size=512).to(device)

    print("正在加载VGG19模型")
    vgg_model=VGG19Features().to(device)

    trainer=StyleTransferTrainer(
        model=vgg_model,
        content_img=content_img,
        style_img=style_img,
        device=device,
        content_weight=1.0,
        style_weight=1e6,
        tv_weight=1e-6,
        lr=0.01
    )

    trainer.train(steps=500,print_every=50)

if __name__=='__main__':
    main()

