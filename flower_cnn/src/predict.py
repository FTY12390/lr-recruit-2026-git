import torch
from PIL import Image
from torchvision import transforms
from model import MyModel
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

def predict_image(image_path):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names=['daisy','dandelion','rose','sunflower','tulip']

    model=MyModel(num_classes=5).to(device)
    model_path=os.path.join(BASE_DIR,'best_model.pth')
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    image=Image.open(image_path).convert('RGB')
    image_tensor=transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output=model(image_tensor)
        _,predicted=torch.max(output,1)
        result=class_names[predicted.item()]

    print(f"图片路径：{os.path.basename(image_path)}")
    print(f"预测结果：{(result)}")

if __name__=='__main__':
    test_image_path=test_image_path = r"D:\A\lr2\flower_cnn\data\flower_copy\test\rose\295257304_de893fc94d.jpg"
    predict_image(test_image_path)
