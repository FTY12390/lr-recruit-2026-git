import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def load_image(image_path,max_size=512,shape=None):
    image=Image.open(image_path).convert('RGB')

    if shape is not None:
        transform_list=[transforms.Resize(shape)]
    else:
        transform_list=[transforms.Resize(max_size)]
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])

    transform=transforms.Compose(transform_list)

    return transform(image).unsqueeze(0)

def unnormalize(tensor):
    tensor=tensor.clone().detach()

    for t,m,s in zip(tensor,IMAGENET_MEAN,IMAGENET_STD):
        t.mul_(s).add_(m)

    return tensor.clamp(0,1)

def gram_matrix(input_tensor):
    b,c,h,w=input_tensor.size()
    features=input_tensor.view(b,c,h*w)

    G=torch.bmm(features,features.transpose(1,2))

    return G.div(c*h*w)

if __name__=='__main__':
    test_image_path=r"D:\A\lr2\style_transfer\data\images\5fe0c0d64907cea80209d80b852386ae.jpg"
    img_tensor=load_image(test_image_path)
    print(img_tensor.shape)
