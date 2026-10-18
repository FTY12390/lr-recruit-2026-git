import os
import shutil
from PIL import Image
from torchvision import transforms

train_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_test_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def copy_dataset(src_dir,dst_dir):
    if os.path.exists(dst_dir):
        print(f"副本已存在，跳过复制：{dst_dir}")
    else:
        print(f'正在创建数据副本：{src_dir}->{dst_dir}')
        shutil.copytree(src_dir,dst_dir)
        print("副本创建完成！")

def move_corrupted_images(root_dir,corrupted_dir):
    os.makedirs(corrupted_dir,exist_ok=True)
    corrupted_count=0

    for dirpath,_,filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.jpg','.jpeg','png')):
                filepath=os.path.join(dirpath,filename)
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except (IOError,SyntaxError,OSError):
                    print(f'发现损坏图片，已删除：{filepath}')
                    shutil.move(filepath,os.path.join(corrupted_dir,filename))
                    print(f"发现损坏图片，已移动至回收站：{filename}")
                    corrupted_count+=1
    print(f'处理完成！共发现并移动了{corrupted_count}张损坏图片至{corrupted_dir}')

if __name__=='__main__':
    RAW_DATA_ROOT=r'D:\A\lr2\flower_cnn\data\flower'
    COPY_DATA_ROOT=r'D:\A\lr2\flower_cnn\data\flower_copy'
    CORRUPTED_DIR=r'D:\A\lr2\flower_cnn\data\corrupted_imageas'
    copy_dataset(RAW_DATA_ROOT,COPY_DATA_ROOT)
    move_corrupted_images(COPY_DATA_ROOT,CORRUPTED_DIR)
