import os
from data_process import copy_dataset,move_corrupted_images
from train import train_model

def main():
    print("==================================================")
    print("        🌸 欢迎使用花卉图片识别系统 🌸        ")
    print("==================================================")

    BASE_DIR=os.path.dirname(os.path.abspath(__file__))
    DATA_DIR=os.path.join(os.path.dirname(BASE_DIR),'data')
    MODEL_PATH=os.path.join(BASE_DIR,'best_model.pth')

    RAW_DATA=os.path.join(DATA_DIR,'flower')
    COPY_DATA=os.path.join(DATA_DIR,'flower_copy')
    CORRUPTED_DIR=os.path.join(DATA_DIR,'corrupted_images')
    
    print("\n[第一阶段] 进入数据预处理")
    copy_dataset(RAW_DATA,COPY_DATA)
    move_corrupted_images(COPY_DATA,CORRUPTED_DIR)
    if os.path.exists(MODEL_PATH):
        print("\n[第二阶段] 发现已存在训练好的模型 best_model.pth,跳过训练！")
        print("如果你想重新训练，请手动删除 src/best_model.pth 文件。")
    else:
        print("\n[第二阶段] 未找到模型,开始模型训练与测试评估")
        train_model()
    print("\n🎉 所有流程执行完毕！请查阅 docs 目录获取图表与笔记。")

if __name__=="__main__":
    main()