import glob  
import cv2  
import numpy as np  
import os  

def main():  
    # 图像路径  
    path = 'data/isic2017/train' 
    # 存放训练所用的 npz 文件的路径  
    path2 = 'data/ISIC2017/train_npz'
    # 如果输出路径不存在则创建  
    os.makedirs(path2, exist_ok=True)  
    
    for i, img_path in enumerate(glob.glob(path)):  
        # 读取图像  
        image = cv2.imread(img_path)  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        # 读取标签  
        label_path = img_path.replace('images', 'labels')  
        label = cv2.imread(label_path, flags=0)  
        # 保存 npz  
        np.savez(os.path.join(path2, f'{i}.npz'), image=image, label=label)  
        print('------------', i)  
    
    print('All images processed and npz files saved.')  

if __name__ == "__main__":  
    main()