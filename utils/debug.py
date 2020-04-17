import torchvision.transforms as transforms
import cv2
import numpy as np
"""
some function used to debug
"""
def show_labels_img(img,label):
    """
    show the label and the corresponding image to check the correctness of dataset

    """
    #转换图像格式
    aug = transforms.Compose([
        transforms.ToPILImage()
    ])
    img = np.array(aug(img))
    print(label)
    cv2.imshow("img",img)
    cv2.waitKey(0)