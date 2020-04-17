from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
#from debug import *
import cv2
import numpy as np
def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image
class CASIA_Dataset(Dataset):

    """
    load the data of CASIA
    """
    def __init__(self,path,img_size=416,augment=False):
        """

        :param path: the path of the file containing the paths of the images
        :param img_size: the size of image
        """
        with open(path, "r") as file:
            self.files = file.readlines()
        self.img_size=img_size
        self.augment = augment
        self.classes=[]
        self.batch_count = 0
        #self.words=["扼","遏","鄂","饿","恩","而","儿","耳","尔","饵"]
    def get_index(self,word):
        for i,w in enumerate(self.words):
            if w == word:
                return i

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        #print("11111111111")
        img_path = self.files[index % len(self.files)].rstrip()[:-1]
        #print(img_path)
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # ---------
        #  Label
        # ---------

        label=self.files[index % len(self.files)].replace(img_path,'')
        if label =="a\n":
            label=10
        elif label=="b\n":
            label=11
        elif label=="c\n":
            label=12
        elif label=="d\n":
            label=13
        elif label=="e\n":
            label=14
        else:
            label=int(label)

        """
        word = self.files[index % len(self.files)].replace(img_path,'').rstrip()
        if word not in self.classes:
            self.classes.append(word)
        """
        #label=self.get_index(word)


        """
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
        """
        return img_path, img, label

    def collate_fn(self, batch):
        paths, imgs, labels = list(zip(*batch))
        # Remove empty placeholder targets
        #labels = [boxes for boxes in labels if boxes is not None]
        # Add sample index to targets
        #for i, boxes in enumerate(targets):
            #boxes[:, 0] = i
        #targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        #if self.multiscale and self.batch_count % 10 == 0:
            #self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        labels=torch.tensor(labels)
        self.batch_count += 1
        return paths, imgs, labels

    def __len__(self):
        return len(self.files)
if __name__=="__main__":
    print(len(CASIA_Dataset("../train.txt")))
    d = CASIA_Dataset("../train.txt")
    for ii,e,r, in d:
        print(ii)