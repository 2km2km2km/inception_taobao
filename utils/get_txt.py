import os
import tqdm

def get_txt_CASIA(path,max_class_num=10,train_jpg_num=300):

    #get the train_txt and the test_txt
    #书写CASIA的train_dataset.txt和test_dataset.txt
    #path为CASIA数据集路径
    #max_class_num=最多种类的数量
    #the max number of the classes
    class_max_num=10

    test_path = path+"1.0test-gb1/"
    train_path = path+"1.0train-gb1/"

    train_txt = open('./train.txt',mode='w')
    test_txt = open('./test.txt', mode='w')

    #读取数据库中已有数据
    train_img_paths = os.walk(train_path)
    test_img_paths = os.walk(test_path)
    index=0
    train_img_paths=list(train_img_paths)
    test_img_paths = list(test_img_paths)

    # 训练集数据
    for l in tqdm.tqdm(train_img_paths[1:]):
        index+=1
        label = l[0].replace(train_path, "")
        for jpg_num,img_path in enumerate(l[2]):
            if jpg_num>=train_jpg_num:
                break
            train_txt.write(l[0]+"/"+img_path+label)
            train_txt.write("\n")

        # 测试集数据
        for j in test_img_paths[1:]:
            keyj=j[0].replace(test_path,"")
            if keyj==label:
                for img_path in j[2]:
                    test_txt.write(j[0] +"/"+ img_path+label)
                    test_txt.write("\n")
        if index>=max_class_num:
           break
    print("done")
def get_txt_taobao(path):
    #书写train_dataset.txt和test_dataset.txt
    #同时起到将数据集划分成训练集和测试集的功能，目前采取的方式是按顺序划分，后续可以优化
    Max_size = 200
    labels = [2,3,4]
    train_txt = open('./train.txt',mode='w')
    test_txt = open('./test.txt', mode='w')
    #trainval_txt = open('./dataset/trainval.txt', mode='w')
    #val_txt = open('./dataset/2007_val.txt', mode='w')
    #读取数据库中已有数据
    #ltxts = os.walk("dataset/labels")
    dirs = os.walk(path)
    #print(dirs)
    for d in dirs:
        #print(d[0])
        imges = d[2]
        data_size = len(imges)
        label=d[0].replace(path,"").replace("/","")
        #训练集数据
        for i in range(int(0.7*data_size)):
            img_path=imges[i]
            train_txt.write(d[0]+"/"+img_path+label)
            #trainval_txt.write()
            #print( txtes[i][:-4])
            #if i != int(0.7*data_size):
            if 1:
                train_txt.write("\n")
                #trainval_txt.write("\n")
        #测试集数据
        for i in range(int(0.7*data_size),int(data_size)):
            img_path = imges[i]
            test_txt.write(d[0]+"/"+img_path+label)
            #print(txtes[i][:-4])
            #if i != int(data_size):
            if 1:
                test_txt.write("\n")
        """
        #验证集
        for i in range(int(0.8*data_size),data_size):
            val_txt.write(f"/mnt/darknet/darknet/scripts/VOCdevkit/VOC2007/JPEGImages/{txtes[i][:-4]}.jpg")
            trainval_txt.write(f"/mnt/darknet/darknet/scripts/VOCdevkit/VOC2007/JPEGImages/{txtes[i][:-4]}.jpg")
            if i != int(data_size):
                val_txt.write("\n")
                trainval_txt.write("\n")
        """
#get_txt_CASIA("/media/xzl/Newsmy/数据集/CASIA-HWDB/Character Sample Data/1.0/",10,1000)
#get_txt_taobao("/media/xzl/Newsmy/德显/数据集/data")