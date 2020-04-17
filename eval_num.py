import train
import utils.get_txt
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
nums=[5,10,50,100,200,300,400]

epochs=50
epoch=range(1,epochs+1)
for num in nums:
    utils.get_txt.get_txt_CASIA("/media/xzl/Newsmy/数据集/CASIA-HWDB/Character Sample Data/1.0/",10,num)
    precisions=train.train(epochs)
    plt.plot(epoch,precisions,label=str(num))

    #plt.plot([1,2,3,4])
    plt.xlabel("epoch")
    plt.ylabel("pre")

    plt.title(str(num))
    plt.legend()
    savefig("./eval_num/"+str(num)+".jpg")