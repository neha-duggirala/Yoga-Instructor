from tqdm import tqdm
import numpy as np
import cv2
import os
from random import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def augmentation(source,labels,image_size):
    datagen = ImageDataGenerator(
            # rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.5,
            horizontal_flip=True,
            # fill_mode='nearest',
            # brightness_range=(0.0,1.5),
            # rescale=1.0/255.0
            # vertical_flip=True
            )
    for label in labels:
        src=source + label+"/Train/"
        print("Image augmentation of Label-{} at source-{}".format(label,src))
        for img in tqdm(os.listdir(src)):
            try:
                # image = load_img(src+img,0)  # this is a PIL image
                image=cv2.imread(src+img,0)
                # image=cv2.Canny(image,50,150)
                image=cv2.resize(image,(image_size,image_size))
                img_arr = np.expand_dims(img_to_array(image), axis=0)
                img_arr = img_arr.reshape((1,) + (image_size,image_size,1))
                i = 0

                for batch in datagen.flow(
                    img_arr,
                    batch_size=64,
                    save_to_dir=src,
                    save_prefix=label,
                    save_format='jpg'):
                    if i >=1:
                        break
                    i += 1

                # print(batch.next().shape,batch.next().max,end="---")
            except:
                print("***********ERROR in",img)


#train and test split
#source should have a "/" at the end
def split(source,labels,image_size):
    for label in labels:
        src=source+label+"/"
        total=len(os.listdir(src))
        data=os.listdir(src)
        shuffle(data)
        print("Number of",label,"are",total)
        train_size,test_size=total*0.65,total*0.35
        os.makedirs(src+"Test",exist_ok=True)
        os.makedirs(src+"Train",exist_ok=True)
        train_data=data[:int(train_size)]
        test_data=data[int(train_size):]
        count=0
        for img_name in train_data:
            try:
                img=cv2.imread(src+img_name,0)
                # img=cv2.resize(img,(image_size,image_size))
                count+=1
                name=src+"Train/"+str(count)+".jpg"
                cv2.imwrite(name,img)
            except:
                # print("ERROR in train",img_name,"in",label)
                print("ERROR")
        count=0
        for img_name in test_data:
            try:
                img=cv2.imread(src+img_name,0)
                count+=1
                name=src+"Test/"+str(count)+".jpg"
                cv2.imwrite(name,img)
            except:
                print("ERROR in test",img_name,"in ",label)
def label_maker(labels,label):
    return (labels.index(label))

def create_data(src,labels,image_size):
    for trainTest in ["Test","Train"]:
        data=[]
        for label in labels:
            dir_path=src+label+"/"+trainTest
            print(dir_path)
            for img_name in (os.listdir(dir_path)):
                try:
                    image_path=os.path.join(dir_path,img_name)
                    img=cv2.imread(image_path,0)
                    img=cv2.resize(img,(image_size,image_size))
                    # if len(img)==image_size**2:
                        # img=[img,img,img]

                    data.append([np.array(img),label_maker(labels,label)])
                    # break
                except:
                        print("ERROR in",img_name)
        shuffle(data)
        # data=np.asarray(data)
        # print((data).shape,(data[0]).shape,(data[0][0]).shape)
        data_name="{}-{}-{}_data.npy".format(data[0][0].shape,trainTest,len(data))
        print(data_name)
        np.save("numpy_data/"+data_name,data)
        print("Numpy data saved")


def create_dataALL(src,labels,image_size):
    for trainTest in ["Test","Train"]:
        data=[]
        for label in labels:
            dir_path=src+label+"/"+trainTest
            print(dir_path)
            for img in (os.listdir(dir_path)):
                    # print(img)
                try:
                    image_path=os.path.join(dir_path,img)
                    img=cv2.imread(image_path,0)
                    if len(img)==image_size**2:
                        img=cv2.resize(img,(image_size,image_size,1))
                    data.append([img,label_maker(labels,label)])
                    # print(type(data),type(data[0]),type(data[0][0]))
                except:
                        # print("ERROR in",label)
                        pass
        data=np.asarray(data)
        print((data).shape,(data[0]).shape,(data[0][0]).shape)
        # shuffle(data)
        data_name="{}-{}-{}_data.npy".format(data[0][0].shape,trainTest,len(data))
        try:
            print(data_name)
            # np.save("numpy_data/"+data_name,data)
            print("Numpy data saved")
        except:
            size=len(data)
            parts=3
            print("MEMORY ERROR\nDivinding",len(data),"into",parts,"parts")
            start=0
            for i in range(parts):
                end = start+(size//parts)
                new_data=data[start:end]
                data_name="{}-{}-{}_data{}.npy".format(data[0][0].shape,trainTest,len(new_data),i)
                print(data_name,start,end)
                start = end
                # np.save("numpy_data/"+data_name,new_data)
                print("Numpy data saved")
def convertRGB(numpy_data):
    data = []
    for gray in numpy_data:
        try:
          colored= cv2.cvtColor(gray[0],cv2.COLOR_GRAY2RGB)
          data.append([colored,gray[1]])
        except:
          print(gray[0].shape)
    data = np.asarray(data)
    return data


def XYsplit(data):
    X= np.asarray([images[0] for images in data])
    y= np.asarray([images[1] for images in data])

    return X,y

def numpy_attach(src,parts):
    name = src+"(90, 90)-Train-2397_data"
    data = np.load(name+"0.npy",allow_pickle=True)
    for i in range(1,parts):
        n=name+str(i)+".npy"
        new_data = np.load(n,allow_pickle=True)
        print(n,len(data),(new_data).shape)
        data = np.concatenate((data,new_data))
    return data
def main():
    src="data5/"
    labels=os.listdir(src)
    image_size = 300
    print("The classes are -->",labels)
    # labels=["Pose 1","Pose 3","Pose 4","Pose 5","Pose 6"]
    # augmentation(src,labels,image_size)
    # split(src,labels,image_size)
    create_data(src,labels,image_size)
    # data = numpy_attach("numpy_data/",5)
    # print(data.shape)
main()
