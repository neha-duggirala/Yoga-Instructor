import numpy as np,cv2,os
def image(src):
    list_images = os.listdir(src)
    list_images = list_images
    print(list_images)
    for img in list_images:
        try:
            image=cv2.imread(src+img,0)
            print(image.shape)
            cv2.imshow("Image",image)
            cv2.waitKey()
        except:
            print("!!!!!ERROR in {}!!!!!Value-{}".format(src+img,image))
    cv2.destroyAllWindows()

def rename(sr):
    dst=0
    for filename in os.listdir(sr):
         src =sr+ filename
         dst +=1
         os.rename(src, sr+str(dst)+".jpg")

# rename("Test/Demo/")
# image("test_images/")
