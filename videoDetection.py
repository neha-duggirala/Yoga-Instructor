from tensorflow.keras.models import load_model
import numpy as np
import cv2,os
import tensorflow as tf

def TFLite_video(model_path,poses,src=0,destination="Videos"):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    cap = cv2.VideoCapture(src)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter(destination+'/trail1/day2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(gray,tuple(input_shape[1:-1]))
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # canny=cv2.Canny(gray,50,150)
        img=np.reshape(img,input_shape)
        img=img.astype("float32")
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pose = np.argmax(output_data)+1
        cv2.putText(frame,str(pose)+poses[pose],(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        print(pose)
        cv2.imshow('Prediction',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def TFLite_Image(model_path,poses,src):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    for img_name in os.listdir(src):
        image=cv2.imread(src+img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=cv2.resize(image,(700,500))
        img=cv2.resize(gray,tuple(input_shape[1:-1]))
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img=np.reshape(img,input_shape)
        img=img.astype("float32")
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pose = np.argmax(output_data)+1
        # cv2.putText(frame,str(pose)+poses[pose],(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        cv2.putText(image,str(pose)+poses[pose]+img_name,(40,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow(img_name,image)
        cv2.waitKey()
    cv2.destroyAllWindows()


def image(model,src,poses):
    for img_name in os.listdir(src):
        # try:
            image=cv2.imread(src+img_name)
            gray=cv2.resize(image,(700,500))
            # print(image.shape)
            img=cv2.resize(image,(300,300))
            # canny=cv2.Canny(img,50,150)
            img=img.reshape((300,300,3))
            # print(img.shape)
            prediction = model.predict( np.array( [img,] ))
            pose = np.argmax(prediction)+1
            cv2.putText(gray,str(pose)+poses[pose]+img_name,(40,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow(img_name,gray)
            cv2.waitKey()
        # except:
            # print("!!!!!ERROR in {}!!!!!Value-{}".format(img_name,image))
    cv2.destroyAllWindows()
def video(model,poses,src=0,destination="Videos"):
    cap = cv2.VideoCapture(src)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(destination+'/trail1/day2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(gray,(300,300))
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # canny=cv2.Canny(gray,50,150)
        img=img.reshape((300,300,3))
        prediction = model.predict( np.array( [img,] ))
        pose = np.argmax(prediction)+1
        cv2.putText(frame,str(pose)+poses[pose],(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        print(pose,prediction)
        cv2.imshow('Prediction',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(frame)
    cap.release()
    cv2.destroyAllWindows()

def demo(model,poses):
    pose = 0
    for i in range(1,9):
        #it hsould be like 2 threads running parallelly
        # Webcam = cv2.VideoCapture(1)
        # ret, Webframe = Webcam.read()
        # cv2.imshow("Webcam",Webframe)
        cap = cv2.VideoCapture("test/demo yoga.mp4")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        black = np.zeros((frame_width,frame_height))
        while pose!=i:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img=cv2.resize(gray,(300,300))
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            img=img.reshape((300,300,3))
            pose = np.argmax(model.predict( np.array( [img,] )))+1
        while True:
            cv2.putText(frame,str(pose)+poses[pose],(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
            cv2.imshow("Users Face",frame)
            instructions = "Put namaskar and chant OM \n for 3 seconds and when u are ready \n press Q key"
            cv2.putText(black,instructions,(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
            cv2.imshow(str(pose),black)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



'''
    prev=0
    for pose in range(1,10):
            black = np.zeros((300,300,3))
            if pose == prev+1:
                text = "Pose "+str(pose)
                prev = pose
            else:
                text = "Wrong position Please try again"
            print(prev,text)
            cv2.putText(black,text,(40,50),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 5)
            cv2.imshow(str(pose),black)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
'''

src="test/Demo/Final/"
model_path="models/Trail1/VGG16-(300,300,3)D.tflite"
# model_path = "models/Trail2/mobileNet-(224,224,3)H.h5"
# model=load_model(model_path)

# model_path = "models/Trail5/mobileNet-(300,300,3)A.tflite"

# model = None
poses = ["None",".Pranamasana",".Hastauttanasana",".Hasta Padasana",".Ashwa Sanchalanasana",".Dandasana",".Ashtanga Namaskara",".Bhujangasana",".Parvatasana"]
# image(model,src,poses)
TFLite_Image(model_path,poses,src)
# src = "test/demo yoga.mp4"
# TFLite_video(model_path,poses)
# video(model,poses)
# demo(model,poses)
