import numpy as np,cv2
import tensorflow as tf
def video(model_path,poses,src=0,destination="Videos"):
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details,output_details)
    input_shape = input_details[0]['shape']
    cap = cv2.VideoCapture(src)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter(destination+'/trail1/day2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
'''
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
'''

poses = ["None",".Pranamasana",".Hastauttanasana",".Hasta Padasana",".Ashwa Sanchalanasana",
".Dandasana",".Ashtanga Namaskara",".Bhujangasana",".Parvatasana"]
model_path="models/yoganet.tflite"
video(model_path,poses)
