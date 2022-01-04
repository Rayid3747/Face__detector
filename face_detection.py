import cv2

face_data=cv2.CascadeClassifier('D:\\CODE\PROJECTS\\PYTHON Projects\\Face Detector.py\\haarcascade_frontalface_default.xml')

# img=cv2.imread('D:\\Face detection\\OIP.jpeg')
# cv2.waitKey()
img=cv2.VideoCapture(0)

while True:

 frames, frame=img.read()

 grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


 face_codinates=face_data.detectMultiScale(grayscaled_img)

 for(x,y,w,h) in face_codinates:

  cv2.rectangle(frame, (x,y), (x+w, y+h),(0,150,0),3)
 cv2.imshow("grayscaled_img",frame)
 cv2.waitKey(1)
