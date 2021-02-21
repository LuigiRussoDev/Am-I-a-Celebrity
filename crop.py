import glob
import cv2


def crop_faces(path, scale):
    img_list = glob.glob(path + '/*.jpg')
    haar_face_cascade = cv2.CascadeClassifier('C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')

    count = 0
    for img_name in img_list:
        img = cv2.imread(img_name)
        faces = haar_face_cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=5)


        for (x, y, w, h) in faces:
            face_cropped = img[y:y + h, x:x + w]
            face_resized_img = cv2.resize(img[y:y + h, x:x + w], (256, 256), interpolation=cv2.INTER_AREA)
            count = count + 1

            new_img_name = img_name.replace('.jpg', '')
            new_img_name = 'DaYoutube2/face_crop/LuigiRusso/' + 'img_luigiRusso%d.jpg' % count

            cv2.imwrite(new_img_name,face_resized_img)

'''
Let's to define the crop_faces function. It gives in input the relative path of folder frames. 
'''
crop_faces('DaYoutube2/LuigiRusso',1.2)
