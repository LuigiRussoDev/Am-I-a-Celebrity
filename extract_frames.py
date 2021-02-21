import os
import cv2

def build_dataset_image_from_video(path):
    for r, d, f1 in os.walk(path):
      i = 0
      for file in f1:
        name_file_to_save = os.path.splitext(f1[i])[0]
        name_folder = name_file_to_save

        os.mkdir("DaYoutube2/"+name_folder)
        dir_to_save = 'DaYoutube2/' + name_folder

        vidcap = cv2.VideoCapture('DaYoutube2/'+f1[i])
        i += 1

        success, image = vidcap.read()
        count = 0
        while success:

          image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
          cv2.imwrite(dir_to_save+"/"+name_file_to_save+"%d.jpg"% count,image)
          success, image = vidcap.read()
          count += 1



path_video = 'DaYoutube2/'
build_dataset_image_from_video(path_video)