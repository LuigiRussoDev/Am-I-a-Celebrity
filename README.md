# Am-I-a-celebrity-
The ironic title is an excuse to introduce a problem and show the results.
As can be seen from the title, in this problem, we want to address a very important target in the field of computer vision and artificial intelligence: Visual Recognition.
The main idea is based on a set of data (people's faces) and recognize their face.
The resolution of the task, features a set of Celebrity images data set (The best way to obtain images easily available from the internet).
The dataset was obtained through youtube videos for each celebrity. The celebrities in question are 6: ArianaGrande, Bill Gates, Donald Trump, Selena Gomez, Taylor Swift, Emma Stome and Luigi Russo (myself).

## Preprocessing
The preprocessing phase was very important in order to obtain a valid image dataset.
The images were obtained from videos taken from youtube, where each video was represented by a celebrity. Generally the videos taken referred to interviews in order to give more emphasis to the structure of the video.
For each video, the following steps were performed:
1) Each Video: TaylorSwift.mp4 consisting of 10/15 min has been converted into frames of images, so as to obtain the i-th folder represented only by Taylor images. And so on..
2) After obtaining the 6 classes, where each class is represented by N images, a boundingbox algorithm has been applied to each folder.
The BoundingBox algorithm is like a function where given an image with a face, it must output only the face.
The bounding box algorithm was performed for each folder in order to obtain another folder with all the extractions of the faces.

## Neural Network Model

The neural network model I wanted to use is an Odenet Neural Network. For this Task, however, I also wanted to use a small block of resnet before the ReteNeurale Odenet.
The Odenet neural network, recently in the field of artificial intelligence and in particular as features extraction, has obtained excellent results expressed in terms of validation accuracy and training accuracy.

## Dataset

As mentioned above, the image dataset was obtained from videos taken from Youtube for each celebrity.
The training was therefore performed on data sets represented by images and the latter were divided into 70% training images and 30% test images.
The training images are 2940 and the test images are 1260.

Frames             |  Face 
:-------------------------:|:-------------------------:
 <img src="https://github.com/LuigiRussoDev/Am-I-a-Celebrity/blob/main/imgs/BillGates3S456.jpg"  width="120" height="120">  |   <img src="https://github.com/LuigiRussoDev/Am-I-a-Celebrity/blob/main/imgs/img636.jpg"  width="120" height="120"> 