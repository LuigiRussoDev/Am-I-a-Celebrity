import itertools
from itertools import cycle
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import interp
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import backend as K
import imageio
import cv2
from tqdm import tqdm
import skimage
from skimage.transform import resize

def get_data(folder):
  X = []
  y = []

  for folderName in os.listdir(folder):
    if not folderName.startswith('.'):
      if folderName in ['ArianaGrande']:
        label = 0
      if folderName in ['BillGates']:
        label = 1
      if folderName in ['DonaldTrump']:
        label = 2
      if folderName in ['EmmaStone']:
        label = 3
      if folderName in ['SelenaGomez']:
        label = 4
      if folderName in ['TaylorSwift']:
        label = 5
      if folderName in ['LuigiRusso']:
        label = 6

      for image_filename in tqdm(os.listdir(folder +"/" + folderName)):
        #print("Nome Immagine: ",image_filename)
        img_file = cv2.imread(folder + '/' + folderName + '/' + image_filename)
        #img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
        #img_file = imageio.imread(str(folder + '/' + folderName + '/' + image_filename))
        if img_file is not None:
          img_size=(128,128,3)
          img_file = skimage.transform.resize(img_file, img_size)
          '''plt.imshow(img_file)
          plt.show()'''
          img_arr = np.asarray(img_file)
          X.append(img_arr)
          y.append(label)

  X = np.asarray(X)
  y = np.asarray(y)
  return X, y

def roc_each_classes(test_y, y_pred,num_classes):
  n_classes = num_classes
  # Plot linewidth.
  lw = 2
  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_pred.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  # Compute macro-average ROC curve and ROC area

  # First aggregate all false positive rates
  all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

  # Then interpolate all ROC curves at this points
  mean_tpr = np.zeros_like(all_fpr)
  for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

  # Finally average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = all_fpr
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  # Plot all ROC curves
  plt.figure(7)
  plt.plot(fpr["micro"], tpr["micro"],
           label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
           color='deeppink', linestyle=':', linewidth=4)

  plt.plot(fpr["macro"], tpr["macro"],
           label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
           color='navy', linestyle=':', linewidth=4)

  colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
  for i, color in zip(range(n_classes), colors):

    if i == 0:
      cl = 'ArianaGrande'
    if i == 1:
      cl = 'BillGates'
    if i == 2:
      cl = 'DonaldTrump'
    if i == 3:
      cl = 'EmmaStone'
    if i == 4:
      cl = 'SelenaGomez'
    if i == 5:
      cl = 'TaylorSwift'
    if i == 6:
      cl = 'LuigiRusso'

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f}) %s'
                   ''.format(i, roc_auc[i]) % cl)

  plt.plot([0, 1], [0, 1], 'k--', lw=lw)
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.savefig('Roc_each_classes.jpg')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  plt.figure(10)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  y = np.repeat(np.arange(0, 7), 15)
  plt.xlim(-0.5, len(np.unique(y)) - 0.5)  # ADD THIS LINE
  plt.ylim(len(np.unique(y)) - 0.5, -0.5)  # ADD THIS LINE
  plt.savefig("confusion_matrix_big.png")
