import pandas as pd
import numpy as np
from sklearn import svm, model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import pickle
import os


def plot_confusion_matrix(test_y, predictions, classifier):
   cmap = plt.cm.Blues
   # Compute confusion matrix
   confusion_mat = confusion_matrix(test_y, predictions)
   print(confusion_mat)

   # Classification report
   print("Classification report:\n", classification_report(test_y, predictions))

   # Only use the labels that appear in the data
   labelled_classes = np.array(['Ads', 'Music'], dtype='<U10')
   classes = labelled_classes[unique_labels(test_y, predictions)]

   fig, ax = plt.subplots()
   im = ax.imshow(confusion_mat, interpolation='nearest', cmap=cmap)
   ax.figure.colorbar(im, ax=ax)
   # We want to show all ticks...
   ax.set(xticks=np.arange(confusion_mat.shape[1]),
          yticks=np.arange(confusion_mat.shape[0]),
          # ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes,
          title="Confusion matrix for" + str(classifier) + "classifier",
          ylabel='True label',
          xlabel='Predicted label')

   # Rotate the tick labels and set their alignment.
   plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

   # Loop over data dimensions and create text annotations.
   fmt = 'd'
   thresh = confusion_mat.max() / 2.
   for i in range(confusion_mat.shape[0]):
      for j in range(confusion_mat.shape[1]):
         ax.text(j, i, format(confusion_mat[i, j], fmt),
                 ha="center", va="center",
                 color="white" if confusion_mat[i, j] > thresh else "black")
   fig.tight_layout()
   # return ax

   np.set_printoptions(precision=2)
   plt.show()


def svm_classifier(train_x, train_y, test_x, test_y):
   classifier = svm.SVC(kernel="linear")
   classifier.fit(train_x, train_y)
   prediction = classifier.predict(test_x)
   accuracy = metrics.accuracy_score(test_y, prediction)
   print("Accuracy for the SVM classifier:", accuracy)
   plot_confusion_matrix(test_y, prediction, "SVM")


def mlp_classifier(train_x, train_y, test_x, test_y):
   classifier = MLPClassifier(hidden_layer_sizes=(100, 100))
   classifier.fit(train_x, train_y)
   prediction = classifier.predict(test_x)
   accuracy = metrics.accuracy_score(test_y, prediction)
   print("Accuracy for MLP classifier:", accuracy)
   plot_confusion_matrix(test_y, prediction, "MLP")
   return classifier


def train_model(train_x, test_x, train_y, test_y,
               classifier=mlp_classifier, file_name='mlp_classifier.pickle', save=True, path="../../models/"):

   # Training classifier
   model = classifier(train_x, train_y, test_x, test_y)
   if save:
       save_model(path, model, file_name)


def data_split(data):

   # Splitting data frame into features and target
   features = data.iloc[:, :-1]
   target = data.iloc[:, -1]
   train_x, test_x, train_y, test_y = model_selection.train_test_split(features, target, test_size=0.3)

   return train_x, test_x, train_y, test_y


def save_model(path, classifier, file_name):
   with open(os.path.join(path, file_name), 'wb') as model:
       pickle.dump(classifier, model)


def main():
   ads_df = pd.read_csv("../../data/processed/ads_wav.csv", header=None)
   music_df = pd.read_csv("../../data/processed/music_wav.csv", header=None)

   # Fixing class imbalance by sampling
   music_df = music_df.iloc[0:ads_df.shape[0], :]

   # Adding target class information for each sample
   ads_df["13"] = 0
   music_df["13"] = 1

   # Shuffling the data-frames and truncating it


   temp_music = music_df.sample(frac=1).reset_index(drop=True)
   temp_music = temp_music.truncate(before=0, after=200000)

   # Concatenating samples from each class
   result = ads_df.append(music_df)

   # Shuffling data samples
   result = result.sample(frac=1).reset_index(drop=True)
   train_x, test_x, train_y, test_y = data_split(result)

   # Function call for an MLP classifier
   train_model(train_x, test_x, train_y, test_y)

   # Function call for an SVM classifier
   # train_model(train_x, test_x, train_y, test_y, classifier=svm_classifier, file_name='svm_classifier.pickle')


if __name__ == '__main__':
   main()
