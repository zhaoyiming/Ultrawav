from __future__ import print_function
import os
import torch
import torch.nn.functional as F

# from models.resnet56_moe_debug import  resnet56, L1_loss
from config.config_origin import Config
from models.cnn import resnet20
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
model = svm.SVC(kernel='linear', C=10, gamma=0.5, decision_function_shape='ovr')
imgs = []
labels = []
for root, sub_folders, files in os.walk("data"):
    for name in files:
        imgs.append(np.loadtxt(os.path.join(root, name), delimiter=',', dtype='float'))
        labels.append(root.split("\\")[1])
imgs = np.stack(imgs, axis=0)
imgs = imgs.reshape(imgs.shape[0], -1)
print(imgs.shape)
le = preprocessing.LabelEncoder()
targets = le.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(imgs, targets, test_size=0.2, random_state=42, shuffle=True)

print(y_test)
model.fit(X_train, y_train)
acu_train = model.score(X_train, y_train)
acu_test = model.score(X_test, y_test)

print(acu_train)
print(acu_test)

