from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import myfe
import mne
import scipy
import numpy as np
import os
import fnmatch
import re
import sys


def main():
    mne.set_log_level(verbose='WARNING')
    if (re.match('win', sys.platform) != None):
        folder = r'D:\IDM\下载\抑郁症数据集\MODMA数据集\EEG_3channels_resting_lanzhou_2015'
    elif (re.match('linux', sys.platform) != None):
        folder = '/home/lyb/data/MODMA_data/EEG_3channels_resting_lanzhou_2015'
    files = []
    labels = scipy.io.loadmat(os.path.join(folder, 'y_type.mat'))
    labels = labels['y_type']

    flag = True
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, '*_still.mat'):
            files.append(file)
            data = scipy.io.loadmat(os.path.join(folder, file))
            data = data['data'] / 1000000
            data = data.T
            newraw = myfe.filter_eeg(data, 250, 0.5, 100, 50)
            PSE = myfe.pse(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
            PSE = PSE.reshape(1, PSE.shape[0] * PSE.shape[1])
            if flag:
                flag = False
                features = PSE
                continue
            features = np.vstack([features, PSE])

    param_grid = {'C': [0.01, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  # 'degree': [0, 1, 2, 3],
                  'kernel': ['rbf']}

    model = svm.SVC()
    创建一个GridSearchCV对象
    grid = GridSearchCV(model, param_grid, refit=True, n_jobs=16, verbose=3)
    使用训练集的特征和标签训练分类器
    grid.fit(features, labels.squeeze())
    print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
    y_pred = grid.predict(x_test)

    # model = svm.SVC(C=100, gamma=0.001, kernel='rbf')
    # x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    # model.fit(x_train, y_train.squeeze())
    # # 使用测试集的特征进行预测
    # y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    main()
