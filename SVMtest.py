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
    # files = []
    labels = scipy.io.loadmat(os.path.join(folder, 'y_type.mat'))
    labels = labels['y_type'].squeeze()

    flag = True
    for file in os.listdir(folder):
        if fnmatch.fnmatch(file, '*_still.mat'):
            # files.append(file)
            feature = np.array([])
            data = scipy.io.loadmat(os.path.join(folder, file))
            data = data['data'] / 1000000
            data = data.T
            data = data[:, list(range(60 * 250))]
            newraw = myfe.filter_eeg(data, 250, 0.5, 100, 50)

            PSE = myfe.pse(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
            PSE = myfe.compress(PSE)
            PSE = PSE[[2, 4, 10, 11]]

            CF = myfe.cf(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
            CF = myfe.compress(CF)
            CF = CF[[16]]

            RCF = myfe.rcf(newraw.get_data(), newraw.info['sfreq'], newraw.info['nchan'])
            RCF = myfe.compress(RCF)
            RCF = RCF[[8]]

            feature = np.append(feature, PSE)
            feature = np.append(feature, CF)
            #feature = np.append(feature, RCF)

            if flag:
                flag = False
                features = feature
                continue
            features = np.vstack((features, feature))

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    # 利用网格搜索得出较好的超参数
    param_grid = {'C': [0.01, 1, 10, 100],
                  'gamma': [1, 0.1, 0.01, 0.001],
                  'degree': [0, 1, 2, 3],
                  'kernel': ['rbf', 'poly', 'linear']}

    model = svm.SVC()
    # 创建一个GridSearchCV对象
    grid = GridSearchCV(model, param_grid, refit=True, n_jobs=16, verbose=3)
    # 使用训练集的特征和标签训练分类器
    grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
    y_pred = grid.predict(x_test)

    # # SVM分类
    # model = svm.SVC(C=100, gamma=1, kernel='rbf')
    # model.fit(x_train, y_train)
    # # 使用测试集的特征进行预测
    # y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


if __name__ == '__main__':
    main()
