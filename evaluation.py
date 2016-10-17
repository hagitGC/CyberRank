import numpy as np
from sklearn import svm, linear_model, cross_validation
import pandas as pd
from sklearn.cross_validation import train_test_split
import random
from collections import defaultdict

AHPwights= [0.030420825, 0.133504122, 0.01506668, 0.103644788, 0.106375118, 0.032145798, 0.059287755, 0.061884989, 0.201151162, 0.190558878, 0.030574248, 0.035385638]

## reading the users data
## the each pair of exaples the users had to rank is represented by one row and direction (Y)
Xp = []
for line in open("Xp.csv"):
    data = [float(x) for x in line.rstrip().split(",")]
    Xp.append(data)
Y = [float(line.rstrip()) for line in open("Y.csv")]

synthXp = []
for line in open("synthXp.csv"):
    data = [float(x) for x in line.rstrip().split(",")]
    synthXp.append(data)

synth_yp = [float(line.rstrip()) for line in open("synth_yp.csv")]



#*************************************************************************************#
#                                                                                     #
#    After building the X and y vectors we can use cross validation    #
#    and create a test and train sets                                                 #
#                                                                                     #
#*************************************************************************************#
X_train, X_test, y_train, y_test = train_test_split(Xp, Y, test_size=0.3, random_state=42)
hits, mis, acc = {}, {}, {}
hitsAHP =0
hitsRSVM =0
mytable = defaultdict(list)
"""
The idea of CyberRank is to create synthetic data for training a supervised model (Ranking SVM here)
Here we test the different approaches using some real data collected with experts (171 pairs of transactions ranked as 1/-1)
The data is already in the format of x1-x2 (rank svm)

Vanilla - Simply ranking svm using only the annotated data (10-80 data points, under 10 points this will not converge)
rank_svm_plus_AHP - CyberRank: train with synthetic data and annotated data, without over sampling of real data
rank_svm_plus_AHP_balanced - CyberRank: train with synthetic data and annotated data, with over sampling of real data
see the paper.


"""
for model_type in ["vanilla","rank_svm_plus_AHP","rank_svm_plus_AHP_balanced"]:
    for train_size in range(10, 81, 5):   #range (0, len(Y)):
        for time in range(0, 30):
            random.seed(time)
            index_list = random.sample(range (0, len(y_train)), train_size)
            X_train_kova, y_train_kova = [], []
            for i in index_list:
                X_train_kova.append(X_train[i])
                y_train_kova.append(y_train[i])

            #duplicate samples for rank_svm_plus_AHP_balanced
            if model_type == "rank_svm_plus_AHP_balanced":
                temp_kova = []
                temp_y = []
                while len(temp_kova) < len(synthXp):
                   temp_kova.extend(X_train_kova)
                   temp_y.extend(y_train_kova)
                X_train_kova = temp_kova
                y_train_kova = temp_y

            #add the synthetic data
            if model_type in ["rank_svm_plus_AHP","rank_svm_plus_AHP_balanced"]:
                X_train_kova.extend(synthXp)
                y_train_kova.extend(synth_yp)

                #print y_train_kova
            #self learn from ranking:
            #if ((model_type in ["rank_svm_plus_AHP","rank_svm_plus_AHP_balanced"]): or  (train_size >= 10)):
            clf = svm.SVC(kernel='linear', C=.3)
            clf.fit(X_train_kova, y_train_kova)

            coef = clf.coef_.ravel() / np.linalg.norm(clf.coef_)
            ncoef = np.dot(coef, 1/sum(coef))

            CulcY, sighCulcY, YpAHP, YpRSVM, tmpRSVM, tmpAHP = [], [], [], [], [], []
            for arr in X_test:
                CulcY.append(sum(arr * ncoef))
                sighCulcY.append(np.sign(CulcY[-1]))

            gap = (y_test - np.array(sighCulcY))
            hits[time] = len(gap[gap == 0])
            mis[time] =  len(gap[gap <> 0])
            acc[time] = float(hits[time])/float(len(y_test))

        a = list(acc.values())
        print model_type,train_size,'mean accuracy:', np.mean(a), np.std(a)
        mytable[model_type].append((np.mean(a), np.std(a)))

## applaying pure AHP over test data set:
#h = np.sign(np.dot(X_test, AHPwights))
d = y_test - np.sign(np.dot(X_test, AHPwights))
print 'Pure AHP scoore:' , float(len(d[d == 0]))/float(len(d))