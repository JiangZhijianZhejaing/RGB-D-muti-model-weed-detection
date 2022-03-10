# -*- coding: utf-8 -*-
# 作者    : JZJ
# 创建时间 ：2021/11/4 20:40
# 文件    : sklearnSVM.py
# IDE    : PyCharm
# Usage  :
import pickle
import random
import joblib
import  numpy as np
import time
import  matplotlib.pyplot  as plt
import sklearn
from sklearn.svm import SVC
#导入数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    #读入数据
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(' ')
        #读入标签信息
        ls =[]
        for i in range(len(lineArr)-1):
            ls.append(float(lineArr[i]))
        dataMat.append(ls)
        if abs(float(lineArr[-1])+1)<1e-6:
            labelMat.append(-1)
        else:
            labelMat.append(1)
    #打乱顺序
    cc = list(zip(dataMat, labelMat))
    random.shuffle(cc)
    dataMat, labelMat=zip(*cc)
    dataMat,labelMat=np.array(dataMat), np.array(labelMat).reshape(-1,1)
    #对异常点进行处理,结合上下分位点进行划分
    dataMat_low=np.percentile(dataMat, 25,axis=0)
    dataMat_high=np.percentile(dataMat, 75,axis=0)
    # 结合3sigma准则筛选数据
    k = 3
    IQR =dataMat_high-dataMat_low
    dataMat_low=dataMat_low-k*IQR
    dataMat_high=dataMat_high+k*IQR
    index1=np.where(dataMat>dataMat_high)#返回tuple类型，每个元素包含对应的坐标
    index2=np.where(dataMat<dataMat_low)

    #拼接横坐标，并且去重
    aim=np.vstack([index1[0].reshape(-1,1),index2[0].reshape(-1,1)])
    aimIndex,_,_ = np.unique(aim, return_index=True, return_inverse=True)
    #删除对应横坐标元素
    for i in range(aimIndex.shape[0]):
        dataMat = np.delete(dataMat, aimIndex[i],axis=0)
        labelMat = np.delete(labelMat, aimIndex[i],axis=0)
        aimIndex-=1
    return dataMat,labelMat

#清洗数据
def preprocess(data):
    X=data
    X-=np.mean(X,axis=0)
    X/=np.std(X,axis=0,ddof=1)
    return data
    pass

if __name__ == '__main__':
    print('运行了GridSearchCV,运行时间较长请耐心等待,')
    stratTime=time.time()
    data,label=loadDataSet('DatasWithHeight.txt')
    data=preprocess(data)
    '''
        可以构建对应的转换器
        全部步骤的流式化封装和管理
        可放在Pipeline中的步骤可能有：
        特征标准化是需要的，可作为第一个环节
        中间可加上比如数据降维（PCA）
        classifier也是少不了的，自然是最后一个环节
    '''
    #数据的预处理，例如标准化，中心化，scaling，二值化等
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    scaler = StandardScaler()
    #数据标准化
    x_train = scaler.fit_transform(data)
    with open('model/DataNormalization.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    joblib.dump(scaler,'model/DataNormalization.m')
    #对应的均值和方差
    # print(scaler.mean_,scaler.var_)
    # svm_clf=Pipeline((('scaler',StandardScaler()),
    #                   ('linear_svc',LinearSVC(C=1,loss='hinge')),))
    # svm_clf.fit(data, label)
    # res = svm_clf.predict([[5.5, 1.7]])
    # print(res)
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    # from sklearn.metrics import f1_score
    model = OneVsRestClassifier(SVC(probability=True, random_state=40))
    parameters = {
        "estimator__C": [ 10,20,30,50],
        "estimator__kernel": ["poly", "rbf", "sigmoid"],
        "estimator__degree": [0.05,0.1,0.5,1, 2, 3,],
        "estimator__gamma": [0.1, 0.01, 0.001, 0.0001],
    }
    model = GridSearchCV(model, param_grid=parameters)
    model.fit(x_train[:700,:], label[:700])
    joblib.dump(model.best_estimator_, 'model/SVCmodel.m')

    # model_svm=SVC(C=1,kernel='rbf',degree=3 ,coef0=3.0)
    # model_svm.fit(x_train[:600,:],label[:600])

    print("模型的最优参数：", model.best_params_)
    print("最优模型分数：", model.best_score_)
    print("最优模型对象：", model.best_estimator_)
    # 输出网格搜索每组超参数的cv数据
    for p, s in zip(model.cv_results_['params'],
        model.cv_results_['mean_test_score']):
        print(p, s)
    s = model.best_estimator_.score(x_train[700:, :], label[700:])
    print("测试集误差为：%.2f " % (s * 100))



