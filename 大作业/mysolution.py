from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def model_metrics(clf, X_train, X_test, y_train, y_test):
    # 预测
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # 准确率
    print('[准确率]', end=' ')
    print('训练集：', '%.4f' % accuracy_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % accuracy_score(y_test, y_test_pred))

    # 精准率
    print('[精准率]', end=' ')
    print('训练集：', '%.4f' % precision_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % precision_score(y_test, y_test_pred))

    # 召回率
    print('[召回率]', end=' ')
    print('训练集：', '%.4f' % recall_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % recall_score(y_test, y_test_pred))

    # f1-score
    print('[f1-score]', end=' ')
    print('训练集：', '%.4f' % f1_score(y_train, y_train_pred), end=' ')
    print('测试集：', '%.4f' % f1_score(y_test, y_test_pred))

    # auc取值：用roc_auc_score或auc
    print('[auc值]', end=' ')
    print('训练集：', '%.4f' % roc_auc_score(y_train, y_train_proba), end=' ')
    print('测试集：', '%.4f' % roc_auc_score(y_test, y_test_proba))

    # roc曲线
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba, pos_label=1)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label=1)

    label = ["Train - AUC:{:.4f}".format(auc(fpr_train, tpr_train)),
             "Test - AUC:{:.4f}".format(auc(fpr_test, tpr_test))]
    plt.plot(fpr_train, tpr_train)
    plt.plot(fpr_test, tpr_test)
    plt.plot([0, 1], [0, 1], 'd--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(label, loc=4)
    plt.title("ROC curve")
    plt.show()


def readCSV(filename):
    """ 读取数据"""
    dataset = pd.read_csv(filename, encoding='gbk')

    """ 数据处理"""
    # 删除固定信息列
    dataset = dataset.drop(["user_id"], axis=1)

    dataset = dataset.fillna(0)  # 使用 0 替换所有 NaN 的值
    col = dataset.columns.tolist()[1:]

    def missing(df, columns):
        """
        使用众数填充缺失值
        df[i].mode()[0] 获取众数第一个值
        """
        col = columns
        for i in col:
            df[i].fillna(df[i].mode()[0], inplace=True)
            df[i] = df[i].astype('float')

    missing(dataset, col)

    # 将object类型转成folat
    dataset = dataset.convert_objects(convert_numeric=True)
    return dataset

def preTrainHandle(dataset):

    """ 数据划分"""
    X = dataset.drop(["y"], axis=1)
    Y = dataset["y"]

    # 数据按正常的3、7划分
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=666)
    # not enough values to unpack (expected 4, got 2)

    from sklearn.preprocessing import minmax_scale  # minmax_scale归一化，缩放到0-1
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    # Input contains NaN, infinity or a value too large for dtype('float64').

    """ 数据归一化"""
    from sklearn.preprocessing import minmax_scale
    # 归一化，缩放到0-1
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    return X_train,X_test,y_train,y_test


def train(X_train,X_test,y_train,y_test):
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=20,
        max_depth=4,
        min_child_weight=1,
        gamma=1,
        subsample=0.6,
        colsample_bytree=0.6,

        objective='binary:logistic',scale_pos_weight=1,
        nthread=4,
        max_delta_step=10,
        #scale_pos_weight=1,
        seed=27, cv=3,
        reg_alpha=0.01,
        eval_metric="error")

    print(X_train.shape)
    #print("修改前")
    # print(y_train.shape[1])
    # xgb1.set_params(params)
    xgb.fit(X_train, y_train)
    model_metrics(xgb, X_train, X_test, y_train, y_test)
    return xgb

def dropFeathers(dataset,model):
    col = dataset.columns.tolist()[1:]
    losefeather = []
    newfeatherscore = []
    for i in range(len(col)):
        #print(col[i] + "  " + str(model.feature_importances_[i]))
        if model.feature_importances_[i] < 0.003 :#and model.feature_importances_[i]>0.05:
            losefeather.append(col[i])

    dataset = dataset.drop(losefeather, axis=1)
    return dataset,losefeather

def predict(model,dropfeathers):

    """ 读取数据"""
    ttest = pd.read_csv('./test.csv', encoding='gbk')
    user_id = ttest['user_id']
    print()
    """ 数据处理"""
    # 删除固定信息列
    ttest = ttest.drop(["user_id"], axis=1)
    ttest = ttest.fillna(0)  # 使用 0 替换所有 NaN 的值
    col = ttest.columns.tolist()[0:]

    def missing(df, columns):
        """
        使用众数填充缺失值
        df[i].mode()[0] 获取众数第一个值
        """
        col = columns
        for i in col:
            df[i].fillna(df[i].mode()[0], inplace=True)
            df[i] = df[i].astype('float')

    missing(ttest, col)
    # print(losefeather)
    # dataset = dataset.drop(losefeather,axis = 1)
    # 将object类型转成folat
    #print("col" + str(col))
    #print(len(col))
    ttest = ttest.convert_objects(convert_numeric=True)

    ttest = ttest.drop(dropfeathers, axis=1)
    from sklearn.preprocessing import minmax_scale  # minmax_scale归一化，缩放到0-1
    ttest = minmax_scale(ttest)

    """ 数据归一化"""
    from sklearn.preprocessing import minmax_scale
    # 归一化，缩放到0-1
    ttest = minmax_scale(ttest)
    ans = model.predict(ttest)
    #print(ans)
    import csv

    writerr = []
    for i in range(len(ans)):
        eachh = []
        eachh.append(ans[i])
        writerr.append(eachh)

    # 输出到ans.csv
    dataframe = pd.DataFrame({'user_id': user_id, '预测结果': ans})  # 'a_name':ttest["user_id"],
    dataframe.to_csv("ans.csv", index=False, sep=',')

if __name__ == '__main__':
    dataset = readCSV('./model.csv')
    X_train, X_test, y_train, y_test = preTrainHandle(dataset)
    print("筛选前")
    model = train(X_train, X_test, y_train, y_test)
    dataset, dropfeathers = dropFeathers(dataset, model)
    X_train, X_test, y_train, y_test = preTrainHandle(dataset)
    print("筛选后")
    model = train(X_train, X_test, y_train, y_test)
    # 根据训练的模型，预测test.csv中的元组，并写入ans.csv
    predict(model, dropfeathers)

