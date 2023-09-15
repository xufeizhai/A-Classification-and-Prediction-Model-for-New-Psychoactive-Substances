# A-Classification-and-Prediction-Model-for-New-Psychoactive-Substances
# A-Classification-and-Prediction-Model-for-New-Psychoactive-Substances
# 导入我们需要的包
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# 引入随机森林和决策树模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# 引入多层感知器（MLP）模型
from sklearn.neural_network import MLPClassifier, MLPClassifier
from sklearn.tree import DecisionTreeClassifier
# 引入划分训练集和和数据集的模块
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# 多分类指标,混淆矩阵
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import mglearn
# 引入可视化工具包
import seaborn as sn
import matplotlib.pyplot as plt
import umap





if __name__ == '__main__':
    data = pd.read_excel('MS data.xlsx')
    x = data.iloc[1:, 1:360].values
    y = data.iloc[1:, 361].values
    # print(data.head())
    # knn算法之前的降维
    pca = PCA(n_components=14)
    pca = pca.fit(x)
    x = pca.transform(x)
    # 通过PCA降维可视化整体数据的数据分布
    pca = PCA(n_components=2)
    pca = pca.fit(x)
    x_dr = pca.transform(x)

  julei = KMeans(n_clusters=8)
    # 对聚类的数据进行聚类
    julei.fit(x_dr)


  # 使用函数train_test_split()拆分数据集,test_size为测试集所占的比例，这边测试集所占的比例为30%
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)


  # 用神经网络模型训练数据，以及调参。hidden_layer_sizes指得是神经网络中隐藏层的数量和大小，
   ann_class = MLPClassifier(hidden_layer_sizes=(30, 50), solver='adam', activation='relu', max_iter=3000,
                            alpha=0.0001,
                             random_state=62)
    ann_class.fit(x_train, y_train)
    ann_classscore = ann_class.score(x_test, y_test)
    predict_ann = ann_class.predict(x_test)
    print(ann_classscore)
    print(confusion_matrix(y_test, predict_ann))
    print(classification_report(y_test, predict_ann))
    sn.heatmap(confusion_matrix(y_test, predict_ann), annot=True, cmap='BuPu')
    plt.ylabel("true label")
    plt.xlabel("predict label")
    plt.show()

  k=8
  clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
  clf.fit(x_train, y_train)
  clfscore = clf.score(x_test, y_test)
  predict_knn = clf.predict(x_test)
  print(clfscore)
  print(confusion_matrix(y_test, predict_knn))
  print(classification_report(y_test, predict_knn))
  sn.heatmap(confusion_matrix(y_test,predict_knn), annot=True, cmap='BuPu')
  plt.ylabel("true label")
  plt.xlabel("predict label")
  plt.show()



  # 用支持向量机算法训练模型以及调参
  svm_model = SVC(kernel='rbf', gamma=0.27,  C=22.65)
  svm_model.fit(x_train, y_train)
  # 预测测试数据集
  predicted_y = svm_model.predict(x_test)
  # 输出预测结果及模型评分
  score = svm_model.score(x_test, y_test)
  print(score)
  print(confusion_matrix(y_test, predicted_y))
  print(classification_report(y_test, predicted_y))
  sn.heatmap(confusion_matrix(y_test, predicted_y), annot=True, cmap='BuPu')
  plt.ylabel("true label")
  plt.xlabel("predict label")
  plt.show()
  print("Predicted lables:", predicted_y)
  print("Accuracy score:", svm_model.score(x_test, y_test))

  # gamma调参
  score = []
    gamma_range = np.logspace(-10, 1, 50)
    for i in gamma_range:
        clf = SVC(kernel='rbf', gamma=i).fit(x_train, y_train)
        score.append(clf.score(x_test, y_test))
  # 绘制学习曲线
  print('The best score is ', max(score), " , and it's gamma is ", gamma_range[score.index(max(score))])
  plt.plot(gamma_range, score)
  plt.show()

  # c调参
  score = []
  c_range = np.linspace(0.01, 30, 50)
    for i in c_range:
        clf = SVC(kernel='rbf', gamma=0.27, C=i).fit(x_train, y_train)
      score.append(clf.score(x_test, y_test))
   print('The best score is ', max(score), " , and it's c is ", c_range[score.index(max(score))])

  forest = RandomForestClassifier(n_estimators=125, random_state=2)
    # 用训练集数据来训练模型
    forest.fit(x_train, y_train)
    pred = forest.predict(x_test)
    # 输出随机森林的类别和类别的数量
    print(forest.classes)
  
  #  导入测试集，计算模型准确率
  score = forest.score(x_test, y_test)
    print(score)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    sn.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='BuPu')
    plt.ylabel("true label")
    plt.xlabel("predict label")
    plt.show()

  scores_imagine = mglearn.tools.heatmap(confusion_matrix(y_test, pred), xlabel='predicted label',
                                          ylabel='True label', xticklabels=y,
                                          yticklabels=y,
                                          cmap=plt.cm.gray_r, fmt="%d")
    plt.title("Confusion matrix")
    plt.gca().invert_yaxis()
