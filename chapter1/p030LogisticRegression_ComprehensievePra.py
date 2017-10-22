# coding:utf8
# 对应的工具包文档pandas,matplotlib，除了编辑器提示，还可以参考：
# http://matplotlib.org/contents.html
# http://scikit-learn.org/stable/user_guide.html
# http://pandas.pydata.org/pandas-docs/stable/

# 数据和示例代码
# https://pan.baidu.com/s/1dENAUTr
# http://pan.baidu.com/s/1geN6QbD

# 导入pandas工具包，更名为pd
import pandas as pd

# 调用pandas工具包的read_csv函数/模块，传入训练文件的地址参数，获得返回的数据并存至变量df_train，df_test
df_train = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('../Datasets/Breast-Cancer/breast-cancer-test.csv')
# 选取测试集中的‘Clump Thickness’ 与 ‘Cell Size’作为特征，构建正负分类样本  （选取某字段等于xx的样本中的yy字段）
df_test_negative = df_test.loc[df_test['Type']
                               == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type']
                               == 1][['Clump Thickness', 'Cell Size']]
# print "positive:", df_test_positive
# print "negative:", df_test_negative
# positive:      Clump Thickness  Cell Size
# 3                  5          5
# 7                  6          6
# 8                  4         10
# first line is the field, 3 7 8 is the index

# 导入matplotlib工具包中的pyplot并简化命名为plt
import matplotlib.pyplot as plt

# 绘制良性肿瘤样本点，标记为红色o, 恶性肿瘤标记为黑色的x, Marker size is scaled by s
plt.figure("1 the test set")
plt.scatter(df_test_negative['Clump Thickness'],
            df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'], marker='o', s=150, c='black')
# 绘制x,y轴说明标签
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# 这个图作为1-2显示
# plt.show()放到最后


# 导入numpy工具包，重命名为np
import numpy as np
# numpy的random函数，随机产生直线的截距和系数
intercept = np.random.random((1))  # 输入参数shape:tuple of in
coef = np.random.random((2))
lx = np.arange(0, 10)
ly = (-intercept - lx * coef[0]) / coef[1]
# 绘制这条随机的直线
plt.figure("2 random line for the test set")
plt.plot(lx, ly, c='yellow')


# 如1-2一样，绘制1-3
plt.scatter(df_test_negative['Clump Thickness'],
            df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'], marker='o', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# plt.show()放到最后



#从sklearn工具包中的子包linear_model导入LogisticRegression, 并分类器实例化这样的分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# 使用前十条训练样本学习直线的系数和截距，
# 这是线性模型，所以用系数和截距来表征
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
# 打印出来 score： 准确度
print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])


# 提取训练了十个样本的线性模型
intercept=lr.intercept_
coef=lr.coef_[0,:]
#原本分类面应该是 lx * coef[0] + ly * coef[1] + intercept =  0,映射到二维平面，应该是
ly = (-intercept - lx * coef[0]) / coef[1] #这里是一般式 和 y函数式吧

#像前面一样绘制图 1-4
plt.figure("3 model with 10 training samples for the test set")
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],
            df_test_negative['Cell Size'], marker='o', s=200, c='red')
plt.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'], marker='o', s=150, c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
# plt.show()放到最后


#这次训练所有的数据
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type']) # 两个中括号，代表多字段？？ 多一维的列表
#lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10]) 对比只训练了10个
print 'Testing accuracy (all training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])

#再像前面一样提取 截距 和 斜率，画出最后的曲线
intercept=lr.intercept_
coef=lr.coef_[0,:]
# coef[0]*x+coef[1]*y+intercept=0
ly=(-lx*coef[0]-intercept)/coef[1]

#绘制图1-5
plt.figure("4 model with all training samples for the test set")
plt.plot(lx,ly,c='blue')
plt.scatter(df_test_negative['Clump Thickness'],
            df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],
            df_test_positive['Cell Size'],marker='o',s=150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()