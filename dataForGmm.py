import numpy as np
import pandas as pd
#from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.mixture import GaussianMixture
import time

start_time=time.time()


#genomeScore=np.genfromtxt('D:/emmaPractice/ml-20m/pythonGmm/genome-scores.csv', skip_header=1,delimiter=',')
#提取每个电影的向量表示
genomeScore=pd.read_csv('D:/pythonGmm/genome-scores.csv',dtype={'movieId':int,'tagId':int,'relevance':float},low_memory=False)
#M=csr_matrix(genomeDcore['relevance'],(genomeScore['movieId'],genomeScore['tagId']),shape(131170,1128)).toarray()
movieId = genomeScore['movieId'].unique()
#geSc2 =genomeScore['tagId']
geSc3=genomeScore['relevance']
M = geSc3.values.reshape((10381,1128))
#矩阵M进行主成分分析
pca=PCA(n_components=200)
M=pca.fit_transform(M)

#读取评分数据
ratings=pd.read_csv('D:/pythonGmm/ratings.csv',dtype={'userId':int,'movieId':int,'rating':float,'timestamp':float},low_memory=False)
#ratings=pd.DataFrame(ratings)
ratings1=ratings['userId']
countNum=ratings1.value_counts()#用户评分次数统计
countNum=pd.DataFrame(countNum)
#根据打分数目提取用户
countMore2000=countNum[countNum['userId']>2000]

MAE=0
count=0
MAElist=[]
for u in range(255):
#chosenRatings=pd.DataFrame()
#for userId in countMore2000.index[0]:
    chosenRatings=ratings.loc[ratings['userId']==countMore2000.index[u]]
#chosenRatings=chosenRatings.append(chosen)
#建立movieId-tag矩阵
    M=pd.DataFrame(M)
    M['movieId']=pd.Series(movieId)
#movieId.columns=['movieId']
    M1=pd.merge(chosenRatings,M)
    count=count+len(M1)
#进行10-fold 验证
    kf=cross_validation.KFold(len(M1),n_folds=10)
    MAEperUser=0
    for train_index,test_index in kf:
        print(train_index,test_index)
    #提取训练数据集和测试数据集 y为评分
        Mx_train=M1.iloc[train_index,4:-1]
        My_train=M1.iloc[train_index,2]
        Mx_test=M1.iloc[test_index,4:-1]
        My_test=M1.iloc[test_index,2]
   #定义GMM模型
        clf = GaussianMixture(n_components=5, covariance_type='full')#diagonal
#frames=[Mx_train,My_train]
        TrainData=pd.concat([Mx_train,My_train],axis=1)#按照列名连接
        TestData=pd.concat([Mx_test,My_test],axis=1)
    #训练模型
        clf=clf.fit(TrainData)
#predict=clf.predict(TestData)
    #print
        testScore=[]
        My_test=pd.DataFrame(My_test)
    
        for i in range(len(Mx_test)):
            score=0
            p=0
            s=pd.DataFrame([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
            M_Test=np.matlib.repmat(Mx_test.iloc[i],10,1)
            M_Test=pd.DataFrame(M_Test)
            M_Test=pd.concat([M_Test,s],axis=1)
            predictPro=clf.score_samples(M_Test)
            score=predictPro.argmax()/2
            testScore.append(score)
        testScore=pd.DataFrame(testScore,index=My_test.index,columns=My_test.columns)   
        MAEperKfoldabs=abs(testScore-My_test)   
        MAEperKfoldsum= MAEperKfoldabs.apply(lambda x: x.sum())
        print('MAEperKfoldsum:',MAEperKfoldsum)
        MAEperUser =MAEperUser+ MAEperKfoldsum  
#df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1) #计算各列的行并将其添加至最后一列
    MAE=MAE+MAEperUser
    MAEperUser=MAEperUser/len(M1)
    MAElist.append(MAEperUser)   
MAE=MAE/count
print(time.time()-start_time)