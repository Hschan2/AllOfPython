## DBSCAN 군집화
# DBSCAN - 반지름과 샘플 갯수가 주어지면 그 반지름 안에 해당하는 샘플 개수만큼 있는 이웃을 확장해 나가는 군집 알고리즘

# k-mean 예제
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# n_samples = 샘플수, n_features = feature 수, centers = 클러스터의 중심 값의 수
# cluster_std = 표준 편차, shuffle = 데이터 순서 셔플로, random_state = 랜덤성은 제로
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# 샘플의 x좌표, y좌표, 마커 색상, 마커 모양, 스토로크 색상, 마커의 크기
plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolors='black', s=100)

plt.grid()
plt.tight_layout()
plt.show()

## PCA (주성분 분석) 예제
import mglearn
import seaborn as sns
import pandas as pd

mglearn.plots.plot_pca_illustration()

# pandas로 알아보는 예제
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df.head()

# sklearn으로 알아보는 예제
# 표준화 패키지 라이브러리 추가
from sklearn.preprocessing import StandardScaler
# 독립 변인들의 value 값만 추출
x = df.drop(['target'], axis=1).values
# 종속 변인 추출
y = df['target'].values

# x 객체에 x를 표준화한 데이터 저장
x = StandardScaler().fit_transform(x)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
pd.DataFrame(x, columns=features).head()

# sklearn, PCA
from sklearn.decomposition import PCA

# 주성분 갯수 정하기
pca = PCA(n_components=2)
principalComponenets = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponenetsin, columns = ['principal component1', 'principal component2'])

# 주성분으로 이루어진 데이터 프레임 구성
# 성분으로 데이터를 설명할 수 있으며, 각 변수 중요도 마지막 행 참고 부호는 양과 음 상관 관계를 말한다.
principalDf

# 그리드로 주성분 성분 파악하기
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# targets의 각 색 지정
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component1']
               , finalDf.loc[indicesToKeep, 'principal component2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

## 군집화 예제

# K평균 군집화
# 주어진 데이터를 K개의 클러스터로 묶은 알고리즘
# 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작
# K는 클러스터의 중심 수를 의미

# 사이킷런 라이브러리로 K-평균 군집화 함수 불러오기
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# iris 데이터셋 불러오기
iris_dataset = load_iris()

# train 데이터셋 설정
train_input, train_label, test_input = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=42)

# K 값을 의미하는 군집의 개수 설정
k_means = KMeans(n_clusters=3)

# train 데이터를 이용해서 모델 학습시키기
k_means.fit(train_input)

# 클러스터 군집 확인
# 0번째 군집인지 1번째 혹은 2번째 군집인지 나타내는 라벨
# 실행될 때마다 결과가 변동될 수 있음
k_means.labels_

# 각 군집의 종 확인하기
train_label[k_means.labels_ == 0]
train_label[k_means.labels_ == 1]
train_label[k_means.labels_ == 2]

# test 데이터를 이용해서 결과값 (라벨) 예측하기
predict_cluster = k_means.predict(test_input)

# 군집을 의미하는 값을 iris의 종을 의미하는 값으로 변경하기
import numpy as np

np_arr = np.array(predict_cluster)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2] = 3, 4, 5
np_arr[np_arr == 3] = 1
np_arr[np_arr == 4] = 0
np_arr[np_arr == 5] = 2

predict_label = np_arr.tolist()

# 예측 값과 실제 값을 비교하여 성능 비교
print(f"test accuracy: {np.mean(predict_label == test_label):.2f}")