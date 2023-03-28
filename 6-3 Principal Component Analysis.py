# 주성분 분석 (PCA)

# 배경 : 과일가게 주인은 고객들이 원하는 과일 사진을 잘 분류하는 알고리즘에 만족.
# 그러나 너무 많은 사진이 등록되어 저장 공간이 부족해 업로드된 사진의 용량을 줄여야 함

# 차원 : 예를 들어 과일 사진의 경우 10,000개의 픽셀이 있기 때문에 10,000개의 특성 = 10,000개의 차원

# 이 차원을 줄이는 것이 이번 장의 목표.

# 다차원 배열에서의 차원 : 배열의 축 개수

# 1차원 배열에서의 차원 : 원소의 개수

# 차원 축소 알고리즘 : 비지도 학습 작업 중 하나

# 너무 많은 차원이나 특성이 있으면 훈련 데이터에 쉽게 과대적합됨 따라서 차원 축소로 데이터 크기를 줄이고 성능 향상이 필요함.

# 분산 : 데이터가 멀리 퍼져있는 정도 (분산이 큰 방향이란 데이터를 잘 표현하는 어떤 벡터 = 주성분 벡터)

# 주성분 분석의 순서 : 1) 분산이 큰 주성분을 찾음 2) 두번째 주성분 : 이 벡터에 수직이고 분산이 큰 다음 방향을 찾음 3) 주성분은 원본 특성의 개수만큼 찾음


# 과일 데이터 불러오기
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.decomposition import PCA

# 주성분으로 처음에 50개를 지정
pca = PCA(n_components=50)
pca.fit(fruits_2d)

# 첫번째 차원 50 = 50개의 주성분 찾음
print(pca.components_.shape)

import matplotlib.pyplot as plt


def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

# 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸 것
draw_fruits(pca.components_.reshape(-1, 100, 100))

# 100 x 100 = 10000개의 픽셀을 가진 이미지 300개
print(fruits_2d.shape)

# 주성분 분석을 통해 50개의 특성을 가진 이미지 300개
fruits_pca = pca.transform(fruits_2d)

print(fruits_pca.shape)


# 원본 데이터 재구성 #

# 앞서 10,000개의 특성을 50개로 줄였기 때문에 어느 정도 손실이 발생
# 앞서 50개로 축소한 데이터를 10,000개의 특성으로 복원
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)

# 이 데이터를 100개씩 나눠서 출력
# 결과 : 잘 복원한 것을 알수 있음 = 50개의 특성을 잘 선택했다는 뜻
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")


# 설명된 분산 #

# 설명된 분산 : 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값
# 결과 : 92%의 분산을 유지하고 있음 = 앞서 원본 데이터 복원이 잘 됐던 이유
print(np.sum(pca.explained_variance_ratio_))

# 분산을 그래프로 출력
plt.plot(pca.explained_variance_ratio_)
plt.show


# 다른 알고리즘과 함께 사용하기 #

# 배경 : 과일 사진 원본 데이터와 차원 축소한 데이터를 지도 학습에 적용하여 비교해보기 (3개의 과일 사진 비교)

# 로지스틱 회귀 모델 사용
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

target = np.array([0] * 100 + [1] * 100 + [2] * 100)

from sklearn.model_selection import cross_validate

# 먼저 원본데이터인 fruits_2d를 사용
# 결과 : 교차 검증 점수 0.997로 높음
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 다음으로 주성분 분석 수행한 fruits_pca를 사용
# 결과 : 정확도 100%, 훈련시간도 감소 했음
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))


# 원하는 분산의 비율을 입력 = 50%
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

# '2개'의 주성분을 찾음 > '2개'의 특성만으로 원본 데이터에 있는 분산의 50%를 표현!!
print(pca.n_components_)

# 이 모델로 원본 데이터를 변환
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

# 2개의 특성만 사용했을 뿐인데 99% 정확도 달성!
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))



# K-평균 알고리즘 사용 (차원 축소된 데이터 사용)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

# 클러스터 각각 91개, 99개, 110개의 샘플을 포함
# 결과 : 원본 데이터와 비슷함
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

# 훈련 데이터의 차원을 줄이면 시각화를 얻을 수 있음
# 2개의 특성이 있기 때문에 2차원으로 표현 가능!
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()


# 이번 장에서는 주성분 분석으로 '차원축소'를 해보았음.
# 차원 축소는 비지도 학습의 대표적인 문제로 '데이터 크기'감소와 '훈련 시간'감소, '시각화'가 가능함
# 차원 축소한 데이터를 지도, 비지도 학습 알고리즘에 재사용하여 성능을 높이거나 훈련 속도를 빠르게 만들 수 있음
# 설명된 분산 : '주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값'을 통해 원하는 비율만큼 주성분을 찾을 수 있음
# 복원 : PCA 클래스는 원본 데이터를 복원하는 메서드도 제공함