# k - 평균

# 배경 : 과일처럼 평균값을 미리 알고 있는 경우가 아닌, 비지도 학습의 경우 k-평균이 평균값을 자동으로 찾아줌
# 무작위로 k개의 클러스터 중심을 정함 - 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
# 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경 - 클러스터 중심에 변화가 없을 때까지 이 과정을 반복
  
# 한 마디로 군집의 중심을 자동으로 반복해서 찾아준다. (평균을 찾아준다)

# KMeans 클래스

# 과일 데이터를 불러옴
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

# 3차원 배열 (샘플 개수, 너비, 높이) → 2차원 배열 (샘플 개수, 너비x높이)
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans

# n_clusters=3이므로 labels_은 0,1,2이다.
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 레이블 샘플의 개수 확인 (큰 의미X)
print(km.labels_)
print(np.unique(km.labels_, return_counts=True))

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

# 레이블 0으로 클러스팅된 91개 이미지 모두 출력 
draw_fruits(fruits[km.labels_==0])    

# 레이블 1으로 클러스팅된 98개 이미지 모두 출력 
draw_fruits(fruits[km.labels_==1])

# 레이블 2으로 클러스팅된 111개 이미지 모두 출력 (완벽하게 구현은 X)
draw_fruits(fruits[km.labels_==2])


# 클러스터 중심 #
# KMeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_ 속성에 저장됨.
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

# 슬라이싱 연산자??
print(km.transform(fruits_2d[100:101]))

# 클러스터 중심을 예측 클래스로 출력 (2라는 파인애플 값이 나옴)
print(km.predict(fruits_2d[100:101]))

# 샘플은 파인애플인 것을 확인
draw_fruits(fruits[100:101])

# 알고리즘이 반복해서 최적의 클러스터를 찾은 횟수
print(km.n_iter_)


# 최적의 k 찾기 #

# 배경 : k-평균 알고리즘의 단점 중 하나는 클러스터 개수를 사전에 지정해야 한다는 것

# 적절한 클러스터 개수를 찾는 방법 : 엘보우 방법

# 이너셔 : 클러스터에 속한 샘플이 얼마나 가깝게 모여있는지 나타내는 값

# 이너셔가 크게 줄어들지 않는 꺾이는 점 = 최적의 클러스터 개수

inertia = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')

# 그래프에서 k=3에서 그래프가 살짝 꺾이는 것을 볼 수 있음
plt.show()



# 이번 장에서는 비지도 학습의 경우로 평균값을 정해주지 않는 k-평균에 대해 배웠음 
# k - 평균 알고리즘은 k개의 군집 중심의 개수를 설정하고, 그에 맞게 군집 중심과 샘플들의 거리를 계산하여 자동으로 과일을 찾아줌
# 최적의 k 찾는 방법 : 이니셔와 엘보우를 이용함
