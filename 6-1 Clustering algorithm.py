# 군집 알고리즘

# 비지도 학습 : 타깃이 없을 때 사용하는 머신러닝 알고리즘

# 배경
# 한빛 마켓은 농산물을 판매하는데 고객들에게 사진으로 과일 세일 추천을 받으려고한다.
# 이때 여러장의 사진을 분류하는 알고리즘이 필요하다.
# 아이디어1 : 사진의 픽셀값을 모두 평균내서 분류


# 과일 사진 데이터 준비하기 #


import wget




def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
    return progress

def download(url, out_path="."):
    wget.download(url, out=out_path, bar=bar_custom)

if __name__ == "__main__":
    url = 'https://bit.ly/fruits_300_data'
    download(url)




import numpy as np
import matplotlib.pyplot as plt

fruits = np.load('fruits_300.npy')

# (샘플의 개수, 이미지 높이, 이미지 너비)
print(fruits.shape)

# 첫번째 이미지의 첫번째 행 100개의 값
print(fruits[0, 0, :])

# 저장된 이미지를 그림 - 사과
# 관심 대상을 높은 숫자로 → 사과를 흰색으로 (0에 가까울수록 검게)
plt.imshow(fruits[0], cmap='gray')
plt.show()

# 관심 대상을 낮은 숫자로 → 사과를 검은색으로
plt.imshow(fruits[0], cmap='gray_r')
plt.show()


# 바나나와 파인애플 이미지도 출력
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()


# 픽셀 값 분석하기 #

# 100 x 100인 이미지를 펼쳐서 길이가 10,000인 1차원 배열로 만들기 

# 슬라이싱 연산자????
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

print(apple.shape)

# 샘플마다 픽셀의 평균값을 계산
print(apple.mean(axis=1))

# 히스토그램과 범례 결과 - 사과와 파인애플은 구분이 힘듦
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()


# 픽셀별 평균값을 비교해보고자 함 (전체 샘플에 대해 각 픽셀의 평균을 계산)
# 과일 3 종류별로 각각의 픽셀의 평균을 계산
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 위의 픽셀 평균값을 100x100 크기로 바꿔서 이미지처럼 출력
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()



# 평균값과 가까운 사진 고르기 #

# 배경 : 사과 사진의 평균값인 apple_mean과 가장 가까운 사진 고르기

# 모든 샘플에서 apple_mean 값을 뺀 절댓값의 평균을 계산
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2))
print(abs_mean.shape)

# 값이 가장 작은 순서대로 샘플 100개 고르기
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
        axs[i, j].axis('off')
# apple_mean과 가장 가까운 사진 100개를 골랐더니 모두 사과가 나옴
plt.show()



# 흑백 사진에 있는 픽셀값을 사용해 과일 사진을 모으는 작업을 해봤음
# 비슷한 샘플끼리 그룹으로 모으는 작업을 군집이라고함 (비지도 학습 작업 중 하나)
# 클러스터 : 군집 알고리즘에서 만든 그룹
# 이번 실험은 사실 사과, 바나나, 파인애플 타깃값을 알고 있었기 때문에 타깃값의 평균값을 미리 구할 수 있었음!!!