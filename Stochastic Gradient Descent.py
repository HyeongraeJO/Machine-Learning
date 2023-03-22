# 점진적 학습을 위한 확률적 경사 하강법 #
# 손실 함수라는 산을 정의하고 가장 가파른 경사를 따라 조금씩 내려오는 알고리즘
# 딥러닝에 꼭 필요
# 훈련하는 방법일 뿐 (최적의 값을 찾아줌) 확률 결과를 내는 알고리즘 X
# Epoch : 훈련 세트를 한 번 돌리기
# 미니 배치 경사 하강법 : 몇 개의 무작위 샘플 선택하여 경사 내려가기
# 배치 경사 하강법 : 전체 샘플 사용
# 손실함수 : 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준 (정확도는 미분가능X → 손실함수 X)
# 로지스틱 손실 함수 : 회귀에서는 정확도 대용으로 사용 가능, 분류(정확도로 측정하기 때문에 X)
# 0은 타깃으로 사용할 수 없음. 예측 x 타깃 = 0 되기 때문에
# 로그 함수 사용  (=이진 크로스엔트로피 손실 함수)
# 조건 : 스케일이 같아야 함 (데이터 전처리해서 특성의 스케일을 표준 점수로)

# SGDClassifier : 분류 모델일때 어떤 Classifier를 최적화 할 지
# SGDRegressor : 회귀 모델일때

# SGD Classifier > 배치 or 미니 배치는 지원 X

import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier

# max_iter = Epoch
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

# 정확도 0.775로 과소적합이기때문에 더 학습을 진행해야함 epoch 늘리기
# 규제 작아지면 과대 적합이 됨 따라서 최적점을 찾아야 함 (테스트 세트가 떨어짐)
# EPOCH 커지면 과대 적합이 됨 따라서 최적점을 찾아야 함 (테스트 세트가 떨어짐)
print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# fit과 다르게 기존 학습했던 절편(w, b)유지하고 다시 훈련 (정확도 점진적 향상)
sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))


# EPOCH 과대/과소 적합 #
#조기종료#
import numpy as np

sc = SGDClassifier(loss='log', random_state=42)

train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes) #partial_fit 데이터가 일부분만 전달
    
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

    import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 그래프를 보니 100 정도의 EPOCH가 좋아보임
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))

# hinge : 서포트벡터머신 - SGDClassifier
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
print(sc.score(test_scaled, test_target))