
# 데이터 준비하기 #
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()

# 데이터에 어떤 종류가 있는지 확인
print(pd.unique(fish['Species']))

# 6개 열 중 'Species' 빼고 5개 열은 입력 데이터로 사용
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()

# 확인 : 5개 행을 출력
print(fish_input[:5])

# [데이터 준비] 타겟 지정, 훈련세트와 테스트세트를 표준화 전처리
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# KNN의 확률 예측 #
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)

# 테스트 세트의 처음 5개 샘플
print(kn.predict(test_scaled[:5]))

# 위 예측이 어떤 확률인지 확인 (kn.classes 순서대로)
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# 4번째 샘플의 최근접 이웃 클래스 확인
distances, indexes = kn.kneighbors(test_scaled[3:4]) #Roach가 1개, Perch가 2개
print(train_target[indexes]) # 앞 배열에서 4번째 줄의 Perch가 0.67로 같다는 것을 볼 수 있음 (KNN 성공)

# 로지스틱 회귀 (분류 모델이지만 선형 방정식) #
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1) # 시그모이드 함수 출력 (이진분류 : 0.5를 기준 양성 음성 클래스)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 불리언 인덱싱 (True, False 값을 전달하여 행 선택)
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

# 도미(Bream)이 True가 되도록 적용
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 훈련 모델 사용해 처음 5개 샘플 예측
print(lr.predict(train_bream_smelt[:5]))  ### train_bream_smelt에는 누가 샘플을 넣었는지???

# 예측 확률을 출력해봄
print(lr.predict_proba(train_bream_smelt[:5]))

# 두번째로 나온 Smelt가 양성 클래스
print(lr.classes_)

# 로지스틱 회귀가 학습한 계수
print(lr.coef_, lr.intercept_)

# z값을 출력(시그모이드 함수)
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# decisions 배열 = predict_proba() 메서드 출력의 두번째 열의 값과 동일
from scipy.special import expit

print(expit(decisions))


# 로지스틱 회귀로 다중 분류 수행하기 #
# 규제와 C는 반비례
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target) # train scaled, target은 이미 7개의 생선 데이터 들어있음

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 테스트 세트의 처음 5개 샘플 예측
print(lr.predict(test_scaled[:5]))

# 테스트 세트의 처음 5개 샘플 예측 확률 출력
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

# 배열의 속성을 확인 > 첫 샘플(1행)은 0.841(3번째) Perch를 예측
print(lr.classes_)

# 학습한 계수를 확인 (5개의 특성을 사용하므로 coef_ 배열의 열은 5개)
# 행은 7개 = z를 7개 계산
# 다중 분류는 클래스마다 z값을 하나씩 계산
# 당연히 가장 높은 z값을 출력하는 클래스가 예측 클래스
# 이진분류 = 시그모이드 함수, 다중 분류 = 소프트맥스 함수
print(lr.coef_.shape, lr.intercept_.shape)

# 이진분류처럼 decision_function() 매서드로 z값 구함
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 다음 소프트맥스 함수를 이용해 확률로 변환
from scipy.special import softmax

# 결과:  앞서 구한 proba 배열과 일치하면 성공!
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
