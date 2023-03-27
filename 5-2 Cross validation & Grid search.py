# 5-2 교차 검증과 그리드 서치

# 배경 : "이런저런 값으로 모델을 많이 만들어서 테스트 세트로 평가하면 결국 테스트 세트에 잘 맞는 모델이 만들어지는 것 아닌가요?"

# 테스트 세트는 가장 마지막에 딱 한 번만 수행하는 것이 좋음.

# 검증 세트 : 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련 세트를 또 나누는 것 (80%의 훈련세트에서 또 20%를 떼어 검증세트로 사용한다.)

# 전체에서 보통 20~30%를 테스트 세트와 검증 세트로 떼어 놓는다. 


# 검증 세트 #

import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)


# 검증세트 = val_input, val_target (train input의 약 20%를 val_input으로 만듦)
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

#1 훈련 세트와 검증 세트의 크기를 확인 (훈련세트는 5197 → 4157, 검증세트는 1040)
print(sub_input.shape, val_input.shape)

# 모델을 만들고 평가하는 과정
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

# 99.7%, 86.4%로 훈련세트에 과적합되어 있음
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))


# 교차 검증 #

# 교차 검증의 배경 : 검증 세트를 만드느라 훈련 세트가 줄었음. 많은 데이터를 훈련에 사용할수록 좋은 모델 만들어짐.

# 교차 검증 : 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복! 그 다음 점수를 평균하여 최종 검증 점수를 얻음. (3-폴드 교차 검증 : 훈련 세트 세 부분으로 나누는 것)

# 함수 cross_validate : 직접 검증 세트를 떼어 넣는 것이 아니라 훈련 세트 전체를 함수에 전달함.
from sklearn.model_selection import cross_validate

# fit_time : 모델 훈련하는 시간, score_time : 검증하는 시간, test_score : 5개의 최종 점수
scores = cross_validate(dt, train_input, train_target)
print(scores)

# 5개의 최종 점수의 평균
import numpy as np

print(np.mean(scores['test_score']))


# 분할기(splitter) : 교차 검증시 훈련 세트를 섞기 위함
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))

# n_splits=10 몇 폴드로 나눠서 교차검증 할 것인지
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))


# 배경 : 테스트 세트를 사용하지 않고 교차 검증을 통해 좋은 모델을 선정 → 결정 트리의 매개변수 값을 바꿔가며 가장 좋은 성능 모델을 찾기.


# 하이퍼파라미터 튜닝 → '그리드 서치' #

# 하이퍼파라미터 : 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터

# '그리드서치' : 친절하게도 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행함.

from sklearn.model_selection import GridSearchCV

# 파라미터는 0.0001부터 순차적으로 올라가는 5개의 값을 시도
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

# jobs는 cpu 수 1 = 전체 cpu 사용
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

# 그리드 서치를 수행
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))

# 가장 좋은 파라미터를 선택 → 0.0001
print(gs.best_params_)

# 결과 확인
print(gs.cv_results_['mean_test_score'])

# 최상의 결과를 내는 가장 좋은 파라미터 → 0.0001
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])

# 조금 더 복잡한 매개변수 조합을 탐색
# np.arange는 1번부터 2번까지 3번의 간격으로..
# range는 정수만 사용 가능, 1번부터 2번까지 3번의 간격으로.. 총 모델 수 : 6,750개
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }

# 그리드서치 실행
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

# 최상의 매개변수 조합 확인
print(gs.best_params_)

# 최상의 교차 검증 점수도 확인
print(np.max(gs.cv_results_['mean_test_score']))


# 랜덤 서치 #

# 배경 : 매개변수 값의 범위나 간격을 정하기 어려울 때 랜덤 서치를 사용

from scipy.stats import uniform, randint

# 간단한 연습 : 난수 발생기와 유사한 랜덤 서치
rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)

# 탐색할 매개변수 범위
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }

# 그리드 서치 실행 (랜덤 서치)
# 총 100번을 샘플링(난수 발생) 하여 훨씬 교차 검증 수를 줄이고, 넓은 영역을 효과적으로 탐색
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# 가장 좋은 파라미터 확인
print(gs.best_params_)

# 최적의 교차 검증 점수도 확인
print(np.max(gs.cv_results_['mean_test_score']))

# 최종 테스트 점수 확인 : 다양한 매개변수를 충분히 테스트 해서 얻은 결과.
dt = gs.best_estimator_

print(dt.score(test_input, test_target))

# 이번 장에서는 '교차 검증'이 주제였고, 교차 검증이란 훈련데이터 내에서 여러가지를 바꿔보고 변경해가면서 좋은 결과를 찾는 과정이다. (테스트 세트를 최대한 건드리지 x)
# 이 과정을 자동화 하여 최적의 파라미터를 찾는 과정이 '그리드 서치'가 되겠다. '랜덤 서치'는 더 편하게 파라미터를 찾도록 해준다.

