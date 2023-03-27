# 5-3 트리의 앙상블


# 배경 
# 정형데이터 - 앙상블 학습 - 대부분 결정 트리를 기반
# 비정형데이터 - 신경망 알고리즘 (전통적인 머신러닝 방법 X) - 사진과 텍스트 등의 비정형 데이터터

# 랜덤 포레스트 사용
# 앙상블 학습의 대표 주자, 안정적인 성능
# 결정 트리를 랜덤하게 만들어 숲을 만든다. 각 트리의 예측을 사용해 최종예측을 만든다.

# 부트스트랩 방식 : 데이터 세트에서 중복을 허용하여 데이터를 샘플링


# 랜덤포레스트 #

# 배경 
# 와인 데이터셋을 불러와서 각각의 알고리즘의 특징을 설명함.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

# 교차검증 수행 : cross_validate
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)

# 다소 훈련 세트에 과대적합
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 특성 중요도를 출력해봄 (특성 중요도 = 결정 트리에서 가장 큰 장점)
rf.fit(train_input, train_target)
print(rf.feature_importances_)

# OOB 샘플 : 부트스트랩 샘플에 포함되지 않고 남는 샘플 (마치 검증 샘플)
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)

# OBB 샘플의 결과
rf.fit(train_input, train_target)
print(rf.oob_score_)


# 엑스트라 트리 # 

# 특징 : 부트스트랩 샘플 사용 X, 전체 훈련 세트를 사용, 무작위성이 더 큼, 빠른 계산 속도

from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 특성 중요도를 출력해봄 
et.fit(train_input, train_target)
print(et.feature_importances_)



# 그레이디언트 부스팅 #

# 특징 : 깊이가 얕아 오차를 보완, 과대적합에 강하고, 일반적으로 높은 일반화 성능 그러나 속도가 느림
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

# 0.88, 0.87로 과대적합이 되지 않은 결과!
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 결정트리 개수를 500개로 늘렸지만 과대적합을 잘 억제함
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 특성 중요도를 출력해봄
gb.fit(train_input, train_target)
print(gb.feature_importances_)


# 히스토그램 기반 그레이디언트 부스팅 #

# 특징 : 정형 데이터 알고리즘 중 가장 인기가 높음, 트리개수 대신 부스팅 반복 횟수 지정

from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True, n_jobs=-1)

# 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10,
                                random_state=42, n_jobs=-1)

# 훈련세트 - 특성 중요도 계산
print(result.importances_mean)


# 테스트세트 - 특성 중요도 계산
result = permutation_importance(hgb, test_input, test_target, n_repeats=10,
                                random_state=42, n_jobs=-1)
print(result.importances_mean)


# 테스트 세트 - 성능 확인 - 87%로 앙상블 모델은 단일 결정 트리보다 좋은 결과
hgb.score(test_input, test_target) 


# 그외 라이브러리 #

# XGBoost #

from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))


# LightGBM #
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))



# 5-3 장 주제는 '트리의 앙상블'
# 앙상블 학습이란 정형 데이터에서 가장 뛰어난 알고리즘 중 하나.
# 랜덤 포레스트, 엑스트라 트리, 그레이디언트 부스팅, 히스토그램 기반 그레이디언트 부스팅을 학습.
