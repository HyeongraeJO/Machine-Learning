
# 로지스틱 회귀로 와인 분류하기#

import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 와인 확인하기
wine.head()

#데이터 프레임 정보
wine.info()
wine.describe()

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

#데이터 나눠주기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 사용
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

# 가중치 혹은 계수 3개와 절편 1개
print(lr.coef_, lr.intercept_)


# 결정 트리 #
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

# 굉장히 좋은 분류 정확도
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 루트노드 : 가장 위에 노드, 리프노드 : 가장 아래의 노드  
plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

# 트리 줄이기
# 루트(부모) 노드 : 맨 위 노드, 자식 노드 : 맨 아래 노드
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# 지니 불순도 : criterian + gini = 1-(음성 클래스 비율 제곱 + 양성클래스 비율 제곱) 0.5 일때 최악
# 공식이 따로 존재함

# 가지치기 #

# 과소, 과대 적합 줄이기 위해
# max_depth : 더 이상 분할하지 않는 수 3개까지만 분할
# 로지스틱 함수와 다르게 결정 트리함수는 선형함수를 학습할 필요가 없다. (스케일 조정하지 않은 특성 사용한다는 장점.)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# Sugar가 가장 중요한 특성 (2번째)
print(dt.feature_importances_)

# 결정 트리를 여러개 사용하는 앙상블이 좋은 성과를 낸다. (기본)
# 결정 트리는 눈으로 보기 쉽고 설명하기 쉽다. (PT)

# 확인문제 #
dt = DecisionTreeClassifier(min_impurity_decrease=0.0005, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


