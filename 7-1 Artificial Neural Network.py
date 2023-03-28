# 인공 신경망 #

# 패션 마케팅 팀은 이전보다 럭키백의 정확도를 높이고자 함
# 럭키백 ex) 신발이 들어있을 확률 87%, 모자가 들어있을 확률 13%로 알려주는 것

# 패션 MNIST 데이터를 다운로드

from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# 전달받은 데이터 크기는 60,000개의 이미지, 각 이미지 크기는 28x28, 타깃도 60,000개의 원소가 있음
print(train_input.shape, train_target.shape)

# 테스트 세트의 크기는 10,000개의 이미지 20%
print(test_input.shape, test_target.shape)

# 훈련 데이터에서 몇 개의 샘플을 그림으로 출력
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

# 처음 10개의 샘플의 타깃값을 리스트로 만든 후 출력
print([train_target[i] for i in range(10)])


# 레이블 당 샘플의 개수를 확인 = 정확히 6000개씩 10개가 들어있음
import numpy as np

print(np.unique(train_target, return_counts=True))


# 로지스틱 회귀로 패션 아이템 분류하기 #

# 2차원인 배열을 1차원 배열로 만듦
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

# 변환된 train_scaled의 크기를 확인 = 784개의 픽셀로 만들어진 60,000개의 샘플을 준비
print(train_scaled.shape)

# 교차 검증으로 성능을 확인 = 만족할 만한 수준은 X
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))

# 지금까지 로지스틱 회귀 (분류 모델)로 패션 아이템을 분류해보았음. 높은 결과X

# 딥러닝 : deep neural network, DNN, 인공신경망과 거의 동의어로 사용되는 경우가 많음

# 가장 인기가 높은 딥러닝 라이브러리인 텐서플로를 사용해 인공 신경망 모델을 만들어보기


# 텐서플로와 케라스 #
import tensorflow as tf
from tensorflow import keras


# 인공 신경망으로 모델 만들기 #

from sklearn.model_selection import train_test_split

# 검증 세트 나누기
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 훈련세트
print(train_scaled.shape, train_target.shape)

# 검증 세트 (훈련세트에서 20%를 검증세트로 덜어냄)
print(val_scaled.shape, val_target.shape)


# 밀집층 만들기 : 784개의 픽셀과 10개의 뉴런이 모두 연결되어 밀집
# 뉴런 개수 10개, 뉴런 출력에 적용할 함수 softmax, 입력의 크기 784
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))

# 이 밀집층을 가진 신경망 모델을 만들어야 함 (Sequential 클래스를 사용)
model = keras.Sequential(dense)


# 인공 신경망으로 패션 아이템 분류하기 #

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 정수로 되어있음
print(train_target[:10])

# 원핫인코딩 : 타깃값 해당 클래스만 1이고 나머지는 모두 0인 배열로 만드는 것

# 그런데 패션 MNIST 데이터의 타깃값은 어떤지 확인 > 그냥 정수로 되어있음
# 텐서플로에서는 그냥 사용이 가능

# 모델을 훈련 
# 결과가 85%를 넘는 이전보다 좋은 결과를 보여줌
model.fit(train_scaled, train_target, epochs=5)


# 앞서 떼놓은 검증 세트에서 모델의 성능을 확인
# 결과는 훈련 세트보다는 낮은 결과가 나옴
model.evaluate(val_scaled, val_target)



# 이 절에서는 28 x 28 크기의 흑백 이미지로 저장된 패션 아이템 데이터 셋인 패션 MNIST를 사용
# 먼저 로지스틱 손실 함수를 사용한 SGDClassifier 모델을 만들어 교차 검증 점수를 확인
# 케라스를 이용해 간단한 인공 신경망 모델을 만들어 패션 아이템을 분류해 보았음
