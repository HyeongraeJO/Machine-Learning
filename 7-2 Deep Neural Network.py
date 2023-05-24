# 목표 : 여러 개의 층을 추가하여 다층 인공 신경망을 만들기 = 심층 신경망
# (또 케라스 API를 이용하여 층을 추가하는 여러가지 방법)

# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf

tf.random.set_seed(42) # 랜덤 시드를 42로 설정하여 랜덤 연산이 코드를 실행할 때마다 같도록!

# 병렬 처리 설정 변경 from CHAT GPT.....
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)



#2개의 층
# 패션데이터셋을 로드함 (패션 아이템 이미지로 구성 -> 훈련 / 테스트)

from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data() 


from sklearn.model_selection import train_test_split

# train_input은 '0과 255'사이의 값을 가지는 이미지 데이터 -> 0과1 사이로 스케일링하기 위해..
train_scaled = train_input / 255.0

# 2차원 배열 -> 1차원 배열 (-1은 원본 배열의 길이를 유지하면서 나머지 차원을 28x28로 조정하라는 의미)
train_scaled = train_scaled.reshape(-1, 28*28)

# train set vs target set로 나눔 
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 은닉층 = dense 1 : 몇개의 뉴런 ->  상당한 경험, 적어도 출력층의 뉴런보다 많게!
# 출력층 = dense 2
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')



#심층 신경망 만들기 = 인공 신경망의 강력한 성능! 층 추가 + 연속적인 학습
# 결과를 보면... None = 샘플 개수가 아직 정의 X, 은닉층의 뉴런개수 100개 (784개 픽셀값 -> 100개로 압축)
# 마지막으로 모델 파라미터 개수 - 784 x 100 + 100
model = keras.Sequential([dense1, dense2])
model.summary()


#층을 추가하는 다른 방법 (name 매개 변수)

model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')
model.summary()

#층을 추가하는 다른 방법 (add 메서드)

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

# 모델을 컴파일 (손실함수 : 다중 클래스 분류 문제에 사용, 평가지표 설정 : 정확도!)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델을 훈련하는 단계 (5번 반복)
model.fit(train_scaled, train_target, epochs=5)


# 활성화 함수의 주요 역할은 비선형성을 추가
#렐루 : 활성화 함수 (층이 많은 심층 신경망 -> 학습이 어려움 -> 렐루함수 ???)
# 입력 데이터를 Flatten하여 1차원 벡터로 변환한 후, Dense 레이어를 거쳐 최종 출력을 만들어냅니다.
# Flatten 층을 신경망 모델에 추가하면 입력값의 차원을 짐작할 수 있음
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()


(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 결과 : 시그모이드 함수를 사용했을 때와 비교하면 성능이 약간 향상
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
model.evaluate(val_scaled, val_target)


#옵티마이저 = 하이퍼파라미터(은닉층의 개수, 뉴런 개수, 활성화 함수, 층의 종류)
# sgd 확률적 경사 하강법 옵티마이저
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

sgd = keras.optimizers.SGD(learning_rate=0.1)

sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)


adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')

rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')

# 기본 Adam 이용
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

# 기본 RMSprop과 거의 같은 성능
model.fit(train_scaled, train_target, epochs=5)

# 검증세트에서 성능 -> 기본 RMSprop과 조금 좋은 성능
model.evaluate(val_scaled, val_target)


