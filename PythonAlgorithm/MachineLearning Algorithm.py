## 머신 러닝을 위한 알고리즘
# 2021-01-13, 머신 러닝 알고리즘 겉할기로 미리 알아보기

# 1. 지도 학습 알고리즘
# 의도하는 결과가 있을 때 사용하는 알고리즘.
# 학습하는 동안 모델은 입력으로 들어온 값으로 변수를 조정해서 출력에 매핑한다.

# 2. 비지도학습 알고리즘
# 의도하는 결과가 없을 때 사용하는 알고리즘.
# 입력 데이터 집합을 비슷한 유형의 집합으로 분류한다.

# 3. 강화학습 알고리즘
# 결정을 내리도록 할 때 사용하는 알고리즘.
# 결정했을 때 성공 혹은 실패에 따라 주어진 입력 값에 대한 결정이 점차 달라진다.
# 학습을 할수록 입력에 대한 결과 예측이 가능하다.

## 머신러닝 알고리즘 세부 목록
# 1) 선형 회귀 알고리즘
# 지도 학습 중 예측 문제에 사용하는 모델
# 데이터 모델에 가장 적합한 선을 찾기 위해 데이터의 점들을 사용
# y = mx + x의 방정식으로 나타낼 수 있는 알고리즘
# y = 종속변수, x = 독립변수, m과 c = 주어진 데이터셋을 기초적인 미적분을 사용하여 찾기
# 독립변수가 하나만 사용되는 단순 선형회귀, 독립변수가 여러개 사용되는 다중 선형회귀로 분류
# 모델의 학습 - 라벨이 있는 데이터로부터 올바른 가중치와 편향값을 결정
# 모델의 손실 - 잘못된 예측에 대한 벌점 (모델의 예측이 얼마나 잘못되었는지 나타내는 수), 모델의 예측이 완벽하면 손실은 0이며 그렇지 않으면 손실은 크다.
# 모델 학습의 목표는 평균적으로 작은 손실을 갖는 가중치와 편향의 집합을 찾는 것

# sklearn 라이브러리를 이용하여 선형회귀 구현 가능

## 최소제곱법
# 어떤 계의 해방정식을 근사적으로 구하는 방법
# 근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법
# 노이즈에 취약하며 특징 변수와 샘플 건수에 비례해서 계산 비용이 높다

# 최소제곱법과 선형회귀 알고리즘
import mglearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 50개의 데이터 생성
x = np.arange(50)
# 기울기
a = 0.3
# 절편
b = 0.8

# 선형회귀 식으로 작성. y값 산출하기
y = a * x + b

# 위 y 데이터 생성 시 일직선으로 표현되는 단순한 선형함수이기 때문에 noise 추가
# noise 랜덤으로 생성
noise = np.random.uniform(-1.25, 1.25, size=y.shape)
# y 값에 노이츠 추가
yy = y + noise

# 그래프로 생성하기
plt.figure(figsize=(10, 7))
plt.plot(x, y, color='r', label='y = 0.3 * x + b')
plt.scatter(x, yy, label='data')
plt.legend(fontsize=18)
plt.show()

## 위 코드를 함수형으로 만들기
def make_linear(w = 0.5, b = 0.8, size = 50, noise = 1.0):
    x = np.arange(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y + noise
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='r', label='y = {w} * x + {b}')
    plt.scatter(x, yy, label='data')
    plt.legend(fontsize=20)
    plt.show()
    print(f'w: {w}, b: {b}')
    return x, yy

x, y = make_linear(size = 50, w = 1.5, b = 0.8, noise = 5.5)

## 최소제곱법 공식
# 실제 값(y)과 가설(y_hat)에 의한 예측 값의 차이가 가장 작은 계수 계산

# x_bar(x 평균), y_bar(y 평균)
x_bar = x.mean()
y_bar = y.mean()

# w의 계수 값 찾기 (공식 활용)
calculated_weight = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar)**2).sum()

# b의 계수 값 찾기 (공식 활용)
calculated_bias = y_bar - calculated_weight * x_bar

## 최소제곱법은 노이즈에 취약하다 (노이즈 값을 증가 시켰을 때)
x, y = make_linear(size = 50, w = 0.7, b = 0.2, noise = 5.5)

# 임의의 2개의 outlier를 추가
y[5] = 60
y[10] = 60

plt.figure(figsize(10, 7))
plt.scatter(x, y)
plt.show()

# 위에서 구한 노이즈를 추가한 데이터를 공식에 적용하여 각각 계수 구하기
x_bar = x.mean()
y_bar = y.mean()
calculated_weight = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar)**2).sum()
calculated_bias = y_bar - calculated_weight * x_bar

## 선형 회귀 알고리즘 실습하기
# w[0] = 기울기, b = y축과 만나는 절편
# w는 각 특성에 해당하는 기울기를 가진다.
# 예측값은 입력 특성에 w의 각 가중치를 곱해서 더한 가중치의 합
# 1차원 데이터 셋을 활용하여 파라미터 w[0]와 b를 직선처럼 되도록 학습시키기

# 라이브러리에 설정되어 있는 데이터 이용
mglearn.plots.plot_linear_regression_wave()

# 기울기가 0.393906로 약 0.4가 출력하는 것을 확인 가능
# 회귀 모델의 특징 - 특성이 하나일 때 직선, 두 개일때는 평면이며 더 높은 차원에서는 초평면
# 특성이 많은 데이터셋이라면 선형 모델을 매우 좋은 성능을 낸다.

## 선형회귀
# 예측과 훈련 데이터에 있는 타겟 y 사이의 평균제곱오차(MSE)르르 최고화하는 파라미터 w, b를 찾는 것
# 선형회귀는 매개변수가 없다

# 선형회귀 실습
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples = 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# 기울기 파라미터 (w)는 가중치(계수)라고 하며 lr 객체의 coef_ 속성에 저장되어 있다.
# 편향 또는 절편 파라미터 (b)는 intercept_ 속성에 저장되어 있다.
lr = LinearRegression().fit(X_train, y_train)

print("LR.coef_: ", lr.coef_)
print("LR.intercept_: ", lr.intercept_)

# intercept_ 속성은 항상 실수 값 하나
# coef_ 속성은 입력 특성에 하나씩 대응되는 numpy 배열로 이루어진다.

# 훈련 세트 점수
format(lr.score(X_train, y_train))
# 테스트 세트 점수
format(lr.score(X_test, y_test))

# 세트 점수가 약 0.66~0.67이 나오기 때문에 좋은 점수가 아니다.
# train 점수와 test 점수가 거의 비슷하기 때문에 under-fitting 상태

# 1차 데이터 셋은 모델이 매우 단순하여 over-fitting 문제 없음
# 고차원 선형 모델의 성능이 매우 높아져 over-fitting의 가능성이 높아짐

## 보스턴 주택 가격 데이터를 활용하여 고차원 데이터셋에서 회귀모델 동작 확인하기
B_X, B_y = mglearn.datasets.load_extended_boston()
BX_train, BX_text, By_train, By_test = train_test_split(B_X, B_y, random_state = 0)
B_lr = LinearRegression().fit(BX_train, By_train)

# 보스턴 주택 가격 훈련 세트 점수
format(lr.score(BX_train, By_train))
# 보스턴 주택 가격 테스트 세트 점수
format(lr.score(BX_text, By_test))

# 보스턴 주택 가격 훈련 세트 점수는 0.95로 1과 가까워 매우 정확한 값에 가깝다.
# 그러나 보스턴 주택 가격 테스트 세트 점수가 매우 낮다. 이는 over-fitting의 신호이며 복잡도를 제어해주어야 한다.
# Ridge 회귀로 자주 해결한다.

## 릿지 회귀
# 선형 모델로 예측 함수
# w 선택은 훈련 데이터를 잘 예측하기 위해서 뿐만 아니라 추가 제약 조건을 만족시키기 위한 목적도 있다.
# 가중치의 절대값을 가능한 작게 만드는 것 (0에 가깝게)
# 모든 특성이 출력에 주는 영향을 최소한으로 만들기 = 기울기를 작게 만들기

# 제약을 over-fitting을 막기 위해 강제로 제한하는 게 규제

from sklearn.linear_model import Ridge
ridge = Ridge().fit(BX_train, By_train)

# 규제한 후 보스턴 주택 가격 훈련 세트 점수
format(ridge.score(X_train, y_train))
# 규제한 후 보스턴 주택 가격 테스트 세트 점수
format(ridge.score(X_test, y_test))

# 규제한 후 보스턴 주택 가격 훈련 세트 점수는 0.89
# 규제한 후 보스턴 주택 가격 테스트 세트 점수는 0.75
# 훈련 세트 점수는 이전보다 낮아졌지만 테스트 점수는 높아졌다

# 릿지 회귀에서 규제로 인해 over-fitting이 적어졌다
# 모델이 복잡도가 낮아지면 훈련에 대한 성능은 나빠지지만 일반화된 모델이 된다.
# 테스트 데이터에 대한 예측이 목적이기 때문에 선형 회귀보다는 릿지 모델을 사용

# 릿지 모델의 alpha 기본 값은 1.0이며 값을 높이면 계수를 0에 더 가깝게 만들어서 훈련 성능은 나빠지지만 일반화에는 도움이 된다.

ridge10 = Ridge(alpha = 10).fit(BX_train, By_train)

# 규제한 후 Alpha 값 10으로 수정한 보스턴 주택 가격 훈련 세트 점수
format(ridge10.score(X_train, y_train))
# 규제한 후 Alpha 값 10으로 수정한 보스턴 주택 가격 테스트 세트 점수
format(ridge10.score(X_test, y_test))

# 위의 alpha 값 10의 훈련 세트 점수는 0.79, 테스트 세트 점수는 0.64로 일반화에 가까워진다.

# 반대로 alpha 값을 줄이게 되면 그 만큼 제약이 풀려 선형 회귀 모델과 거의 비슷해진다.
ridge01 = Ridge(alpha = 0.1).fit(BX_train, By_train)

# 규제한 후 Alpha 값 0.1로 수정한 보스턴 주택 가격 훈련 세트 점수
format(ridge10.score(X_train, y_train))
# 규제한 후 Alpha 값 0.1로 수정한 보스턴 주택 가격 테스트 세트 점수
format(ridge10.score(X_test, y_test))

# 위 처럼 좋은 성능을 낼 수도 있다. 테스트 점수의 성능이 높아질 때까지 alpha 값을 조정해야 한다.
# 높은 alpha 값은 제약이 더 많은 모델이므로 coef_의 절대값의 크기가 작을 것이 예상
# alpha 값이 클수록 계수의 크기가 0에 가깝게 분포한다.
# 반대로 alpha 값이 없는 선형 회귀의 경우 계수 크기가 범위 밖으로 넘어가기도 한다.