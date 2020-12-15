# numpy => 행렬 관련 계산을 빠르게 와주는 패키지, ndarray 넘파이 전용 리스트를 사용하는데 쉽게 해주는 행렬
# ndarray => N차원의 배열 객체, 오직 같은 종류의 데이터만 배열에 담을 수 있다

import numpy as np

li = [1, 2, 3, 4, 5, 6]
arr1 = np.array(li) # ndarray로 변환
print(arr1) # [1 2 3 4 5 6]

arr2 = np.arange(1, 8)
print(arr2) # [1 2 3 4 5 6 7]

# 다중 배열 생성
arr3 = np.zeros((2, 3))
print(arr3)
# [[0. 0. 0.]
#  [0. 0. 0.]]

arrOne = np.ones((2, 3)) # 성분이 1로 채워진 배열 생성
print(arrOne)

# 0대신 다른 값 넣는 함수
arr4 = np.full((2, 3), 1)
print(arr4)
# [[1 1 1]
#  [1 1 1]]

# 단위 행렬 (주대각성분이 1이고 나머지는 0) 생성 함수
arr5 = np.eye(2)
print(arr5)
# [[1. 0.]
#  [0. 1.]]

# 사칙 연산 + 함수
add = np.add(1, 2)
print(add)
# 3

# 사칙 연산 - 함수
substr = np.subtract(5, 1)
print(substr)
# 4

# 사칙 연산 * 함수
multi = np.multiply(2, 3)
print(multi)
#6

# 사칙 연산 / 함수
divide = np.divide(4, 2)
print(divide)
# 2.0

# 행렬 곱
dots = np.dot(2, 3)
print(dots)
# 6

# 행렬의 합
sums = np.sum([2, 3])
print(sums)
# 5

# 행렬의 곱
prods = np.prod([2, 3])
print(prods)
# 6

# 최대값
max = np.max([2, 3, 5])
print(max)
# 5

# 최소값
min = np.min([2, 3, 5])
print(min)
# 2

argmax = np.argmax([2, 3, 5])
print(argmax)
# 2, 2번째 인덱스가 가장 큰 숫자다

argmin = np.argmin([2, 3, 5])
print(argmin)
# 0, 0번 째 인덱스가 가장 작은 숫자다

# ndarray의 차원을 반환
data1 = np.arange(1, 6)
data1 = data1.shape # shape는 함수가 아니기 때문에 ()를 사용하지 않아도 된다.
print(data1)
# (5, ) => (1 2 3 4 5)

# reshape => 행렬의 차원을 바꾸는 데에 사용. -1를 넣어 피는 용도로 자주 사용하기도 함. 여러 차원을 넣을 때는 꼭 ()로 묶어주어야 함
data2 = np.arange(1, 7)
data2 = data2.reshape((2, 3)) # np.reshape(data2, (2, 3)) 으로 대체 가능
print(data2)
# [[1 2 3]
#  [4 5 6]]

data3 = np.arange(1, 7)
data3 = data3.reshape(-1) # -1를 넣을 시, 하나의 행렬로 쭉 펴서 출력
print(data3)
# [1 2 3 4 5 6]

data4 = np.arange(1, 7)
data4 = data4.reshape((3, -1)) # 3행으로 (열은 알아서 생성)
print(data4)
# [[1 2]
#  [3 4]
#  [5 6]]

# np.transpose() => 행렬의 전치 행렬을 구하는 것, 3차원 이상의 행렬에서 우선 순위를 정해줄 수 있음
data5 = np.arange(1, 7)
data5 = data5.reshape((2, 3, 1))
print(data5.shape)
# (2, 3, 1)
data6 = data5.transpose((0, 2, 1))
print(data6.shape)
# (2, 1, 3)

# np.random.seed() => 랜덤이 들어간 함수에서 함수에서 항상 같은 결과를 표출하기 위해서 사용. 모델끼리 성능비교할 때 항상 같은 숫자를 넣은 것이 좋음
random = np.random.seed(42)
print(random)
# 랜덤이 들어간 함수가 아니기에 None

# 5개의 숫자 무작위 생성
rand = np.random.randn(5)
print(rand)
# [ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]

# 랜덤한 정수를 생성, min과 sup 지정도 가능
randint = np.random.randint(0, 10, 5) # 0이상 10 미만의 숫자 5개 무작위 생성
print(randint)
# [7 4 3 7 7]

names = np.array(["Charles", "Kilho", "Hayoung", "Charles", "Hayoung", "Kilho", "Kilho"])
datas = np.array([[ 0.57587275, -2.84040808,  0.70568712, -0.1836896 ],
                [-0.59389702, -1.35370379,  2.28127544,  0.03784684],
                [-0.28854954,  0.8904534 ,  0.18153112,  0.95281901],
                [ 0.75912188, -1.88118767, -2.37445741, -0.5908499 ],
                [ 1.7403012 ,  1.33138843,  1.20897442, -0.58004389],
                [ 1.11585923,  1.02466538, -0.74409379, -1.55236176],
                [-0.45921447,  2.53114818,  0.5029578 , -0.24088216]])

print(names == "Charles") # 이런 식으로 생성되는 boolean Array는 Mask라고 불린다
# [ True False False  True False False False] => Charles 인덱스에는 True

print(datas[names == "Charles", :]) # Charles의 이름을 가진 인덱스의 datas 값을 출력
# [[ 0.57587275 -2.84040808  0.70568712 -0.1836896 ]
#  [ 0.75912188 -1.88118767 -2.37445741 -0.5908499 ]]

print(datas[(names == "Charles") | (names == "Kilho"), :]) # Charles 또는 Kilho의 값을 가진 인덱스의 datas 값을 출력
# [[ 0.57587275 -2.84040808  0.70568712 -0.1836896 ]
#  [-0.59389702 -1.35370379  2.28127544  0.03784684]
#  [ 0.75912188 -1.88118767 -2.37445741 -0.5908499 ]
#  [ 1.11585923  1.02466538 -0.74409379 -1.55236176]
#  [-0.45921447  2.53114818  0.5029578  -0.24088216]]

print(datas[:, 3]) # 4번째 열의 값을 출력 (0, 1, 2, 3)의 4번째는 3
print(datas[:, 3] < 0) # 4번째 열의 값이 0보다 작은 값을 boolean으로 출력

x = np.array([ 0.97121145,  1.74277758,  0.17706708, -1.14078851,  1.02197222,
                     -0.75747493,  0.4994057 , -0.03462392])
y = np.array([ 0.91984849,  1.98745872, -0.11232596,  1.47306221,  1.24527437,
                   -0.77047603,  0.30708743, -1.76476678])

print(np.maximum(x, y)) # x와 y 배열을 비교해서 각 인덱스마다 큰 값을 출력

sorted = np.array([-0.21082527, -0.0396508 , -0.75771892, -1.9260892 , -0.18137694,
                   -0.44223898,  0.32745569,  0.16834256])

print(np.sort(sorted)) # 오름차순 정렬
print(np.sort(sorted, axis=0)) # axis = 0 => 0일 때 행 방향으로 오름차순 정렬, 1일 때 열 방향으로 오름차순 정렬
print(np.sort(sorted)[::-1][int(0.05 * len(sorted))]) # 상위 5%에 위치하는 값 가져오기

# 그 외 데이터형
# int8, int16, int32, int64 => 부호가 있는 비트 정수
# uint8, uint16, uint32, uint64 => 부호가 없는 비트 정수
# float16, float32, float64, float128 => 비트 실수
# complex64, complex128, complex256 => 비트 복소수
# bool => True, False
# object => Python 오브젝트 형
# string_ => 문자열
# unicode_ => 유니코드 문자열