## Matplotlib 배우기
# 데이터를 차트, 플롯(Plot), 히스토그램 등으로 그려주는 라이브러리

import matplotlib.pyplot as plt
import numpy as np

plt.plot([1, 2, 3], [110, 130, 120])
plt.show()
# 행이 1, 2, 3에 열이 110, 130, 120에 그래프 그리기

## 제목과 축 레이블
plt.plot(['Seoul', 'Paris', 'Seattle'], [30, 25, 55]) # 행에 도시 이름, 열에 숫자
plt.xlabel('City') # 행 이름
plt.ylabel('Response') # 열 이름
plt.title('Experiment Result') # 그래프 이름
plt.show()

## 범례 추가
plt.plot([1, 2, 3], [1, 4, 9]) # Mouse
plt.plot([2, 3, 4], [5, 6, 7]) # Cat
plt.xlabel('City') # x축 레이블 이름
plt.ylabel('Response') # y축 레이블 이름
plt.title('Experiment Result') # 그래프 제목
plt.legend(['Mouse', 'Cat']) # 범례 생성
plt.show()

## 다양한 차트 및 Plot
# Bar 차트 = plt.bar()
# Pie 차트 = plt.pie()
# 히스토그램 = plt.hist()

# Bar 차트
y = [5, 3, 7, 10, 9, 5, 3.5, 8]
x = range(len(y)) # x축은 1부터 y의 갯수만큼
plt.bar(x, y, width = 0.7, color = "black") # width = Bar의 폭
plt.show()

## 히스토그램
# 도수분포표를 그래프로 나타낸 것
# 가로축은 계급, 세로축은 도수 (횟수나 개수 등)
weight = [68, 81, 64, 56, 78, 74, 61, 77, 66, 68, 59, 71, 80, 59, 67, 81, 69, 73, 69, 74, 70, 65]
plt.hist(weight)
plt.show()

# 여러 개의 히스토그램

# 표준편차 2.0, 평균 1.0
a = 2.0 * np.random.randn(10000) + 1.0

# 표준정규분포
b = np.random.standard_normal(10000)

# -10.0에서 10.0 사이의 균일한 분포를 갖는 5000개의 임의의 값
c = 20.0 * np.random.rand(5000) - 10.0

# bins => 몇 개의 영역으로 쪼갤지 설정
# density = True => 밀도 함수가 되어서 막대의 아래 면적이 1
# alpha => 투명도
# histtype = 'step' => 막대 내부가 비어있음
# histtype = 'stepfilled' => 막대 내부가 채워짐
plt.hist(a, bins = 100, density = True, alpha = 0.7, histtype = 'step')
plt.hist(b, bins = 50, density = True, alpha = 0.5, histtype = 'stepfilled')
plt.hist(c, bins = 100, density = True, alpha = 0.9, histtype = 'step')
plt.show()

