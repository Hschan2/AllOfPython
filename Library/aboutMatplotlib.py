import matplotlib.pyplot as plt
# 3D 그래프 그리기 라이브러리
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

## Y값 입력하기(설정), Y 라벨 타이틀 설정
plt.plot([1, 2, 3, 4])
plt.ylabel('y-label')
plt.show()

## X열, Y열 설정
# (1, 1), (2, 4), (3, 9), (4, 16)를 잇는 그래프 설정
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
# x축 라벨
plt.xlabel('x-label')
# y축 라벨
plt.ylabel('y-label')
plt.show()

## X열, Y열, 스타일 설정
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
# 축의 범위 [xmin, xmax, ymin, ymax]를 지정
plt.axis([0, 6, 0, 20])
plt.show()

## 0.2씩 간격으로 균일하게 설정
t = np.arange(0., 5., 0.2)
# 첫 번째 그래프 - 빨간 대쉬
# 두 번째 그래프 - 파란 사각형
# 세 번째 그래프 - 녹색 삼각형
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# 선/마커 표시 형식
# 'b' = 파란색 기본 모양
# 'ro' = 빨간색 원형
# 'g-' = 초록색 사각형
# '--' = 기본 색상 대시
# 'k^:' = 검은색 연결된 삼각형

# color 색상 지정하기
# r = red
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], color='r')

# 색상과 마커, 마커스타일 지정하기
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], color='#e35f62', marker='o', linestyle='--')

## 두 그래프 사이 영역 채우기
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]

plt.plot(x, y)
plt.xlabel('x-label')
plt.ylabel('y-label')
# (x[1], y[1]), (x[2], y[2]), (x[1], 0), (x[2], 0)을 잇는 영역 채우기
plt.fill_between(x[1:3], y[1:3], alpha=0.5)

#  (x[2], y[2]), (x[3], y[3]), (0, y[2]), (0, y[3])을 잇는 영역 채우기
plt.fill_betweenx(y[2:4], x[2:4])

## 두 그래프 사이 영역 채우기 예제
x = [1, 2, 3, 4]
y1 = [1, 4, 9, 16]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
# (x[1], y[1]), (x[1], y[2]), (x[2], y[1]), (x[2], y[2]) 사이 영역 채우기
plt.fill_between(x[1:3], y1[1:3], y2[1:3], color='lightgray', alpha=0.5)

plt.show()

## 임의의 영역 채우기
x = [1, 2, 3, 4]
y1 = [1, 4, 9, 16]
y2 = [1, 2, 4, 8]

plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
# fill => x, y점들로 정의되는 다각형 영역을 자유롭게 지정하여 채우기
plt.fill([1.9, 1.9, 3.1, 3.1], [2, 5, 11, 8], color='lightgray', alpha=0.5)

plt.show()

## 여러 곡선 그리기
# 0부터 2까지 0.2씩 범위 만들기
a = np.arange(0, 2, 0.2)

# 3개의 그래프 만들기
plt.plot(a, a, 'r--', a, a**2, 'bo', a, a**3, 'g-.')
plt.show()

## 스타일 지정하기
a = np.arange(0, 2, 0.2)

# 첫 번째 그래프, 파란색 원형
plt.plot(a, a, 'bo')
# 두 번째 그래프, #e35f62 색의 * 마커로, 선 굵기 2
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
# 세 번째 그래프, springgreen 색의 삼각형 마커로, 마커 크기 9
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.show()

## 그리드 만들기
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# 생성한 그래프에 그리드 생성하기
plt.grid(True)

plt.show()

## 축 지정하기
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# axis='y' = y축 그리드만 그리기
plt.grid(True, axis='y')

plt.show()

## 그리드 스타일 설정
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# 빨간색, --스타일, 투명도 0.5, y축의 그리드 그리기
plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')

plt.show()

## 눈금 그리기
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# x축 눈금 그리기, 0 ~ 2까지
plt.xticks([0, 1, 2])
# y축 눈금 그리기, 1 ~ 5까지
plt.yticks(np.arange(1, 6))

plt.show()

## 눈금 레이블 지정하기
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# x축 눈금 그리기
# 0부터 1.8까지 0.2씩 증가
# x축 눈금 라벨 이름 설정
plt.xticks(np.arange(0, 2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
# y축 눈금 그리기
# 0부터 6까지
# y축 눈금 라벨 이름 설정
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

plt.show()

## 눈금 스타일 설정
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
plt.xticks(np.arange(0, 2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

# tick_params = 눈금 스타일 다양하게 설정 가능
# direction = 눈금이 안/밖으로 설정 (in, out, inout)
# length = 눈금 길이
# pad = 눈금과 레이블과의 거리
# labelsize = 레이블 크기 지정
# labelcolor = 레이블 색상 지정
# top/bottom/left/right = 눈금이 표시될 위치 선택
# width = 눈금 너비 지정
# color = 눈금 색상 지정
plt.tick_params(axis='x', direction='in', length=3, pad=6, labelsize=14, labelcolor='green', top=True)
plt.tick_params(axis='y', direction='inout', length=10, pad=15, labelsize=12, width=2, color='r')

plt.show()

## 그래프 타이틀 지정
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# 그리드 설정
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
# 눈금 설정
plt.tick_params(axis='both', direction='in', length=3, pad=6, labelsize=14)
# 그래프 이름 Sample graph로 지정
plt.title('Sample graph')

plt.show()

## 그래프 제목 위치와 오프셋 지정
# 제목: Sample graph, 위치: right, 타이틀과 그래프와의 간격 = 20
plt.title('Sample graph', loc='right', pad=20)

# 그래프 제목 폰트 지정
plt.title('Sample graph', loc='right', pad=20)

title_font = {
    'fontsize': 16,
    'fontweight': 'bold'
}
# fontdict = 그래프 제목 폰트 지정
plt.title('Sample graph', fontdict=title_font, loc='left', pad=20)

## 수직선/수평선 표시하기 (axhline/axvline)
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# x축 눈금 그리기
plt.xticks(np.arange(0, 2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
# y축 눈금 그리기
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

# y 값의 수평선 위치 그리기
# 첫 번째 인자 = y 값으로서 수평선 위치
# 두 번째 인자, 세 번째 인자 = xmin, xmax (0은 y축 왼쪽 끝, 1은 y축 오른쪽 끝)
plt.axhline(1, 0, 0.55, color='gray', linestyle='--', linewidth='1')
# x 값의 수평선 위치 그리기
# 첫 번째 인자 = x 값으로서 수평선 위치
# 두 번째 인자, 세 번째 인자 = ymin, ymax (0은 x축 왼쪽 끝, 1은 x축 오른쪽 끝)
plt.axvline(1, 0, 0.16, color='lightgray', linestyle=':', linewidth='2')

# y 값의 수평선 위치 그리기
plt.axhline(5.83, 0, 0.95, color='gray', linestyle='--', linewidth='1')
# x 값의 수평선 위치 그리기
plt.axvline(1.8, 0, 0.95, color='lightgray', linestyle=':', linewidth='2')

plt.show()

## 수직선/수평선 표시하기 (hlines/vlines)
a = np.arange(0, 2, 0.2)

plt.plot(a, a, 'bo')
plt.plot(a, a**2, color='#e35f62', marker='*', linewidth=2)
plt.plot(a, a**3, color='springgreen', marker='^', markersize=9)
# x축 눈금 생성
plt.xticks(np.arange(0, 2, 0.2), labels=['Jan', '', 'Feb', '', 'Mar', '', 'May', '', 'June', '', 'July'])
# y축 눈금 생성
plt.yticks(np.arange(0, 7), ('0', '1GB', '2GB', '3GB', '4GB', '5GB', '6GB'))

# y축 수직선/수평선 그리기
# 인자 = y, xmin, xmax
plt.hlines(4, 1, 1.6, colors='pink', linewidth=3)
# x축 수직선/수평선 그리기
# 인자 = x, ymin, ymax
plt.vlines(1, 1, 4, colors='pink', linewidth=3)

plt.show()

## 막대 그래프 그리기
x = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

# bar = 막대 그래프 그리기
# x축 기준
plt.bar(x, values)
# x축 눈금 그리기
plt.xticks(x, years)
plt.show()

## 막대 그래프 스타일 꾸미기
x = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

# width = 막대 너비
# align = tick과 막대 위치 조절. edge = 막대의 왼쪽 끝
# color = 막대 그래프 색
# edgecolor = 막대 그래프 테두리 색
# linewidth = 테두리의 두께
# tick_label = array 형태로 지정하면 tick의 어레이 문자열을 순서대로 나타냄
# log = true => y축이 로그 스케일로 표시
plt.bar(x, values, width=0.6, align='edge', color="springgreen",
        edgecolor="gray", linewidth=3, tick_label=years, log=True)
plt.show()

## 수평 막대 그래프
y = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

# barh() => 수평 막대 그래프 생성
# 스타일 꾸미는 방식은 기존 막대 그래프와 같음
# log = false, x축 기준
plt.barh(y, values, height=-0.6, align='edge', color="springgreen",
        edgecolor="gray", linewidth=3, tick_label=years, log=False)
plt.show()

## 산점도 그리기
# 난수 생성, 같은 난수를 재사용할 수 있음
# seed = 0~4294967295
np.random.seed(19680801)

N = 50
# x축 위치 랜덤으로 지정
x = np.random.rand(N)
# y축 위치 랜덤으로 지정
y = np.random.rand(N)
# 색 랜덤으로 지정
colors = np.random.rand(N)
# 면적 랜덤으로 지정
area = (30 * np.random.rand(N))**2

# 산점도 그래프 그리기
# 투명도 50%
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

## 3D 산점도 그래프 그리기
n = 100
# x = 0 ~ 20
# y = 0 ~ 20
# z = 0 ~ 50
xmin, xmax, ymin, ymax, zmin, zmax = 0, 20, 0, 20, 0, 50
# c = 0 ~ 2
cmin, cmax = 0, 2
# 각 범위 안에서 임의의 실수 생성
xs = np.array([(xmax - xmin) * np.random.random_sample() + xmin for i in range(n)])
ys = np.array([(ymax - ymin) * np.random.random_sample() + ymin for i in range(n)])
zs = np.array([(zmax - zmin) * np.random.random_sample() + zmin for i in range(n)])
color = np.array([(cmax - cmin) * np.random.random_sample() + cmin for i in range(n)])
# 아래처럼 동일하게 설정 가능
# xs = (xmax - xmin) * np.random.rand(n) + xmin
# ys = (xmax - xmin) * np.random.rand(n) + ymin
# zs = (xmax - xmin) * np.random.rand(n) + zmin
# color = (xmax - xmin) * np.random.rand(n) + cmin

# rcParams을 이용해 figure 사이즈 설정
plt.rcParams["figure.figsize"] = (6, 6)
fig = plt.figure()
# 3D axes 만들기
ax = fig.add_subplot(111, projection='3d')
# cmap = colorMap을 녹색 계열로 설정
ax.scatter(xs, ys, zs, c=color, marker='o', s=15, cmap='Greens')
plt.show()

## 히스토그램 그리기 (도수분포표를 그래프로 나타낸 것, 가로축은 계급, 세로축은 도수 (횟수나 개수 등)
weight = [68, 81, 64, 56, 78, 74, 61, 77, 66, 68, 59, 71, 80, 59, 67, 81, 69, 73, 69, 74, 70, 65]

# hist = 히스토그램 그리기
plt.hist(weight)
plt.show()

## 여러 개의 히스토그램 그리기
# a, b, c에 임의의 값 만들기
a = 2.0 * np.random.randn(10000) + 1.0
b = np.random.standard_normal(10000)
c = 20.0 * np.random.rand(5000) - 10.0

# a, b, c 히스토그램 그리기
# bins = 몇 개의 영역으로 쪼갤지 설정
# density = True = 밀도 함수가 되어서 막대 아래 면적이 1
# histtype => step = 막대 내부가 비어있음. stepfilled = 막대 내부 채워짐
plt.hist(a, bins=100, density=True, alpha=0.7, histtype='step')
plt.hist(b, bins=50, density=True, alpha=0.5, histtype='stepfilled')
plt.hist(c, bins=100, density=True, alpha=0.9, histtype='step')
plt.show()

## 에러바 그리기
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
yerr = [2.3, 3.1, 1.7, 2.5]

# 에러바 그리기
# yerr = 데이터 편차를 나타내는 리스트를 그리기
# yerr의 각 값들은 데이터 포인트의 위/아래 대칭인 오차로 표시
plt.errorbar(x, y, yerr=yerr)
plt.show()

## 비대칭 편차 나타내기
x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
# 비대칭 편차를 그리기 위해
# (2, N) 형태의 값들로 입력, N: 데이터 개수
# 첫 번째 튜플 값은 아래 방향 편차, 두 번째 튜플의 값은 위 방향 편차
yerr = [(2.3, 3.1, 1.7, 2.5), (1.1, 2.5, 0.9, 3.9)]

plt.errorbar(x, y, yerr=yerr)
plt.show()

## 상한/하한 기호 표시하기
# 1 ~ 4까지
x = np.arange(1, 5)
# 1 ~ 4까지의 제곱
y = x**2
yerr = np.linspace(0.1, 0.4, 4)

plt.errorbar(x, y + 4, yerr=yerr)
# uplims = True = 상한 설정, False일 경우 상한 표시 X
# lolims = True = 하한 설정, False일 경우 하한 표시 X
plt.errorbar(x, y + 2, yerr=yerr, uplims=True, lolims=True)

# Array 형식으로 상한/하한 표시하기
upperlimits = [True, False, True, False]
lowerlimits = [False, False, True, True]
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits)
plt.show()

## 파이 차트 만들기
# 원 그래프, 부채꼴의 중심각을 구성 비율에 비례
# 각 값과 레이블을 설정
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']

# pie = 파이 차트 (원 그래프) 만들기
# autopct = 부채꼴 안에 표시될 숫자의 형식을 지정
# %.1f%% = 소수점 한 자리까지 표시
plt.pie(ratio, labels=labels, autopct='%.1f%%')
plt.show()

## 파이 차트 시작 각도와 방향 설정하기
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']

# startangle = 부채꼴이 그려지는 시작 각도 설정, Default = 0 (양의 방향 x축)
# counterclock = False = 시계 방향 순서로 부채꼴 영역 표시, True = 반시계 방향
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False)
plt.show()

## 파이 차트 중심에서 벗어나는 정도 설정하기
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
# explode = 각 부채꼴이 파이 차트의 중심에서 벗어나는 정도
explode = [0, 0.10, 0, 0.10]

# explode = 각 부채꼴이 파이 차트의 중심에서 벗어나는 정도 설정
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, explode=explode)
plt.show()

## 파이 차트 그림자 나타내기
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
explode = [0.05, 0.05, 0.05, 0.05]

# shadow = True = 그림자 나타내기
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, explode=explode, shadow=True)
plt.show()

## 파이 차트 부채꼴마다 색상 지정하기
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
explode = [0.05, 0.05, 0.05, 0.05]
# 각 부채꼴마다 색상 지정하기
colors = ['silver', 'gold', 'whitesmoke', 'lightgray']

# 색상 Array를 파이 차트 colors에 지정하기
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, explode=explode, shadow=True, colors=colors)
plt.show()

ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
explode = [0.05, 0.05, 0.05, 0.05]
# Hex code로 색상 지정 가능
colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']

plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, explode=explode, shadow=True, colors=colors)
plt.show()

## 파이 차트 부채꼴 스타일 지정하기
ratio = [34, 32, 16, 18]
labels = ['Apple', 'Banana', 'Melon', 'Grapes']
colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']
# wedgeprops = 부채꼴 영역의 스타일
# width =  부채꼴 영역 너비 (반지름에 대한 비율)
# edgecolor = 테두리 색상
# linewidth = 테두리 선 너비
wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

# wedgeprops = 부채꼴 영역의 스타일 지정
plt.pie(ratio, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
plt.show()