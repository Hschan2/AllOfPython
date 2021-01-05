import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Seaborn => matplotlib를 이용한 데이터 시각화, 패키지
# 기존 라이브러리에서 제공하는 데이터 바탕으로 학습

# tips 데이터셋 불러오기
tips = sns.load_dataset('tips')
tipsData = tips.head(6)

# 수채화 데이터 셋
iris = sns.load_dataset('iris')
# 타이타닉 데이터 셋
titanic = sns.load_dataset('titanic')
# 비행기 탑승자 데이터 셋
flights = sns.load_dataset('flights')

## Regplot
# 데이터를 점으로 나타내고 선형성을 함께 확인

# 지불 금액에 따른 팁의 양
ax = sns.regplot(x = 'total_bill', y = 'tip', data = tips)
ax.set_xlabel('TB')
ax.set_ylabel('Tip')
ax.set_title('Total bills and tips')

## Jointplot
# 안쪽은 점 데이터로 분포 확인, 바깥쪽은 막대그래프로 밀집도 확인
# 데이터의 경향 파악

# kind = 'hex' 옵션을 주게 되면 6각 분포로 출력
joint = sns.jointplot(x = 'total_bill', y = 'tip', data = tips)

# regplot과 라벨 붙이는 방식이 다름
# joint에서 set_axis_labels => 라벨 설정
# 라벨 설정 없이 기본 Column 명으로 라벨 형성 가능
joint.set_axis_labels(xlabel = 'TB', ylabel = 'Tip')

## kde
# 이차원 밀집도 그래프
# 등고선 형태로 밀집 정도를 확인 가능

kde, kdeAx = plt.subplots()
# 데이터 가져오기
kdeAx = sns.kdeplot(data = tips['total_bill'],
           data2 = tips['tip'],
           shade = True)
kdeAx.set_title('Kernel Density Plot')

## Barplot
# 막대 그래프

barAx = plt.plot()
barAx = sns.barplot(x = 'time', y = 'total_bill', data = tips)

## Boxplot
# 박스 그래프

# hue = 옵션으로 카테고리별 비교 가능
# palette = 색상 설정 가능
boxAx = sns.boxplot(x = 'day', y = 'total_bill', data = tips, hue = 'smoker', palette = 'Set3')

## Pairplot(data = tips)
# 수치에 해당하는 그래프를 전반적으로 출력
# 관계 그래프를 확인 가능
# 처음 데이터를 확인할 때, 전체 파악하기 좋음

# tips의 데이터를 가진 그래프 전체를 전반적으로 확인
sns.pairplot(data = tips)

# 원하는 그래프와 데이터로 pairplot 그리기
pg = sns.PairGrid(tips) # pairgrid 형태로 생성
pg.map_upper(sns.regplot) # 위쪽 그래프에 넣을 plot
pg.map_lower(sns.kdeplot) # 아래쪽 그래프에 넣을 plot
pg.map_diag(sns.distplot) # 가운데 그래프에 넣을 plot

## Countplot
# 해당 카테고리 별 데이터의 개수 보여주는 그래프

# 요일별 팁 받은 횟수 출력
# 요일별 그래프 색상을 다르게 출력
dayCount = sns.countplot(x = 'day', data = tips)

## Heatmap
# 카테고리별 데이터 분류

# aggfunc = 'size' = 각 데이터의 건수
# pivot_table = 평균, 분산 등 결과가 출력
titanic_size = titanic.pivot_table(index = 'class', columns = 'sex', aggfunc = 'size')

titanic_size

# annot = True => 숫자가 출력
# fmt = 'd' => 지수 형태의 숫자 형태가 아닌 것을 지수 형태의 숫자로 변경
# sns.light_palette => 색상 설정
sns.heatmap(titanic_size, annot = True, fmt = 'd', cmap = sns.light_palette('red'))

flights.head()

# 열인덱스, 행인덱스, 데이터
fp = flights.pivot('month', 'year', 'passengers')
sns.heatmap(fp, linewidths = 1, annot = True, fmt = 'd')

## pandas에서 바로 plot 그리기
# 랜덤으로 100행 3열 생성
# 2020.01.28일 기준으로 100일
df = pd.DataFrame(np.random.randn(100, 3), index = pd.date_range('1/28/2020', periods = 100), columns = ['A', 'B', 'C'])

# 일변 변화량 혹은 변동폭을 그래프로 나타내기
df.plot()
# cumsum() = 누적합, 누적합으로 그래프 긜 시 열의 값이 어떻게 변하는지 확인 가능
# 금융, 주식 등에서 수익률 계산 때 활용
df.cumsum().plot()

## pie 그래프
df = titanic.pclass.value_counts()
df.plot.pie(autopct = '%.2f%%') # 소수점 2자리까지 출력