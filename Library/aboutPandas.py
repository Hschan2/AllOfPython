import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

# Series => 1차원 배열로 데이터를 담는다. 값의 리스트를 넘겨주어 만들 수 있다.
s1 = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s1)
# 0    1.0
# 1    3.0
# 2    5.0
# 3    NaN
# 4    6.0
# 5    8.0
# dtype: float64

# DataFrame => 2차원 배열로 데이터를 담는다.
# DataFrame 생성 단계 => pd.DataFrame() 클래스 생성 -> 행에 해당하는 기준 인덱스를 인수로 전달 -> 열에 해당하는 컬럼을 인수로 전달
date1 = pd.date_range('20201215', periods=6)
print(date1)
# DatetimeIndex(['2020-12-15', '2020-12-16', '2020-12-17', '2020-12-18',
#                '2020-12-19', '2020-12-20'],
#               dtype='datetime64[ns]', freq='D')

# 행에 date1의 값을, 열에 ABCD 값을
df1 = pd.DataFrame(np.random.randn(6, 4), index=date1, columns=list('ABCD'))
print(df1)
#                    A         B         C         D
# 2020-12-15  1.229760 -2.253118  1.144673  0.124581
# 2020-12-16  0.318602 -0.035023 -0.350964  1.949280
# 2020-12-17  0.167231  0.731829 -1.143348  2.445810
# 2020-12-18  1.185568 -1.259022  1.421907  2.218257
# 2020-12-19  2.862907  0.611718  1.436341 -0.021717
# 2020-12-20  0.859771  0.410132  0.232594  0.900474

df2 = pd.DataFrame({
    'A':1.,
    'B':pd.Timestamp('20201215'),
    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
    'D':np.array([3] * 4, dtype='int32'),
    'E':pd.Categorical(['test', 'train', 'test', 'train']),
    'F':'foo',
})
print(df2)
#      A          B    C  D      E    F
# 0  1.0 2020-12-15  1.0  3   test  foo
# 1  1.0 2020-12-15  1.0  3  train  foo
# 2  1.0 2020-12-15  1.0  3   test  foo
# 3  1.0 2020-12-15  1.0  3  train  foo

# Object가 가지고 있는 속성 및 메소드 확인
# print(dir(df2))


## 데이터 확인하기

# 첫 5개 행의 데이터 보기
print(df1.head())
#                    A         B         C         D
# 2020-12-15 -0.451659  0.913234 -0.432931 -0.100900
# 2020-12-16 -0.909284 -0.245025  0.201696  0.004059
# 2020-12-17  0.409492 -0.322244 -2.065413 -1.289158
# 2020-12-18  0.191783  0.181150  0.273280 -0.506518
# 2020-12-19  0.507192  0.025730  0.430157  0.270258

# 마지막 3개 행의 데이터 보기
print(df1.tail(3))
#                    A         B         C         D
# 2020-12-18  0.225282 -1.210089 -0.466037  1.029257
# 2020-12-19 -0.093995 -2.120521 -0.189545 -0.998703
# 2020-12-20  1.106462  1.288113 -2.349655 -0.415444

# 인덱스 확인하기
print(df1.index)
# DatetimeIndex(['2020-12-15', '2020-12-16', '2020-12-17', '2020-12-18',
#                '2020-12-19', '2020-12-20'],
#               dtype='datetime64[ns]', freq='D')

# 컬럼 데이터 확인하기
print(df1.columns)
# Index(['A', 'B', 'C', 'D'], dtype='object')

# numpy 데이터 확인하기
print(df1.values)
# [[-0.15118152 -0.88982484  0.89784928 -0.31859192]
#  [-1.14752024 -0.86470081 -0.7555709  -0.06073823]
#  [-0.33144986 -0.31297806  1.16147141  0.02288578]
#  [ 0.18801686  0.76864412  0.4223063  -0.56119124]
#  [-0.1105144   0.91063586 -1.83349316  0.15149821]
#  [-0.58685331 -0.23615576  0.66716901 -0.82610207]]

# 생성한 DataFrame의 간단한 통계 정보 확인
# 데이터의 개수(count), 데이터 평균 값(mean), 표준 편차(std), 최솟값(min), 4분위 수(25% 50% 75%), 최대값(max)
print(df1.describe())
#               A         B         C         D
# count  6.000000  6.000000  6.000000  6.000000
# mean   0.122083  0.042412  0.034905  0.243846
# std    1.386156  0.946791  0.964524  0.735341
# min   -1.878293 -1.143230 -0.940356 -0.428222
# 25%   -0.612067 -0.628452 -0.819706 -0.366302
# 50%    0.219710 -0.052183 -0.054761  0.032866
# 75%    0.740424  0.861298  0.748769  0.811484
# max    2.146705  1.156815  1.317383  1.253899

# 열과 행을 바꾼 형태의 DataFrame
print(df1.T)

# .T는 속성. 다음과 같이 출력하면 에러
# print(df1.T())

# 정렬. axis=0은 인덱스 기준으로 정렬(기본값), axis=1은 컬럼을 기준으로 정렬
# ascending=false는 내림차순, true는 오름차순
print(df1.sort_index(axis=1, ascending=False))

# B 컬럼 기준으로 정렬
print(df1.sort_values(by='B'))


## 데이터 선택하기

# A 컬럼의 값만 가져오기
print(df1['A'])

# A 컬럼의 타입 가져오기
print(type(df1['A']))
# <class 'pandas.core.series.Series'>

# 0 인덱스부터 3개의 행을 가져오기
print(df1[0:3])

# 각 인덱스 명에 해당하는 값 가져오기
print(df1['20201215':'20201217'])

# 특정 행을 찾기 위해 아래처럼 사용한다면 에러. 인덱스가 아니라 컬럼을 갖고 있는지 찾기 때문
# 현재 데이터 프레임에는 없기 때문에 키 값이 없다는 에러를 출력
# df1['20201216']
# 따라서 df1[컬럼명], df1[시작인덱스:끝인덱스+1], df1[시작인덱스명:끝인덱스명] 으로 사용해야 함

# 첫 번째 인덱스의 값에 해당하는 모든 컬럼 값 가져오기
# df1.loc['20201215'], df1.loc['2020-12-15'] 처럼 날짜를 직접 입력해도 가능
print(df1.loc[date1[0]])
# A    0.358800
# B   -1.423806
# C   -0.272007
# D   -0.769565
# Name: 2020-12-15 00:00:00, dtype: float64

# A와 B의 컬럼의 값 모두 가져오기
print(df1.loc[:, ['A', 'B']])

# 20201215부터 20201217의 A와 B에 해당하는 컬럼의 값 모두 가져오기
print(df1.loc['20201215':'20201217', ['A', 'B']])

# 첫 번째 인덱스에서 A, B에 해당하는 값 가져오기
print(df1.loc[date1[0], ['A', 'B']])
# print(df1.at[date1[0], 'A']) 처럼 at을 사용해도 가능

# 3번째 인덱스에 해당하는 값 선택
print(df1.iloc[3])

# 행 인덱스는 3:5로 네 번째, 다섯 번째 행 선택
# 열 인덱스는 0:2로 첫 번째, 두 번째 열 선택
print(df1.iloc[3:5, 0:2])

# 두 번째, 세 번째, 다섯 번째 행과 첫 번째 세 번째 열을 선택 (인덱스 0이 첫 번째)
print(df1.iloc[[1, 2, 4], [0, 2]])

# 행은 2개를 가져오고 열은 모든 것을 가져오기
# :를 사용하면 열 혹은 행의 전체 값을 가져오기
print(df1.iloc[1:3, :])
print(df1.iloc[:, 1:3])

# 하나의 값 선택
print(df1.iloc[1, 1])
print(df1.iat[1, 1])

# 특정 조건을 이용하여 데이터 선택
# df1의 A값이 0보다 큰 값만 출력
print(df1[df1.A > 0])

# 0보다 큰 데이터 모두 출력
print(df1[df1 > 0])

# 배열 복사
df3 = df1.copy()

# 새로운 값 넣기
df3['E'] = ['one', 'two', 'three', 'one', 'four', 'three']

# 필터링 하기
# E의 값 중에서 two, four의 값 가져오기
df3[df3['E'].isin(['two', 'four'])]


## 데이터 변경하기

# 1차원 배열로 6개의 값 생성하고 인덱스로 20201215부터 6개 생성
du1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20201215', periods=6))

# 데이터 프레임의 새로운 열을 추가할 때, 인덱스를 가진 시리즈 하나를 데이터 프레임의 열 하나를 지정하여 입력
df1['F'] = du1

# 데이터 프레임의 특정 값 하나를 선택해 다른 값으로 변경
df1.at[date1[0], 'A'] = 0

# 위치(인덱스 번호)를 이용하여 특정 값 변경 (0, 1 의 인덱스 값을 0으로 변경)
df1.iat[0, 1] = 0

# 모든 행, D열의 값을 numpy array을 사용하여 변경
# df1의 갯수만큼 5의 값을 넣어줌
df1.loc[:, 'D'] = np.array([5] * len(df1))

# df1의 배열 값을 복사하고 0보다 큰 값을 음수로 바꾸기
df3 = df1.copy()
df3[df3 > 0] = -df3


## 결측치 (Missing Data)
# 데이터를 측정하지 못해 비어있는 데이터
# pandas에서 np.nan으로 나타냄, pandas에서 결측치는 연산에서 제외

# reindex => 해당 축의 인덱스를 변경, 추가, 삭제, 복사된 데이터 프레임을 반환
df2 = df1.reindex(index=date1[0:4], columns=list(df1.columns) + ['E'])
df2.loc[date1[0]:date1[1], 'E'] = 1

# 결측치가 하나라도 존재하는 행을 버리기 (예. NaN이라는 값이 하나라도 있으면 버리기)
df1.dropna(how='any')

# 결측치의 값을 다른 것으로 채우기 (예. NaN의 값을 5로 변경)
df1.fillna(value=5)

# 결측치 유무 판단 (결측치일 때 True, 값이 있을 때 False)
print(pd.isna(df1))


## 연산

# df1의 평균 구하기 (결측치 제외)
print(df1.mean())

# 해당 인덱스 기준으로 평균 구하기
print(df1.mean(1))

# 서로 차원이 달라 인덱스를 맞추어야 할 때 두 오브젝트 간의 연산
s1 = pd.Series([1, 3, 5, np.nan, 6, 8], index=date1).shift(2)

# 인덱스 기준으로 연산 수행
df1.sub(s1, axis='index')


## 함수 적용하기

# 기존 존재하는 함수 사용해서 데이터 프레임에 적용
df1.apply(np.cumsum)

# 사용자가 정의한 람다 함수 사용 가능
df1.apply(lambda x: x.max() - x.min())

# 히스토그램 구하기
# 0부터 7까지 10개의 숫자
s2 = pd.Series(np.random.randint(0, 7, size=10))

# s2의 각 숫자의 빈도수 구하기
print(s2.value_counts())


## 문자열 관련 메소드

s3 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])

# 소문자로 변경 (결측치 제외)
print(s3.str.lower())


## 합치기 (Merging)

# 10개 행, 4개 열 랜덤숫자 생성
df4 = pd.DataFrame(np.random.randn(10, 4))

# 세 부분으로 쪼개기
pieces1 = [df4[:3], df4[3:7], df4[7:]]

# 쪼개진 부분을 다시 합치기
print(pd.concat(pieces1))


## Join

left1 = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right1 = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

# key 값 기준으로 합치기
merged = pd.merge(left1, right1, on='key')

left2 = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right2 = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

# key 값 기준으로 합치기
# foo와 bar로 다르게 key값을 주었을 때, 중복된 값을 하나로 묶어서 내보냄
merged = pd.merge(left2, right2, on='key')


## Append
# 기존 데이터 프레임의 맨 뒤에 한 번 더 추가하는 방법

df5 = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])

# 4번째 행을 기존 데이터 프레임 맨 뒤에 한 번 더 추가
s4 = df5.iloc[3]
df5.append(s4, ignore_index=True)


## 묶기 (Grouping)
# 어떠한 기준을 바탕으로 데이터 나누기 = Splitting
# 각 그룹에 어떤 함수를 독립적으로 적용 = Applying
# 적용되어 나온 결과들을 통합 = Combining

df6 = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                    'C': np.random.randn(8),
                    'D': np.random.randn(8)})

# A열 값을 기준으로 그룹을 묶고 각 그룹의 합계를 구하는 sum()함수 적용
# A열이 인덱스가 되고, C와 D열의 숫자가 합계를 구해진 데이터 프레임 생성
df6.groupby('A').sum()

# 여러 개의 열을 기준으로 이용 가능
df6.groupby(['A', 'B']).sum()


## 변형하기 (Reshaping)

# Stack(압축) 사용
tuples1 = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index1 = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df7 = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

# df7의 A, B의 열을 갖기
df8 = df7[:4]

# A와 B라는 값을 가진 인덱스 레벨을 추가한 형태로 변형
stacked1 = df8.stack()

# stack 이전 방식으로 되돌리기
stacked1.unstack()

# 첫 번째 수준(first) 해체
stacked1.unstack(0)

# 두 번째 수준(second) 해체
stacked1.unstack(1)


## Pivot Tables

df9 = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})

# D를 값으로, A와 B를 인덱스(행)으로, C를 열로 테이블 변형
# 찾지 못한 값은 NaN으로 표시
pd.pivot_table(df9, values='D', index=['A', 'B'], columns=['C'])


## 시계열 데이터 다루기

# 2020/01/01부터 1초 마다 측정된 데이터
# 금융 데이터를 다룰 때, 흔히 하는 연산
# S = second
rng1 = pd.date_range('1/1/2020', periods=100, freq='S')

ts1 = pd.Series(np.random.randint(0, 500, len(rng1)), index=rng1)

# ts1의 데이터를 5분 마다 측정된 데이터 형태로 변형
ts1.resample('5Min').sum()

# 타임존 표현
# D = day
rng2 = pd.date_range('3/6/2020 00:00', periods=5, freq='D')
ts2 = pd.Series(np.random.randn(len(rng2)), rng2)

# UTC 기준으로 변경
ts_utc1 = ts2.tz_localize('UTC')

# US/Eastern로 다른 타임존으로 변경
ts_utc1.tz_convert('US/Eastern')

# M = month
# rng3 길이만큼 랜덤으로 생성
rng3 = pd.date_range('1/1/2020', periods=5, freq='M')
ts3 = pd.Series(np.random.randn(len(rng3)), index=rng3)

# day를 빼고 출력
ps1 = ts3.to_period()

# day를 첫 번째 날로 채우기
ps1.to_timestamp()

# 11월을 끝으로 하는 4분기 체계에서 각 분기의 마지막 달에 9시간을 더한 시각을 시작으로 하는 체계로 변경
prng1 = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts4 = pd.Series(np.random.randn(len(prng)), prng)

ts4.index = (prng1.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts4.head() # 진행


## 범주형 데이터 다루기 (Categoricals)

df10 = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6],
                    'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e']})

# 단순 문자로 되어 있는 것을 범주형으로 변경
df10['grade'] = df10['raw_grade'].astype('category')

# 범주들의 이름을 변경할 수 있다. (a, b, e 대신)
df10['grade'].cat.categories = ['very good', 'good', 'bad']

# 범주의 순서를 재정렬하고 동시에 현재 갖고 있지 않는 범주 추가 가능
df10['grade'] = df10['grade'].cat.set_categories(["very bad", "bad", "medium","good", "very good"])

# 범주에 이미 매겨진 값의 순서대로 정렬
# 범주형 자료를 만들거나 재정의할 때 이루어진 순서가 범주
df10.sort_values(by='grade')

# 각 범주의 해당하는 값의 빈도수 출력
df10.groupby('grade').size()


## 그래프로 표현 (Plotting)

# 2010/01/01부터 1000개 랜덤
ts5 = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2010', periods=1000))
ts5 = ts.cumsum()

# 그래프로 생성
ts5.plot()

# 여러 개의 열을 한 번에 그릴 수 있는 그래프 생성
df11 = pd.DataFrame(np.random.randn(1000, 4), index=ts5.index, columns=['A', 'B', 'C', 'D'])
df11 = df11.cumsum()

# A, B, C, D의 4개 열에 해당하는 데이터를 legend와 함께 표시
plt.figure(); df.plot(); plt.legend(loc='best')


## 데이터 입/출력

# CSV

# 데이터 프레임을 CSV 형식으로 저장
df11.to_csv('foo.csv')

# CSV 형식의 파일 데이터 프레임 형식 읽기
# 기존 행 인덱스를 인식하지 못하고 행 인덱스를 가지는 새로운 열이 추가로 잡힘
pd.read_csv('foo.csv')


# HDF5

# HDF5형식으로 저장
df11.to_hdf('foo.h5', 'df11')

# HDF5형식의 데이터 프레임 읽어오기
pd.read_hdf('foo.h5', 'df11')


# Excel

# Excel 파일로 저장
df11.to_excel('foo.xlsx', sheet_name='Sheet1')

# Excel 파일의 데이터 프레임 읽기
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

## 엑셀 파일 읽기
# xlsx 파일을 불러오려면 engine = 'openpyxl' 필수
# xlsx 파일 읽기
kospi_df = pd.read_excel('./Excel/KOSPI상장회사.xlsx', engine = 'openpyxl')

# 업종이 통신 및 방송 장비 제조업인 데이터 가져오기
industry = kospi_df[kospi_df['업종'] == '통신 및 방송 장비 제조업']
# print(industry)

# 업종이 통신 및 방송 장비 제조업이거나 음식점업인 데이터 가져오기
industry_Food = kospi_df[(kospi_df['업종'] == '통신 및 방송 장비 제조업') | (kospi_df['업종'] == '음식점업')]
print(industry_Food)

# 종목코드 기준으로 오름차순
kospi_df.sort_value(by = '종목코드', ascending = True)

price_df = pd.read_excel('./Excel/KOSPI시가총액.xlsx', engine = 'openpyxl')

# VLOOKUP 하기 (기준이 될 열이 필요)
# set_index에 기준 열에 사용할 이름을 입력
kospi_df.set_index('기업명', inplace = True)

# 시가총액이라느 ㄴ열을 만들어 VLOOKUP 결과 저장
kospi_df['시가총액'] = price_df['상장시가총액(원)']

## 피벗 테이블
item_df = pd.read_excel('./Excel/월별 보유 수량.xlsx', engine='openpyxl')

# 피벗 테이블 만들기
# 값 = 보유 수량, 행 = 지역, 열 = 제조사
FirstPivot = pd.pivot_table(item_df, values='보유 수량', index='지역', columns='제조사', aggfunc=sum)
# print(FirstPivot)

# 행을 두 가지 항목으로 지정하고 싶을 때
SecondPivot = pd.pivot_table(item_df, values='보유 수량', index=['월', '지역'], columns='제조사', aggfunc=sum)
print(SecondPivot)

# sum 외
# min = 최소값, max = 최대값, mean = 평균값, count = 개수