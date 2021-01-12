import pandas as pd
import sweetviz as sv
import pandas_profiling

df = pd.read_csv("D:\\Bitnami\\wampstack-7.1.20-1\\apache2\\htdocs\\AllofPython\\Excel\\titanic.csv")
etc = pd.read_csv("D:\\Bitnami\\wampstack-7.1.20-1\\apache2\\htdocs\\AllofPython\\Excel\\sample1.csv")

### sweetviz ###

## 데이터 시각화
# advert_report = sv.analyze(df)
# advert_report.show_html('./sweetviz_Advertising.html')

## Dataset 비교
# df1 = sv.compare(df.sample(600), df.sample(100))
# df1.show_html('./sweetviz_Comapre.html')

## 마지막 인자에는 Boolean형식 데이터 프레임 값
# df1 = sv.compare([df.sample(600), "Item"], [df.sample(100), "Test"], "SibSp")
# df1.show_html('./sweetviz_Comapre_Target.html')

## 두 개의 csv 파일 비교
# df1 = sv.compare([df, "titanic"], [etc, "sample1"], "SibSp")
# df1.show_html('./sweetviz_Comapre_Titanic.html')


### pandas_profiling ###

# 리포트 생성
pr = df.profile_report()
# 리포트를 html 파일로 생성
# pr.to_file('./pr_report.html')

# 리포트 살펴보기
pr

## 변수
# 데이터에 존재하는 모든 특성 변수들에 대한 결측값, 중복 제외한 유일한 값의 개수 등 통계치 보여주기
# 상위 5개의 값에 대해 우측 바 그래프로 시각화한 결과값 제공

## toggle
# 상세한 정보 확인 가능
# 빈도 값 확인 가능
# 전체 값의 최대 길이, 최소 길이, 평균 길이 등 값의 구성에 대해 확인 가능
# 중복이 존재하는 상위 10개의 내용 확인 가능