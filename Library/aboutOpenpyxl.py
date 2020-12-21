import openpyxl as op

# 가져올 엑셀 파일
wb = op.load_workbook('./Excel/시가총액 Top 15.xlsx')

# 가져온 엑셀 파일에서 가져올 시트
ws = wb['Sheet1']

# 꺾은 선형 차트 만들기
# data 영역, category(가로축) 영역 정확하게 지정하기
chart = op.chart.LineChart() # 차트 생성
chart.title = '주가 일별 등락 패턴' # 차트 이름
chart.x_axis.title = '날짜' # 행 이름
chart.y_axis.title = '종목별 추이' # 열 이름

chart.y_axis.scaling.min = 0.8
chart.y_axis.scaling.max = 1.4

# min_col, min_row => data 영역의 왼쪽 상단 cell의 column 인덱스 '1', row 인덱스 '2'를 입력
# max_col, max_row => data 영역의 오른쪽 하단 cell의 column 인덱스 '21', row 인덱스 '16'를 입력
datas = op.chart.Reference(ws, min_col=1, min_row=2, max_col=21, max_row=16)

# 데이터 추가
# from_rows=True => 데이터들이 행으로 작성되어 있으면 True, 열의 형태로 작성되어 있으면 False
# titles_from_data=True => 데이터의 타이틀이 포함되어 있다면 True, 아닐 경우 False
chart.add_data(datas, from_rows=True, titles_from_data=True)

# 카테고리 지정
# 차트에서 가로축으로 표현하고자 할 때
cats = op.chart.Reference(ws, min_col=2, min_row=1, max_col=21, max_row=1)
chart.set_categories(cats)

# 차트를 B2 위치에 그리기
ws.add_chart(chart, 'B2')

# 엑셀 파일로 저장하기
wb.save('차트 추가하기 결과.xlsx')
wb.close()