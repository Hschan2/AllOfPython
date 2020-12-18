# Excel 파일 읽기
import csv

with open('./Excel/sample1.csv', 'r') as f:
    reader = csv.reader(f)
    # reader = csv.reader(f, delimiter = "|") => |를 기준으로 나누다
    # reader = csv.DictReader(f) => 딕셔너리 형태로 변환

    print(reader)
    print(type(reader))
    print(dir(reader)) # __init__ => for문 가능

    for txt in reader: # Excel 파일 내 데이터 가져오기
        print(txt)

    # 딕셔너리 형태에서 각각의 값들 빼내오기
    for c in reader:
        # k = key, v = value
        for k, v in c.item():
            print(k, v)
        print("===================")