import json

# JSON with Python

# JSON = XML보다 더 자주 사용되는 데이터 포맷, 데이터를 읽어온 후 딕셔너리로 접근 가능
#        데이터를 시스템에서 추출하여 두 시스템 사이에서 간단하게 이동할 수 있게 해줌, Python 문법과 매우 유사

data = '''{
    "name": "Hong",
    "phone": {
        "type": "int1",
        "number": "+10 734 303 4456"
    },
    "email": {
        "hide": "yes"
    }
}
'''

# loads => load String
info = json.loads(data)
# print(info["name"])
print(info["email"]["hide"])

input = '''[
  { "id" : "001",
    "x" : "2",
    "name" : "Chuck"
  } ,
  { "id" : "009",
    "x" : "7",
    "name" : "Hong"
  }
]'''

inputInfo = json.loads(input)
print(len(inputInfo)) # 딕셔너리 개수

# 딕셔너리 개수만큼 반복
for item in inputInfo:
    print(item["name"])

    