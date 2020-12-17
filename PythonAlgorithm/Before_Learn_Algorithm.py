# 문자열 함수 (모르는 것 위주)

## 공백 제거
x = ' Python Algorithm! '

# 왼쪽 공백 제거
x.lstrip()

# 오른쪽 공백 제거
x.rstrip()

# 양쪽 공백 제거
x.strip()

## 슬라이싱
a = [1, 2, 3, 4, 5]

# 처음부터 2개 (index - 1, 2 - 1 = 1 => 0, 1)
a[:2]

# 2개 이후부터 끝까지 (인덱스 0, 1 이후 전체)
a[2:]

# 전체
a[:]

# 리스트 3번 반복
a * 3

# 리스트 일부 삭제, a의 2번째 인덱스 삭제
del a[2]

## 리스트 관련 함수

# pos 위치에 val 삽입
pos = 4
val = '4번째'
a.insert(pos, val)

# 마지막 원소 꺼내기
a.pop()

## 튜플
# 리스트와 달리 값을 바꿀 수 없다 (변경, 삭제 불가)
b = (1, 2, 3, 4, 5)

## 집합
c = set()

# 출력 시 중복 없음
d = set([1, 2, 3, 4, 5, 1, 2, 3])

# 교집합
c & d
c.intersection(d)

# 합집합
c | d
c.union(d)

# 차집합
c - d
c.difference(d)
d - c

## 전역 변수
global result

## 람다 함수
def makeMyFunc(a):
    return lambda x: a * x
myFunc = makeMyFunc(3)
myFunc(4)
# 결과 = 12

## 파일 생성/열기
f = open('txt파일', 'w') # 파일 열기, 'r' 파일 읽기, 'a' 파일 추가, 'rb' 바이너리 읽기 모드, 'wb' 바이너리 쓰기 모드
f.close() # 파일 닫기

all = f.read() # 한 번에 모두 읽어 들이기

f.write('내용') # 파일에 쓰기

## 클래스
class Dog:
    def __init__(self, name, age): # 생성자 메서드
        self.name = name
        self.age = age
    def __del__(self): # 소멸자 메서드
        print('삭제')
    def bark(self): # 멤버 메서드
        print(self.name + ' is barking')

dog1 = Dog('Baduk', 2)

## 상속
class Pet:
    def __init__(self, name):
        self.name = name
class Cat(Pet): # Pet을 상속 받기
    def meow(self):
        print(self.name + ' is meowing')

## 예외처리
    # try:

    # except:
    
    # try:

    # except<특정 예외 처리>:
    
    # try:

    # except<특정 예외 처리> as <에러 변수>: # 변수로 에러 메세지 받기
    
    # try:
    # except:
    # else: # 예외가 발생하지 않은 경우
    # finally: # 예외 발생 여부와 상관 없이 수행

try:
    result = 123 / x
except ZeroDivisionError as err: # 0으로 나누려고 할 때, 에러
    print(err)
else:
    print(result)
finally:
    print('끝')

try:
    result = x.index(1234)
except ValueError as err: # 리스트의 Index 찾기 실패
    print(err)
else:
    print(result)
finally:
    print('끝')

