## 스택
x = 1 # 리스트에 넣을 값 (예. x = 1)

st = []
st.append(x) # 리스트 맨 뒤에 데이터 추가
x = st.pop() # 리스트 맨 뒤에서 데이터 꺼내기

## 큐
qu = []
qu.append(x) # 리스트 맨 뒤에 데이터 추가
x = qu.pop(0) # 리스트 맨 앞(0번 째)에서 데이터 꺼내기

## 딕셔너리

# 딕셔너리 생성
d = {}
d = dict()

# 딕셔너리 내용 삭제
d.clear()

## 그래프
# 노드와 간선으로 이루어진 자료 구조
# 루트 노드의 개념이 없고, 부모-자식 관계 개념이 없음
# Undirected Graph 무방향 그래프 와 Directed Graph 방향 그래프
# 방향 그래프에는 순환 그래프 (Cyclic) 와 비순환 그래프 (Acyclic)

## Binary Search 알고리즘
# 찾고자 하는 값을 위해 주어진 값의 중앙의 위치를 찾아서 해당 값 찾기

def binary_search(arr, x):
    # 탐색할 범위를 정하는 변수 left 와 right.
    left = 0
    right = len(arr) - 1
    while left <= right:             # 탐색을 계속하는 조건 확인.
        middle = (left + right)//2   # 정수로 나누기로 중앙위치 계산.
        if x == arr[middle]:         # 성공!
            return middle
        elif x > arr[middle]:        # 검색 범위를 좁힌다.
            left = middle + 1
        else:                        # 검색 범위를 좁힌다.
            right = middle - 1
    return -1                        # 실패!

## 재귀 알고리즘
# 피보나치 수열
def fibonacci_recursive(n):
    if n <= 1:                           # F(0) = 0 이며 F(1) = 1 이다.
        return n
    else:
        return(fibonacci_recursive(n-1) + fibonacci_recursive(n-2))    # F(n) = F(n-1) + F(n-2). 재귀호출.

## 유클리드 알고리즘
# 최대 공약수
def gcd_recursive(a, b):
    if b ==0:
        return a
    else:
        return gcd_recursive(b, a % b)   # 재귀 호출.

## 다이내믹 프로그래밍
# Memoization
# 피보나치 수열을 구할 때, 더 효율적으로 계산

cache = {} # Cache 메모리.
def fibonacci_memo(n):
    if n in cache:
        return cache[n]
    else:
        if n <= 1: # F(0) = 0 이며 F(1) = 1 이다.
            cache[n] = n
        else:
            cache[n] = fibonacci_memo(n-1) + fibonacci_memo(n-2) # F(n) = F(n-1) + F(n-2). 재귀호출.
        return cache[n]

## decorator function

# 기존의 코드를 건드리지 않고, wrapper 함수를 통해서 새로운 기능의 추가가 가능하다!
def decorator_function(f):
    def wrapper_function():
        print('{} 함수 실행 전 입니다.'.format(f.__name__))
        return f()
    return wrapper_function

@decorator_function
def MyFunc():
    print("이제 Myfunc 함수가 실행 되었습니다.")

# decorator_function + Memoization
def decorator_memoize(f):
    cache = {}
    def wrapper_function(n):
        if n in cache: # 이미 기록되어 있으면, cache에서 가져온다.
            return cache[n]
        else:
            cache[n] = f(n) # 아니면 새롭게 계산하세, cache에 기록한다.
            return cache[n]
    return wrapper_function

# 사실상 재귀 호출 함수인 것을 decorate 한다.
@decorator_memoize
def fibonacci(n):
    if n <= 1: # F(0) = 0 이며 F(1) = 1 이다.
        return n
    else:
        return(fibonacci(n-1) + fibonacci(n-2)) # F(n) = F(n-1) + F(n-2). 재귀호출.

