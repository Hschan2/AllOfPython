## 자연어 처리(분석) 기초

## 자연어
# 인간 고유의 언어, 정보 전달이 수단

# * 인공 언어 *
# 특정 목적을 위해 인위적으로 만든 언어, 자연 언어에 비해 엄격한 구문

# 자연어 분석 용어
# NLP - Natural Language Processing (자연어 처리)
# NLU - Natural Language Undertanding (자연 언어 이해)

# 자연어 분석 단계
# 자연 언어 문장 -> 형태소 분석 -> 구문 분석 -> 의미 분석 -> 화용 분석 -> 분석 결과

## 형태소 분석
# 입력된 문장을 잘 분할해서 효율성을 높이기 위해서 사용
# setnecne splitting : 마침, 느낌표, 물음표 등을 기준으로 분리
# Tokenizing : 문서나 문장을 분석하기 좋도록 나눔
# Morphological : 토큰들을 조금 더 일반적인 형태로 분석해 단어수를 줄여 분석의 효율성을 높임
# Stemming : cars, car -> car로 .. 단어를 원형으로 나누기

## 구문 분석
# 문장이 의미적으로 올바른가. 확인하는 것이 필요하기 때문에

## Discourse Analysis
# 대화의 흐름을 파악하여 발화자의 의도에 맞도록 응답하기
# 대화의 흐름상 어떤 의미를 가지는가 찾기
# 문맥 구조 분석 (문장들의 연관 관계)
# 의도 분석 (전후 관계를 통한 실제 의도)
# 대화 분석 (대표적인 담화 분석)
# EX. 실시간 강연 통역 시스템

## 자연어 처리 이해하기
# 문장 안에 단어 하나 하나가 컴퓨터가 이해할 수 있도록 만들기
# 문장 속 단어는 사람은 이해할 수 있지만 기계는 이해하기 어렵다.
# 컴퓨터가 문장 속 단어를 이해하기 위해 수치로 표한하도록 만든다.
# 즉, 텍스트를 숫자로 표현한다 = Word Embedding

# Document - 문서
# Corpus - 문서의 집합
# Token - 단어처럼 의미를 가지는 요소
# Morphemes (형태소) - 의미를 가지는 언어에서 최소 단위
# POS (품사) - Nouns(명사), Verbs(동사)
# Stopword (불용어) - 자주 나타나지만 실제 의미에 크게 기여하지 못하는 단어
# Stemming (어간 추출) - 어간만 추출 (EX. running, runs, run -> run)
# Lemmatization (음소 표기법) - 앞 뒤 문맥을 보고 단어를 식별

## 자연어 처리 라이브러리
# NLTK - 대표적인 자연어 처리 라이브러리 (말뭉치 다운로드, Word POS, NER Classification, Document Classification...)
# KoNLPy - 한글에 특화된 자연어 처리 라이브러리 (단어 품사별 분류, Hannanum, Kkma, Komoran, Twitter...)
# Gensim - 문서 사이의 유사도 계산과 텍스트 분석을 돕는 라이브러리, Word2Vec 제공 (DicVectorizer, CountVectorizer, TfidVectorizer, HashingVectorizer...)

## 자연어 처리(분석) 예제로 배우기
# 토근화 (spacy 라이브러리)
en_text = "A Dog Run back corner near spare bedrooms"

import spacy

spacy_en = spacy.load("en")

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]

print(tokenize(en_text))
# => ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']

## NLTK 사용하기
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))
# => ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']

## 띄어쓰기로 토큰화
print(en_text.split())
# => ['A', 'Dog', 'Run', 'back', 'corner', 'near', 'spare', 'bedrooms']

## 한글 띄어쓰기 토큰화
kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

print(kor_text.split())
# => ['사과의', '놀라운', '효능이라는', '글을', '봤어.', '그래서', '오늘', '사과를', '먹으려고', '했는데', '사과가', '썩어서', '슈퍼에', '가서', '사과랑', '오렌지', '사왔어']

## 형태소 토큰화 (mecab 라이브러리)
from konlpy.tag import mecab
tokenize = mecab()
print(tokenize.morphs(kor_text))
# => ['사과', '의', '놀라운', '효능', '이', '라는', '글', '을', '봤', '어', '.', '그래서', '오늘', '사과', '를', '먹', '으려고', '했', '는데', '사과', '가', '썩', '어서', '슈퍼', '에', '가', '서', '사과', '랑', '오렌지', '사', '왔', '어']
# '의', '를', '가', '랑' 등이 전부 분리되어 기계는 '사과'라는 단어를 하나의 단어로 처리할 수 있다.

## 문자 토큰화
print(list(en_text))
# => ['A', ' ', 'D', 'o', 'g', ' ', 'R', 'u', 'n', ' ', 'b', 'a', 'c', 'k', ' ', 'c', 'o', 'r', 'n', 'e', 'r', ' ', 'n', 'e', 'a', 'r', ' ', 's', 'p', 'a', 'r', 'e', ' ', 'b', 'e', 'd', 'r', 'o', 'o', 'm', 's']

## 단어 집합 생성
import urllib.request
import pandas as pd
from konlpy.tag import mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

# URL 가져오기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
# 데이터 프레임에 저장
data = pd.read_table('ratings.txt')
# 저장된 data에서 10개 저장
data[:10]
# 저장된 data에서 100개 저장
simple_data = data[:100]

## 정규 표현식을 통해서 데이터 정제
sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# [^ㄱ-ㅎㅏ-ㅣ가-힣 ]의 단어를 빈칸으로

## 토큰화 수행하기

# 불용어 정의
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

tokenizer = Mecab()

tokenized = []
for sentence in sample_data['document']:
    temp = []
    # 토큰화
    temp = tokenizer.morphs(sentence)
    # 불용어 제거
    temp = [word for word in temp if not word in stopwords]
    tokenized.append(temp)

# => [['어릴', '때', '보', '고', '지금', '다시', '봐도', '재밌', '어요', 'ㅋㅋ'], ['디자인', '을', '배우', '학생', '외국', '디자이너', '그', '일군', '전통', '을', '통해', '발전', '해', '문화', '산업', '부러웠', '는데', '사실', '우리', '나라', '에서', '그', '어려운', '시절', '끝', '까지', '열정', '을', '지킨', '노라노', '같', '전통', '있', '어', '저', '같', '사람', '꿈', '을', '꾸', '고', '이뤄나갈', '수', '있', '다는', '것', '감사', '합니다'], ['폴리스', '스토리', '시리즈', '부터', '뉴', '까지', '버릴', '께', '하나', '없', '음', '최고'], ['연기', '진짜', '개', '쩔', '구나', '지루', '할거', '라고', '생각', '했', '는데', '몰입', '해서', '봤', '다', '그래', '이런', '게', '진짜', '영화', '지'], ['안개', '자욱', '밤하늘', '떠', '있', '초승달', '같', '영화'], ['사랑', '을', '해', '본', '사람', '라면', '처음', '부터', '끝', '까지', '웃', '을', '수', '있', '영화'], ['완전', '감동', '입니다', '다시', '봐도', '감동'], ['개', '전쟁', '나오', '나요', '나오', '면', '빠', '로', '보', '고', '싶', '음'], ['굿'], ['바보', '아니', '라', '병', '쉰', '인', '듯']]

## 위에 내용으로 단어의 집합 만들기, NLTK의 FreqDist() 사용 (빈도수)
vocab = FreqDist(np.hstack(tokenized))

vocab['재밌']
# => 10, '재밌'이라는 단어가 총 10번 사용했다는 출력값이 나온다.
#     most_common()은 상위 빈도수를 가진 주어진 수의 단어만을 Return한다.
#     이를 사용하여 등장 빈도수가 높은 단어들을 원하는 개수만큼 얻을 수 있다.

# 등장 빈도수 상위 500개의 단어만 단어 집합으로 저장하기
vocab_size = 500
vocab = vocab.most_common(vocab_size)

## 각 단어에 고유한 정수 부여
# enumerate = 순서가 있는 자료형 (list, set, tuple, dictionary, string)을 입력 받아 인덱스를 순차적으로 함께 Return
# 인덱스 0과 1은 다른 용도로 사용하고 나머지 단어들은 2부터 501까지 순차적으로 인덱스 부여
word_to_index = {word[0] : index + 2 for index, word in enumerate(vocab)}
word_to_index['pad'] = 1
word_to_index['unk'] = 0

encoded = []
# 입력 데이터에서 1줄 씩 문장 읽기
for line in tokenized:
    temp = []
    # 각 줄에서 1개씩 글자 읽기
    for w in line:
        try:
            # 글자를 해당되는 정수로 변환
            temp.append(word_to_index[w])
        # 단어 집합에 없는 단어일 경우 unk로 대체
        except KeyError:
            # unk의 인덱스로 변환
            temp.append(word_to_index['unk'])
    encoded.append(temp)

print(encoded[:10])
# => [[78, 27, 9, 4, 50, 41, 79, 16, 28, 29], [188, 5, 80, 189, 190, 191, 42, 192, 114, 5, 193, 194, 21, 115, 195, 196, 13, 51, 81, 116, 30, 42, 197, 117, 118, 31, 198, 5, 199, 200, 17, 114, 7, 82, 52, 17, 43, 201, 5, 202, 4, 203, 14, 7, 83, 32, 204, 84], [205, 119, 206, 53, 207, 31, 208, 209, 54, 10, 25, 11], [44, 33, 120, 210, 211, 212, 213, 68, 45, 34, 13, 214, 121, 15, 2, 215, 69, 8, 33, 3, 35], [216, 217, 218, 219, 7, 220, 17, 3], [122, 5, 21, 36, 43, 123, 124, 53, 118, 31, 85, 5, 14, 7, 3], [125, 37, 221, 41, 79, 37], [120, 222, 55, 223, 55, 86, 224, 46, 9, 4, 47, 25], [56], [225, 87, 88, 226, 227, 57, 89]]

## 길이가 다른 문장들을 모두 동일한 길이로 바꿔주는 패딩 (Padding)
# 패딩 작업은 정해준 길이로 모든 샘플들의 길이를 맞춰주되, 길이가 정해준 길이보다 짧은 샘플들에는 'pad' 토큰을 추가하여 길이를 맞춰주는 작업

max_len = max(len(l) for l in encoded)

# 최대 길이
max_len
# 최소 길이
min(len(l) for l in encoded)
# 평균 길이
(sum(map(len, encoded)) / len(encoded))

# bins = 개수
plt.hist([len(s) for s in encoded], bins = 50)
plt.xlabel('샘플 길이')
plt.ylabel('샘플 수')
plt.show()
# => 최대 길이 = 63, 최소 길이 = 1, 평균 길이 = 13.900000

# 모든 리뷰의 길이 63으로 통일시키기
for line in encoded:
    # 현재 샘플이 정해진 길이보다 짧으면
    if len(line) < max_len:
        # 나머지를 전부 'pad' 토큰으로 채우기
        # 부족한 수만큼 'pad'로 채우기
        line += [word_to_index['pad']] * (max_len - len(line))

# 길이 통일시킨 후 최대 길이
max(len(l) for l in encoded)
# 길이 통일시킨 후 최소 길이
min(len(l) for l in encoded)
# 길이 통일시킨 후 평균 길이
(sum(map(len, encoded)) / len(encoded))
# => 최대 길이 = 63, 최소 길이 = 63, 평균 길이 = 63.000000

