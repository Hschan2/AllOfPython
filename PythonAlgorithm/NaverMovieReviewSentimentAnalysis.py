import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## 데이터 로드하기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

# Pandas를 이용해 훈련 데이터, 테스트 데이터 저장
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# 확인 후 train_data는 150,000개 존재, test_data는 50,000개 존재

# 상위 5개 출력 시
train_data[:5]
# => id, document, label로 구성되어 있음을 확인 가능, test_data도 같게 구성

## train_data, test_data 정제하기
train_data['document'].nunique(), train_data['label'].nunique()
# => (146182, 2) = 중복된 샘플을 제거한 개수, label 열 수. 약 4000개의 중복 샘플 제거, label 열은 0 또는 1의 값만 가지므로 2 출력

# 중복 샘플 제거
train_data.drop_duplicates(subset = ['document'], inplace = True)

# train_data에서 해당 리뷰의 긍정/부정 유무가 기재되어있는 Label 값 분포 확인
train_data['label'].value_counts().plot(kind = 'bar')
# => 그래프 상에서 긍정과 부정의 값이 각각 약 72,000개가 존재하며 레이블의 분포가 균일한 것을 확인 가능

# 레이블의 분포를 정확하게 확인
train_data.groupby('label').size().reset_index(name = 'count')
# => 0 73342, 1 72841.

# 리뷰 중 Null 값을 가진 샘플 확인
train_data.isnull().value.any()
# => True. 즉, Null 값 존재

# Null 값 존재 열 확인
train_data.isnull().sum()
# => id 0 document 1 label 0 dtype: int64. Null 값 샘플 총 1개

# document 열에서 Null 값이 존재한다는 것을 조건으로 Null 값을 가진 샘플이 어느 인덱스의 위치에 존재하는지 확인
train_data.loc[train_data.document.isnull()]
# =>        id        document    label
#     25857 2172111   NaN         1

# Null 값 샘플 제거
train_data = train_data.dropna(how = 'any')

## 데이터 전처리 진행
# 정규 표현식으로 특수 문자 제거하기

# 진행 전, 영어로 확인하기
text = 'do!!! you expect... people~ to~ read~ the FAQ, etc. and actually accept hard~! atheism?@@'
# 알파벳과 공백을 제외하고 모두 제거
re.sub(r'[^a-zA-Z ]', '', text)
# => do you expect people to read the FAQ etc and actually accept hard atheism

# 한글 샘플에서 특수 문자 제거하기
# 만약 한글이 아닐 경우, 빈 값만 출력
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# train_data에 빈 값이 가진 행이 있다면 Null값으로 변경
train_data['document'].replace('', np.nan, inplace = True)

# Null 값으로 변경 후, Null 값 개수 확인
train_data.isnull().sum()
# => id 0, document 391, label 0 dtype: int64. Null 값이 391개 생성

# 391개의 Null 값 제거
train_data = train_data.dropna(how = 'any')

## test_data로 복습하기
# document 열에서 중복 제거
test_data.drop_duplicates(subset = ['document'], inplace = True)
# 정규 표현식으로 특수 문자 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
# 공백이 존재할 경우, Null 값으로 변경
test_data['document'].replace('', np.nan, inplace = True)
# Null 값이 존재할 경우, 제거
test_data = test_data.dropna(how = 'any')

## Tokenizer, 토근화하기

# 불용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 토큰화를 위한 형태소 분석기 KoNLPy의 Okt 사용, 한국어를 토큰화를 할 때 자주 사용
okt = Okt()
# stem = True => 일정 수준의 정규화 수행, 예. 이런 -> 이렇다, 만드는 -> 만들다
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)
# => '오다', '이렇다', '것', '도', '영화', '라고', '차라리', '뮤직비디오', '를', '만들다', '게', '나다', '뻔'

# train_data에 형태소 분석기를 사용하여 토큰화하기
X_train = []
for sentence in train_data['document']:
    temp_X = []
    # 토큰화
    temp_X = okt.morphs(sentence, stem = True)
    # 불용어 제거
    temp_X =[word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)

# test_data로 토큰화 복습하기
X_test = []
for sentence in test_data['document']:
    temp_X = []
    # 토큰화
    temp_X = okt.morphs(sentence, stem = True)
    # 불용어 제거
    temp_X =[word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)

## 정수 인코딩
# 기계가 텍스트를 숫자로 처리할 수 있도록 train_data, test_data를 정수 인코딩 수행

# 단어 집합 만들기, 생성 동시에 각 단어에 고유한 정수 부여
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

tokenizer.word_index
# => {'영화': 1, '보다': 2, '을': 3, '없다': 4, '이다': 5, '있다': 6, '좋다': 7, ... 중략 ... '디케이드': 43751, '수간': 43752}
# 결과 값은 등장 빈도수가 가장 높은 순서대로 부여, 부여된 정수 값이 클수록 빈도수가 가장 낮음

# 빈도수가 낮은 단어들을 자연어 처리에서 배제
# 등장 빈도수가 3회 미만인 단어 확인
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어 수
rare_cnt = 0 # 등장 빈도수가 3보다 작은 단어의 개수 카운트
total_freq = 0 # train_data의 전체 단어 빈도수의 총 합
rare_freq = 0 # 등장 빈도수가 3보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍을 key와 value로 받기
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt += 1
        rare_freq += value

# 단어 집합의 크기
total_cnt
# 등장 빈도가 3회 미만 = 등장 빈도가 2회 이하, 희귀 단어의 수
threshold - 1, rare_cnt
# 단어 집합에서 희귀 단어 비율
(rare_cnt / total_cnt) * 100
# 전체 등장 빈도에서 희귀 단어 등장 빈도 비율
(rare_freq / total_freq) * 100

# => 43752, 24337, 55.62488571950996, 1.8715872104872904
# 55.624의 수치를 봤을 때, 희귀 단어 비율(2회 이하)이 반 이상을 차지하는 것을 확인 가능
# 등장 빈도 비율은 1.8% 밖에 안되는 것을 확인 가능

## 희귀 단어 제거
# 등장 빈도수가 2회 이하인 단어들의 수를 제외한 단어의 개수를 단어 집합의 최대 크기로 제한
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 + 2
vocab_size = total_cnt - rare_cnt + 2
# => 19417

# Keras Tokenizer의 인자로 넘겨주면 Keras Tokenizer는 텍스트 시퀀스를 숫자 시퀀스로 변경
# 정수 인코딩 과정에서 이보다 큰 숫자가 부여된 단어들은 OOV로 변환 => 정수 1번으로 할당
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train
# => [[51, 455, 17, 261, 660], [934, 458, 42, 603, 2, 215, 1450, 25, 962, 676, 20], [387, 2445, 2316, 5672, 3, 223, 10]]
# 한국어를 기계가 읽을 수 있도록 숫자로 토큰화
# 단어의 개수는 19,417개로 제한. 0번 ~ 19,416번 단어까지 사용. 0번 단어는 패딩을 위한 토큰, 1번 단어는 OOV를 위한 토큰 => 19,417번 이상의 단어는 데이터에 존재하지 않음

# train_data에서 y_train, y_test를 별도로 저장
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

## 빈 샘플 제거
# 빈도 수가 낮은 단어가 삭제되었다. 빈도 수가 낮은 단어만으로 구성되었던 샘플 = 빈 샘플
# 빈 샘플들은 어떤 레이블이 붙어있든지 의미가 없으므로 빈 샘플 제거 작업하기

# 각 샘플들 길이 확인 후 길이가 0인 샘플들의 인덱스 받아오기
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

# 빈 샘플 제거 전 = 145,791

# 빈 샘플 제거
X_train = np.delete(X_train, drop_train, axis = 0)
y_train = np.delete(y_train, drop_train, axis = 0)
# 빈 샘플 제거 후 = 145,380

## 패딩
# 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 패딩 작업 수행

# 리뷰 최대 길이 구하기 => 72
max(len(l) for l in X_train)
# 리뷰 평균 길이 구하기 => 11.002187371027652
sum(map(len, X_train)) / len(X_train)
# 히스토그램 그리기
plt.hist([len(s) for s in X_train], bins = 50)
plt.xlabel('샘플 길이')
plt.ylabel('샘플 수')
plt.show()

## 모델이 처리할 수 있도록 X_train과 X_test 모든 샘플의 길이를 특정 길이로 동일하게 만들기
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s)<= max_len):
            cnt += 1
    (cnt / len(nested_list)) * 100 # 전체 샘플 중 길이가 max_len보다 이하인 샘플의 비율

max_len = 30
below_threshold_len(max_len, X_train)
# => 30이하의 샘플 비율: 94.0830925849498

# 모든 샘플 길이를 30으로 동일
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

## LSTM으로 네이버 영화 리뷰 감성 분류

# 모델 생성하기
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 임베딩 벡터 차원 100으로 설정, LSTM 사용하기
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation = 'sigmoid'))

# 검증 데이터 손실(val_loss)이 증가하면, 과적합 징후로 검증 데이터 손실이 4회 증가하면 학습 조기 종료(Early Stopping)
es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 4)
# ModelCheckpoint를 사용해 검증 데이터 정확도(val_acc)가 이전보다 좋아질 경우메나 모델 저장
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

# 에포크 15번 수행, train 데이터 중 20%를 검증 데이터로 사용하면서 정확도 확인
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 15, callbacks = [es, mc], batch_size = 60, validation_split = 0.2)

# 위 과정을 수행하여 룬련 과정
"""
Train on 116304 samples, validate on 29076 samples
Epoch 1/15
116304/116304 [========================>.] - ETA: 0s - loss: 0.3903 - acc: 0.8236
Epoch 00001: val_acc improved from -inf to 0.84363, saving model to best_model.h5
116129/116129 [==========================] - 24s 210us/sample - loss: 0.3901 - acc: 0.8236 - val_loss: 0.3654 - val_acc: 0.8436
... 중략 ...
116304/116304 [==========================] - 23s 197us/sample - loss: 0.2064 - acc: 0.9198 - val_loss: 0.3708 - val_acc: 0.8432
Epoch 00009: early stopping
"""
# 조기 종료 조건에 따라 9번째 에포크에서 훈련 멈춤
# 훈련 완료 후 데이터 데이터에 대해 정확도를 측정하기
# 훈련 과정에서 검증 데이터의 정확도가 가장 높았을 때, 저장된 모델 'best_model.h5'를 Load
loaded_model = load_model('best_model.h5')
# 테스트 정확도 구하기
loaded_model.evaluate(X_test, y_test)[1]
"""
=> 결과
48296/48296 [==============================] - 9s 180us/sample - loss: 0.3387 - acc: 0.8544
테스트 정확도: 0.8544
"""
# 테스트 정확도 : 0.8544 => 85.44%의 정확도

## 리뷰 예측하기
# 직접 작성한 리뷰에 대해 예측하는 학습으로 현재 한습한 model에 새로운 입력 값에 대해 예측 값을 얻는 것은 model.predict() 사용
# model.fit()을 할 때와 마찬가지로 새로운 입력에 대해 동일한 전처리 수행 후 model.predict()의 입력을 사용
def sentiment_predict(new_sentence):
    new_sentence = okt.morphs(new_sentence, stem = True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩 (기계가 이해할 수 있도록)
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩 (길이 동일하게 맞추기)
    score = float(loaded_model.predict(pad_new)) # 예측
    
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰 \n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰 \n".format((1 - score) * 100))

"""
=> 결과
sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
97.76% 확률로 긍정 리뷰입니다.

sentiment_predict('이 영화 핵노잼 ㅠㅠ')
98.55% 확률로 부정 리뷰입니다.

sentiment_predict('이딴게 영화냐 ㅉㅉ')
99.91% 확률로 부정 리뷰입니다.

sentiment_predict('감독 뭐하는 놈이냐?')
98.21% 확률로 부정 리뷰입니다.

sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')
80.77% 확률로 긍정 리뷰입니다.
"""