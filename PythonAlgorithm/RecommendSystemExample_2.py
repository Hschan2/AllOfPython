## 네이버 뉴스 추천
# doc2vec 활용
from gensim.models import Doc2Vec
import numpy as np
import pandas as pd

# doc2vec Model을 학습하고 저장하는 함수
# 128차원 고정 길이의 벡터로 만들기
# tok = 형태소 분석기 적용 유무
def make_doc2vec_models(tagged_data, tok, vector_size = 128, window = 3, epochs = 40, min_count = 0, workers = 4):
    model = Doc2Vec(tagged_data, vector_size = vector_size, window = window, epochs = epochs, min_count = min_count, workers = workers)
    model.save(f'./datas/{tok}_news_model.doc2vec')

# 전처리 담당 함수
def get_preprocessing_data(data):
    data.drop(['date', 'company', 'url'], axis = 1, inplace = True)

    category_mapping = {
        '경제': 0,
        '정치': 1,
        'IT과학': 2
    }

    data['category'] = data['category'].map(category_mapping)
    data['title_content'] = data['title'] + " " + data['content']
    # 불필요한 컬럼 제거
    data.drop(['title', 'content'], axis = 1, inplace = True)

    return data

# doc2vec에 필요한 데이터 만드는 함수
def make_doc2vec_data(data, column, t_document = False):
    data_doc = []

    for tag, doc in zip(data.index, data[column]):
        doc = doc.split(" ")
        data_doc.append(([tag], doc))
    
    if t_document:
        data = [TaggedDocument(words = text, tags = tag) for tag, text in data_doc]
        return data
    else:
        return data_doc

# 코사인 유사성으로 Top 5개 추천 함수
def get_recommend_contents(user, data_doc, model):
    scores = []

    for tags, text in data_doc:
        trained_doc_vec = model.docvecs[tags[0]]
        scores.append(cosine_similarity(user.reshape(-1, 128), trained_doc_vec.reshape(-1, 128)))

    scores = np.array(scores).reshape(-1)
    # 높은 인덱스 추출
    scores = np.argsort(-scores)[:5]

    return data.loc[scores, :]

# content vector 기반 user history를 평균하여 user embedding 만드는 함수
def make_user_embedding(index_list, data_doc, model):
    user = []
    user_embedding = []

    for i in index_list:
        user.append(data_doc[i][0][0])
    for i in user:
        user_embedding.append(model.docvecs[i])
    
    user_embedding = np.array(user_embedding)
    user = np.mean(user_embedding, axis = 0)

    return user

# 사용자 history 내역보는 함수
def view_user_history(data):
    print(data[['category', 'title_content']])

# Data 가져오기
data = get_data()

# data를 활용해 doc2vec model 생성
# doc2vec에 활용될 수 있도록 데이터 전처리
# tag가 붙은 데이터는 doc2vec model 학습에 사용
# 나머지 변수느는 user embedding, cosine similarity를 구할 때 사용
data_doc_title_content_tag = make_doc2vec_data(data, 'title_content', t_document = True)
data_doc_title_content = make_doc2vec_data(data, 'title_content')
data_doc_tok_tag = make_doc2vec_data(data, 'mecab_tok', t_document = True)
data_doc_tok = make_doc2vec_data(data, 'mecab_tok')

# 형태소 분석기 결과 데이터는 tok = True로 보내기 => 각각 데이터에 맞게 doc2vec model이 저장
make_doc2vec_models(data_doc_title_content_tag, tok = False)
make_doc2vec_models(data_doc_tok_tag, tok = True)

# 사용자 히스토리 생성. 각 카테고리 별로 user history 생성
# 경제 부분
user_category_1 = data.loc[random.sample(data.loc[data.category == 0, :].index.values.tolist(), 5), :]
view_user_history(user_category_1)

# 정치 부분
user_category_2 = data.loc[random.sample(data.loc[data.category == 1, :].index.values.tolist(), 5), :]
view_user_history(user_category_2)

# IT과학 부분
user_category_3 = data.loc[random.sample(data.loc[data.category == 2, :].index.values.tolist(), 5), :]
view_user_history(user_category_3)

# 사용자 히스토리로 user embedding 하기
# doc2vec에서 사용되는 tag 데이터가 아닌 ([tag], doc) 데이터 사용
# model은 형태소 분석 모델이 아닌 제목 + 본문 모델을 보내기
# user_1, user_2, user_3 각각 embedding을 평균화하여 (128,)의 결과값 출력
# 경제
user_1 = make_user_embedding(user_category_1.index.values.tolist(), data_doc_title_content, model_title_content)
# 정치
user_2 = make_user_embedding(user_category_2.index.values.tolist(), data_doc_title_content, model_title_content)
# IT과학
user_3 = make_user_embedding(user_category_3.index.values.tolist(), data_doc_title_content, model_title_content)

# 최종적으로 네이버 뉴스 Top 5 추천하기
# 경제
result_1 = get_recommend_contents(user_1, data_doc_title_content, model_title_content)
pd.DataFrame(result.loc[:, ['category', 'title_content']])
# 정치
result_2 = get_recommend_contents(user_2, data_doc_title_content, model_title_content)
pd.DataFrame(result.loc[:, ['category', 'title_content']])
# IT과학
result_3 = get_recommend_contents(user_3, data_doc_title_content, model_title_content)
pd.DataFrame(result.loc[:, ['category', 'title_content']])