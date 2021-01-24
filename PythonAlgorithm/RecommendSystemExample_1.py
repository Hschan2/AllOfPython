import pandas as pd

data = pd.read_csv('D:\\Bitnami\\wampstack-7.1.20-1\\apache2\\htdocs\\AllofPython\\Excel\\movies_metadata.csv')

# print(data.shape) => (45466, 24)

# 사용할 데이터 설정
daata = data[['id', 'genres', 'vote_average', 'vote_count', 'popularity', 'title', 'overview']]

# 500위 뽑기
m = data['vote_count'].quantile(0.9)
data = data.loc[data['vote_count'] >= m]

# 평균
C = data['vote_average'].mean()

# 최종 Score
def weighted_rating(x, m = m, C = C):
    v = x['vote_count']
    R = x['vote_average']

    return (v / (v + m) * R) + (m / (m + v) * C)

# 전처리하기
data['score'] = data.apply(weighted_rating, axis = 1)

# data.shape => (481, 9)
# 위 영화 추천은 '비슷한'에서 장르와 키워드로 분석 => 그래서 genres, keywords를 이용한다.
data[['genres', 'keywords']]

# But, Python List안에 Dictionary가 포함된 구조 -> 문자열 형태로 되엉 있기 때문에 변환 필요 -> ast안에 있는 literval_eval을 사용해 변환
data['genres'] = data['genres'].apply(literal_eval)
data['keywords'] = data['keywords'].apply(literal_eval)

# Name만 뽑기, id값은 필요 없기 때문에
# lambda... => dict 형태에서 list 형태에서 띄어쓰기로 이루어진 str로 변경
data['genres'] = data['genres'].apply(lambda x: [d['name'] for d in x]).apply(lambda x: " ".join(x))
data['keywords'] = data['keywords'].apply(lambda x: [d['name'] for d in x]).apply(lambda x: " ".join(x))

## 콘텐츠 기반 필터링 추천
# 띄어쓰기를 구분자로 한 문자열을 숫자로 바꾸어 벡터화 하기
# Python scikit learn에 있는 CountVectorizer 사용
count_vector = CountVectorizer(ngram_range = (1, 3))
c_vector_genres = count_vector.fit_transform(data['genres'])

# 벡터화 시킨 값으로 유사 영화 추천
# 보통 많이 사용하는 코사인 유사도 사용. Python scikit learn에서 cosine similarity 사용
# 동시에 argsort로 유사도가 가장 높은 인덱스를 가자 위로 정렬
gerne_c_sim = cosine_similarity(c_vector_genres, c_vector_genres).argsort()[:, ::-1]

# 추천 영화 30개
def get_recommend_movie_list(df, movie_title, top = 30):
    # 특정 영화와 비슷한 영화를 추천하기 위해 '특정 영화' 정보 추출
    target_movie_index = df[df['title'] == movie_title].index.values
    # 코사인 유사도 중 비슷한 코사인 유사도를 가진 정보 추출
    sim_index = gerne_c_sim[target_movie_index, :top].reshape(-1)
    # 특정 영화 본인 제외
    sim_index = sim_index[sim_index != target_movie_index]
    # Data Frame으로 생성하고 vote_count로 정렬
    result = df.iloc[sim_index].sort_values('score', ascending = False)[:10]
    return result

# The Dark Knight Rises와 유사한 영화 추천
get_recommend_movie_list(data, movie_title = 'The Dark Knight Rises')

# 영화 추천 기본 로직
# 1. 영화 제목이 들어오면 영화 제목을 가지고 있는 Index 추출
# 2. 코사인 유사도 중 영화 제목 인덱스에 해당하는 값에서 추천 개수만큼 추출 (30개)
# 3. 본인 제외
# 4. Imdb weighted rating을 적용한 score 기반 정렬
# 5. Python DataFrame으로 생성 후 return

## 협업 필터링 (Collaborative Filtering)
# 아이템 기반 협업 필터링, 행렬 분해 기반 협업 필터링 중 아이템 기반 협업 필터링 사용
rating_data = pd.read_csv('D:\\Bitnami\\wampstack-7.1.20-1\\apache2\\htdocs\\AllofPython\\Excel\\ratings.csv')
movie_data = pd.read_csv('D:\\Bitnami\\wampstack-7.1.20-1\\apache2\\htdocs\\AllofPython\\Excel\\movies.csv')

# csv 안 데이터 전처리
# 불필요한 Column 제거, MoiveID가 공통으로 Column에 있으니 MoiveID를 기준으로 pd.merge => 나누어져 있던 2개의 데이터 하나로 합치기
rating_data.drop('timetamp', axis = 1, inplace = True)

# movieId기준으로 합치기
# 사용자별 영화 평점을 매긴 데이터그 들어가고 title과 genres 정보가 함께 들어간다.
user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')

# 아이템 기반 협업 필터링을 위해 pivot table 생성하기
# 사용자 - 영화에 따른 평점 점수가 데이터로 들어가야 하기 때문에 pivot_table 사용
# data: 영화 평점 rating, index: 영화 title, columns: userId
movie_user_rating = user_movie_rating.pivot_table('rating', index = 'title', columns = 'userId')
user_movie_rating = user_movie_rating.pivot_table('rating', index = 'userId', columns = 'title')

# 만들어진 pivot_table은 index를 사용자 아이디와 영화 타이틀로 생성
# 아이템 기반 협업 필터링할 예정이므로, index를 영화 타이틀로 두고 column 정보는 유저 아이디로 둔다.
# 만들어진 테이블에 값이 현재 NaN이므로 Null값으로 우선 채우기
movie_user_rating.fillna(0, inplace = True)

# 아이템 기반 협업 필터링 추천 시스템은 유사한 아이템을 추천 => 여기 영화에서 평점이 비슷한 아이템
# 평점이 Data로 들어가 있기 때문에 현재 상태로 코사인 유사도를 구하기
item_based_collabor = cosine_similarity(movie_user_rating)

# Pandas DataFrame으로 생성하기
# 데이터 값이 0과 가까울수록 유사성이 작은 것. 1과 가까울수록 유사성이 큰 것
item_based_collabor = pd.DataFrame(data = item_based_collabor, index = movie_user_rating.index, columns = movie_user_rating.index)

# 영화 시청 후 마음에 들었따면 그 영화와 비슷한 영화 추천
def get_item_based_collabor(title):
    return item_based_collabor[title].sort_values(ascending = False)[:6]

# Godfather, the (1972)와 가장 유사한 영화 6개 추출
get_item_based_collabor('Godfather, the (1972)')
