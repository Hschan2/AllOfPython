# BeautifulSoup으로 웹 크롤링하기
from bs4 import BeautifulSoup
import requests
# 다른 방법으로 네이버 실시간 검색어 가져오기에 필요한 라이브러리
import sys
from urllib.request import urlopen

# 크롤링할 사이트 가져오기
url = "https://www.rottentomatoes.com/m/parasite_2019/reviews"

# get = 단순 응답 코드 가져오기
req = requests.get(url)

# print(req)

# 사이트의 html 정보 가져오기
html = req.text

# print(html)

soup = BeautifulSoup(html, 'html.parser')

# 해당 url에서 특정 부분 가져오기
# url에 들어가서 F12, Element에서 원하는 부분 가져오기
name = soup.select(
    'div.review_table > div > div.col-xs-8 > div.col-sm-13.col-xs-24.col-sm-pull-4.critic_name > a.unstyled.bold.articleLink'
)

## 네이버 실시간 검색어 크롤링

# Json으로 실시간 검색어 가져오기
json = requests.get('https://www.naver.com/srchrank?frm=main').json()

ranks = json.get("data")
# print(ranks)

for key in ranks:
    rank = key.get("rank")
    keyword = key.get("keyword")
    print(str(rank) + ". ", keyword)

## 다른 방법으로 네이버 실시간 검색어 가져오기

# browser 설정
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'}
# url 설정
url = 'https://datalab.naver.com/keyword/realtimeList.naver?where=main'
# 응답 코드 설정
res = requests.get(url, headers = headers)
soup = BeautifulSoup(res.content, 'html.parser')
# 실시간 검색어 가져오기
data = soup.select('span.item_title')
i = 1
for item in data:
    print(str(i)+"위: "+item.get_text())
    i += 1

## 네이버 뉴스 10개 가져오기
# 검색하고자 하는 단어를 입력
# input으로 설정 가능
search = '코로나'
url_format = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query=' + search

r = requests.get(url_format)

sp = BeautifulSoup(r.text, 'html.parser')

sources = sp.select('div.group_news > ul.list_news > li div.news_area > a')

i = 0
for source in sources:
    i += 1
    print(str(i) + '.', source.attrs['title'])
# for source in sources:
#     print(source.attrs['title'])