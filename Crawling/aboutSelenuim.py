from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from bs4 import BeautifulSoup

fp = open("./TXT/1.txt", 'r', encoding="utf-8")
text = fp.read()
fp.close()

ready_list = []

while(len(text) > 500):
    # 처음부터 500글자 까지
    temp_str = text[:500]
    last_space = temp_str.rfind(' ')
    temp_str = text[0:last_space]
    ready_list.append(temp_str)

    text = text[last_space:]

ready_list.append(text)

dv = webdriver.Chrome(r"D:\ProgramData/chromedriver")
dv.get("https://www.naver.com")

elem = dv.find_element_by_name("query")
elem.send_keys("맞춤법 검사기")
elem.send_keys(Keys.RETURN)

time.sleep(2)
textarea = dv.find_element_by_name("txt_gray")

new_str = ''

for ready in ready_list:
    textarea.send_keys(Keys.CONTROL, "a")
    textarea.send_keys(ready)

    elem = dv.find_element_by_class_name("btn_check")
    elem.click()

    time.sleep(1)

    soup = BeautifulSoup(dv.page_source, 'html.parser')
    after = soup.select("p._result_text.stand_txt")[0].text

    new_str += after.replace('. ', '.\n')

fp = open("./TXT/result.txt", 'w', encoding='utf-8')
fp.write(new_str)
fp.close()