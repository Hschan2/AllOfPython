import pymysql

# Python과 Mysql 연결하기

# 데이터 베이스 연결
conn = pymysql.connect(host = "localhost", user = "root", password = "111111", db = "opentutorials", charset = "utf8")

try:
    # 연결한 데이터 베이스와 상호 작용
    # curs = conn.cursor()
    curs = conn.cursor(pymysql.cursors.DictCursor) # 컬럼명으로 조회

    sql = "select * from topic"
    curs.execute(sql)
    ## where 조건을 사용 가능
    # sql = "select * from topic where title = %s"
    # curs.execute(sql, 'PHP')
    rows = curs.fetchall()
    # print(rows)

    ## Insert
    # curs = conn.cursor()
    # sql = """insert into topic(title, description, created, author_id)
    #         value(%s, %s, now(), %s)
    # """
    # curs.execute(sql, ('JAVASCRIPT', 'JAVASCRIPT is ...', , '5'))
    # conn.commit()
    # conn.close()

    ## Update
    # curs = conn.cursor()
    # sql = """update topic set title = 'Javascript' where title = 'JAVASCRIPT'"""
    # curs.execute(sql)
    # conn.commit()
    # conn.close()

    ## Delete
    # curs = conn.cursor()
    # sql = """delete from topic where title = %s"""
    # curs.execute(sql, 'Javascript')
    # conn.commit()
    # conn.close()

    for row in rows:
        # print(row[0], row[1]) # 각 행의 0번째, 1번째... 출력
        print(row['title'])
except:
    print("Error")
finally:
    conn.close()