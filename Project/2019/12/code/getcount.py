import pymysql
import requests
import json

db = pymysql.connect("127.0.0.1",
                     "root",
                     "123456",
                     "weiwei",
                     use_unicode=True,
                     charset="utf8mb4"
                     )

cursor = db.cursor()

def getStasticInfo():
    sql1 = "select count(*) from wb"
    cursor.execute(sql1)
    result = cursor.fetchall()
    sql_good = "select count(*) from wb where sentiment='2'"
    cursor.execute(sql_good)
    result_good = cursor.fetchall()
    sql_bad = "select count(*) from wb where sentiment='0'"
    cursor.execute(sql_bad)
    result_bad = cursor.fetchall()
    all_num = result[0][0]
    good_num = result_good[0][0]
    bad_num = result_bad[0][0]
    m_num = all_num-good_num-bad_num
    db.commit()
    return all_num, good_num, m_num, bad_num

def getresult():
    sql="select id,content,sentiment from wb"
    cursor.execute(sql)
    results = cursor.fetchall()
    return results

def getInfoByID(id):
    sql2 = "select content, sentiment, positive_prob, negative_prob, confidence from wb where id = '%d' " % id
    cursor.execute(sql2)
    results = cursor.fetchall()
    content = results[0][0]
    sentiment = results[0][1]
    positive_prob = results[0][2]
    negative_prob = results[0][3]
    confidence=results[0][4]
    return content, sentiment, positive_prob, negative_prob, confidence

def getTopics():
    sql = "SELECT label as topic,COUNT(*) as num FROM wb GROUP BY label order by num desc"
    cursor.execute(sql)
    results = cursor.fetchall()
    topics=[]
    num = []
    for row in results:
        topics.append(row[0])
        num.append(row[1])
    return topics,num

def getLabel(id):
    sql = "select label from wb where id=%d" % id
    cursor.execute(sql)
    result = cursor.fetchall()
    label = result[0][0]
    return label

