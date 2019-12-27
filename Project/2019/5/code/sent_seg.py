'''
author：Fr3ya
date:20191205
function: 对每一条语句分句
'''
from nltk.tokenize import sent_tokenize
import re
import pymysql

# 连接数据库
db = pymysql.connect("127.0.0.1",
                     "root",
                     "123456",
                     "Apollo",
                     use_unicode=True,
                     charset="utf8mb4"
                     )
cursor = db.cursor()
sql = "select user,content,time,url from forums"
cursor.execute(sql)
results = cursor.fetchall()
def seg_sentence(data):
  reg_http = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
  pattern_http = re.compile(reg_http)
  result = []
  for i in sent_tokenize(data):
      result.append(re.sub(pattern_http, '. ', i))
  return result
def write2mysql(user,content,time,url):
  sql2 = "insert into dataset(time,content,usr,url) values('%d','%s','%s','%s')" %(time,pymysql.escape_string(content),\
                                                                                   pymysql.escape_string(user),\
                                                                                   pymysql.escape_string(url))
  cursor.execute(sql2)
  db.commit()
def main():
  for row in results:
    user = row[0]
    content = row[1]
    time = row[2]
    url = row[3]
    contents = seg_sentence(content)
    for i in contents:
      write2mysql(user, i, time, url)
if __name__ == '__main__':
    main()