import pymysql
import csv
import codecs

def get_conn():
    db = pymysql.connect(host="127.0.0.1", 
                     port=3306,
                     user="root", 
                     passwd="112798", 
                     db="weiwei", 
                     use_unicode=True, 
                     charset="utf8mb4")
    return db

def execute_all(cursor,sql,args):
    cursor.execute(sql,args)
    return cursor.fetchall()

# 0消极，1中性，2积极
def red_mysql_to_csv(filename):
    with codecs.open(filename=filename,mode='w',encoding='utf-8-sig')as f:
        write = csv.writer(f,dialect='excel')
        db = get_conn()
        cursor =db.cursor()
        sql ='select content from wb where sentiment=2'
        results = execute_all(cursor=cursor,sql=sql,args=None)
        for res in results:
            write.writerow(res)
