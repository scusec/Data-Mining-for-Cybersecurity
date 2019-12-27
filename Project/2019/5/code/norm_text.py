import re
from ast import literal_eval
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
#sql1 = "select sent,hacker_name,tool,attack_mean,shuyu from neword where id<20000"
sql1 = "select id, content from dataset where id>=2000"
cursor.execute(sql1)
result = cursor.fetchall()

def normalize_text(text):
    #norm_text = text.lower()
    norm_text = text
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', '')
    # Pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", "\\1", norm_text)
    norm_text = ' '.join(norm_text.split())
    return norm_text
def remove_non_ascii(text):
    return ''.join(["" if ord(i) < 32 or ord(i) > 126 else i for i in text])

corpus = []
def write2mysql(id,content):
    sql_norm = "update dataset set content='%s' where id='%d'"%(pymysql.escape_string(content),id)
    cursor.execute(sql_norm)
    db.commit()
for row in result:
    rna = remove_non_ascii(row[1])
    norm_text=normalize_text(rna)
    print(norm_text.replace('[^\w\s]',''))
    write2mysql(row[0],norm_text.replace('[^\w\s]',''))
    corpus.append(norm_text)