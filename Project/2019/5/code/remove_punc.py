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
sql1 = "select id,content from dataset"
cursor.execute(sql1)
results = cursor.fetchall()


punctuation = '!,;:?#."\''
def removePunctuation(text):
	text = re.sub(r'[{}]+'.format(punctuation), '', text)
	return text.strip()

def write2mysql(id,sentence):
	sql_in = "update dataset set sentence='%s' where id=%d"%(pymysql.escape_string(sentence),id)
	cursor.execute(sql_in)
	db.commit()
	
if __name__ == "__main__":
	for row in results:
		id = row[0]
		content = removePunctuation(row[1])
		write2mysql(id,content)
		
		