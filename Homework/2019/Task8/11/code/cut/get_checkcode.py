import requests
import time
import urllib
import urllib.request
import random

pic_path = "/Users/dqy/My/captcha/checkcode/"

my_headers=[
"Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
"Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",  
"Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)"
]  


def downloads_pic(pic_name):
    randdom_header=random.choice(my_headers)
    url = 'https://www.cndns.com/common/GenerateCheckCode.aspx'
    req = urllib.request.Request(url)
    req.add_header("User-Agent",randdom_header)
    req.add_header("GET",url)
    res = urllib.request.urlopen(req)
    with open(pic_path + "%d"%(pic_name)+'.png', 'wb') as f:
        chunk = res.read()
        f.write(chunk)
        f.flush()
        f.close()

if __name__ == '__main__':
    for i in range(300):
        pic_name = int(time.time()*10000)
        downloads_pic(pic_name)