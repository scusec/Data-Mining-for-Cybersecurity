import requests
import numpy as np
import pandas as pd
import re


# responses_list = []
tID_list = []

# for循环range第二个参数为总页数……就直接静态了 能用就行
for i in range(0, 649):
    url = 'https://samate.nist.gov/SARD/view.php?count=20&languages[]=PHP&flawed[]=Bad&status_Candidate=1&status_Accepted=1&sort=desc&first='
    url += str(i*20)
    print(url)
    print('正在爬取第%s页内容...' % str(i+1))
    html_content = requests.request(url=url, method='get').content.decode()
    find_tIDs = re.findall(string=html_content, pattern=r'tID=(\d+)\">')
    print(find_tIDs)
    for tID in find_tIDs:
        tID_list.append(str(tID)) if len(find_tIDs) != 0 else print('错误，该页面未找到tID！')
#     responses_list.append(html_content)
print(len(tID_list))



# 这里用了python字典的keys方法，为了去重
unique_dict = {}.fromkeys(tID_list)
tID_list = np.array(list(unique_dict.keys()))
print(tID_list[:50])
print(tID_list.shape)



file_url_pattern = r"getFile\('(\S+\.php)',"
file_name_pattern = r'\S+\/(\S+\.php)'
file_path = '/Users/devin/Desktop/xss_php_bad/'
file_list = []
php_url_prefix = 'https://samate.nist.gov/SARD/'

for tID in tID_list:
    tID_url = 'https://samate.nist.gov/SARD/view_testcase.php?tID=' + str(tID)
    print('tID URL: %s' %tID_url)
    tID_content = requests.request(url=tID_url, method='get').content.decode()
    php_url_path = re.findall(string=tID_content, pattern=file_url_pattern)[0]
    file_name = re.findall(pattern=file_name_pattern, string=php_url_path)[0]
    print('File name: %s' % file_name)
#     排除掉不是XSS的代码
    if 'cwe_79' not in file_name.lower():
        print('不是XSS漏洞代码！pass')
        continue
        
    
    if file_name not in file_list:
        file_list.append(file_name)
        php_url_path = php_url_prefix + php_url_path
        code_content = requests.request(url=php_url_path, method='get').content.decode('utf-8')
        with open(file_path+file_name, 'w+') as f:
            f.write(code_content)
        print('%s【写成功！】' % file_name)
    else:
        print('漏洞示例已存在，pass')
print('所有该漏洞文件均已写完毕')




