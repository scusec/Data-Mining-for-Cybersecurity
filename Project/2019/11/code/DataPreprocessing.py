#Parsing normalTraffic.txt & anomalousTraffic.txt using Python version 3.6.1
#https://github.com/Monkey-D-Groot/Machine-Learning-on-CSIC-2010/blob/master/main.py 

import urllib
import urllib.parse
import io

normal_file = 'normalTraffic.txt'
anomalous_file = 'anomalousTraffic.txt'

normal_parsed = 'normal_parsed.txt'
anomalous_parsed = 'anomalous_parsed.txt'

#extracting the url
def parse_file(file_in, file_out):
    fin = open(file_in)
    fout = io.open(file_out, "w", encoding="utf-8")
  
    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]
    
            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n','').lower()
        fout.writelines(line + '\n')
    print ("finished parse ",len(res)," requests")
    fout.close()
    fin.close()
	

parse_file(normal_file,normal_parsed)
parse_file(anomalous_file, anomalous_parsed)
