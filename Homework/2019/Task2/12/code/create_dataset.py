import pandas as pd
from os.path import splitext
import csv
from urllib.parse import urlparse
import tldextract
import ipaddress as ip

#2016's top most suspicious TLD and words
Suspicious_TLD=['zip','cricket','link','work','party','gq','kim','country','science','tk']
Suspicious_Domain=['luckytime.co.kr','mattfoll.eu.interia.pl','trafficholder.com','dl.baixaki.com.br','bembed.redtube.comr','tags.expo9.exponential.com','deepspacer.com','funad.co.kr','trafficconverter.biz']
#trend micro's top malicious domains

featureSet = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at',\
'presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','presence of Suspicious_TLD',\
'presence of suspicious domain','label'))

# Method to count number of dots
def countdots(url):
    return url.count('.')


# Method to count number of delimeters
def countdelim(url):
    count = 0
    delim = [';', '_', '?', '=', '&']
    for each in url:
        if each in delim:
            count = count + 1

    return count

# Is IP addr present as th hostname, let's validate
def isip(uri):
    try:
        if ip.ip_address(uri):
            return 1
    except:
        return 0

#method to check the presence of hyphens
def isPresentHyphen(url):
    return url.count('-')

#method to check the presence of @
def isPresentAt(url):
    return url.count('@')

def isPresentDSlash(url):
    return url.count('//')

def countSubDir(url):
    return url.count('/')


def get_ext(url):
    """Return the filename extension from url, or ''."""

    root, ext = splitext(url)
    return ext

def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))

def countQueries(query):
    if not query:
        return 0
    else:
        return len(query.split('&'))

def getFeatures(url, label):
    result = []
    url = str(url)

    # add the url to feature set
    result.append(url)

    # parse the URL and extract the domain information
    path = urlparse(url)
    ext = tldextract.extract(url)

    # counting number of dots in subdomain
    result.append(countdots(ext.subdomain))

    # checking hyphen in domain
    result.append(isPresentHyphen(path.netloc))

    # length of URL
    result.append(len(url))

    # checking @ in the url
    result.append(isPresentAt(path.netloc))

    # checking presence of double slash
    result.append(isPresentDSlash(path.path))

    # Count number of subdir
    result.append(countSubDir(path.path))

    # number of sub domain
    result.append(countSubDomain(ext.subdomain))

    # length of domain name
    result.append(len(path.netloc))

    # count number of queries
    result.append(len(path.query))

    # Adding domain information

    # if IP address is being used as a URL
    result.append(isip(ext.domain))

    # presence of Suspicious_TLD
    result.append(1 if ext.suffix in Suspicious_TLD else 0)

    # presence of suspicious domain
    result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0)

    # result.append(get_ext(path.path))
    result.append(str(label))
    return result

def calFeatures(url):
    result = []
    url = str(url)

    path = urlparse(url)
    ext = tldextract.extract(url)

    result.append(countdots(ext.subdomain))
    result.append(isPresentHyphen(path.netloc))
    result.append(len(url))
    result.append(isPresentAt(path.netloc))
    result.append(isPresentDSlash(path.path))
    result.append(countSubDir(path.path))
    result.append(countSubDomain(ext.subdomain))
    result.append(len(path.netloc))
    result.append(len(path.query))
    result.append(isip(ext.domain))
    result.append(1 if ext.suffix in Suspicious_TLD else 0)
    result.append(1 if '.'.join(ext[1:]) in Suspicious_Domain else 0)

    return result

if __name__ == '__main__':
    df = pd.read_csv('../data/data.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    mapping = {'good':0, 'bad':1}
    df['label'] = df['label'].map(mapping)
    df.dropna(axis=0, how='any')


    with open ('../data/domain.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(df)):
            features = getFeatures(df["url"].loc[i], df["label"].loc[i])
            writer.writerow(features)
