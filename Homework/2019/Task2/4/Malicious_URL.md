# Malicious_URL

###Step1选择数据集并进行处理
[数据集链接][1]
<div align="center">
<img src="https://ae01.alicdn.com/kf/He4cab1dec0434d1c942f2a6bcd46b8f0T.png"  width="300" >
</div>
**选择一些特征作为评判标准**
```python
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
```
**数据集处理完成后**
![](https://ae01.alicdn.com/kf/H3c8f6af7c57d44d2a55adaf2e42d5632t.png)
###Step2用决策树和随机森林两种方法训练数据集


  [1]: https://www.kaggle.com/antonyj453/urldataset
  [2]: https://ae01.alicdn.com/kf/He4cab1dec0434d1c942f2a6bcd46b8f0T.png
