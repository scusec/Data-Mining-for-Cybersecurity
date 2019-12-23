# -*- coding: utf-8 -*-

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


class LDAClustering():
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def cut_words(self, sentence):
        return ' '.join(jieba.lcut(sentence))

    def pre_process_corpus(self, corpus_path, stopwords_path):
        """
        数据预处理，将语料转换成以词频表示的向量。
        :param corpus_path: 语料路径，每条语料一行进行存放
        :param stopwords_path: 停用词路径
        :return:
        """
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = [self.cut_words(line.strip()) for line in f]

        stopwords = self.load_stopwords(stopwords_path)

        self.cntVector = CountVectorizer(stop_words=stopwords)

        cntTf = self.cntVector.fit_transform(corpus)

        return cntTf

    def pre_process_line(self, line, stopwords_path):
        line = [self.cut_words(line.strip())]
        stopwords = self.load_stopwords(stopwords_path)
        self.cntVector = CountVectorizer(stop_words=stopwords)
        result = self.cntVector.fit_transform(line)
        return result


    def fmt_lda_result(self, lda_result):
        ret = {}
        for doc_index, res in enumerate(lda_result):
            li_res = list(res)
            doc_label = li_res.index(max(li_res))
            if doc_label not in ret:
                ret[doc_label] = [doc_index]
            else:
                ret[doc_label].append(doc_index)
        return ret

    def map_topic(self, idx):
        dict = {'新闻': [15, 25],
                '财经': [0, 7, 8],
                '科技': [2, 3, 6],
                '体育': [1, 4, 11],
                '娱乐': [5, 17],
                '汽车': [10],
                '博客': [9],
                '视频': [12],
                '房产': [13],
                '时尚': [14],
                '教育': [16],
                '图片': [18, 21],
                '微博': [19, 20, 22],
                '旅游': [24],
                '游戏': [23],
                '社会': [26, 27, 28, 29]}
        if idx in dict.values():
            return list(dict.keys())[list(dict.values()).index(idx)]

    def lda(self, corpus_path, n_components=5, learning_method='batch',
            max_iter=10, stopwords_path='./stop_words.txt'):
        """
        LDA主题模型
        :param corpus_path: 语料路径
        :param n_topics: 主题数目
        :param learning_method: 学习方法: "batch|online"
        :param max_iter: EM算法迭代次数
        :param stopwords_path: 停用词路径
        :return:
        """
        cntTf = self.pre_process_corpus(corpus_path=corpus_path, stopwords_path=stopwords_path)
        tf_feature_names = self.cntVector.get_feature_names()
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, learning_method=learning_method)
        docres = lda.fit_transform(cntTf)

        print_top_words(lda, tf_feature_names, n_top_words=15)

        return self.fmt_lda_result(docres)

    def lda_predict(self, heading, text, n_components=5, learning_method='batch',
            max_iter=10, stopwords_path='./stop_words.txt' ):
        pre_str = self.pre_process_line(heading+text, stopwords_path)
        tf_feature_names = self.cntVector.get_feature_names()
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter, learning_method=learning_method)
        res = lda.fit_transform(pre_str)
        return max(res[0]), self.map_topic(res[0].index(max(res[0])))
    

if __name__ == '__main__':
    LDA = LDAClustering()
    heading = "周琦"
    text = "周琦男篮世界杯失误"
    ret = LDA.lda_predict(heading, text, stopwords_path='./stop_words.txt', max_iter=100, n_components=30)

