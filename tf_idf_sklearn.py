from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from visual import show_tfidf   # this refers to visual.py in my [repo](https://github.com/MorvanZhou/NLP-Tutorials/)


docs = [
    "it is a good day, I like to stay here",
    "I am happy to be here",
    "I am bob",
    "it is sunny today",
    "I have a party today",
    "it is a dog and that is a cat",
    "there are dog and cat on the tree",
    "I study hard this morning",
    "today is a good day",
    "tomorrow will be a good day",
    "I like coffee, I like book and I like apple",
    "I do not like it",
    "I am kitty, I like bob",
    "I do not care who like bob, but I like kitty",
    "It is coffee time, bring your cup",
]

vectorizer = TfidfVectorizer()
# fit_transform一步到位了，算出所有文档的tf_idf，底下只是在展示，你可以获得训练出的模型中的值
tf_idf = vectorizer.fit_transform(docs)
# 可将vectorizer看做一个训练好的模型，从模型中去除idf，也就是整篇文档的词的属性
print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
# i为索引，方便查找vocabulary的idf值
print("v2i: ", vectorizer.vocabulary_)


q = "I get a coffee cup"
# 这里也是一步到位，算句子的值
qtf_idf = vectorizer.transform([q])
# 计算向量相似度
res = cosine_similarity(tf_idf, qtf_idf)
res = res.ravel().argsort()[-3:]
print("\ntop 3 docs for '{}':\n{}".format(q, [docs[i] for i in res[::-1]]))


i2v = {i: v for v, i in vectorizer.vocabulary_.items()}
# 特意转换为稠密矩阵，做可视化，显示有很多空的
dense_tfidf = tf_idf.todense()
show_tfidf(dense_tfidf, [i2v[i] for i in range(dense_tfidf.shape[1])], "tfidf_sklearn_matrix")