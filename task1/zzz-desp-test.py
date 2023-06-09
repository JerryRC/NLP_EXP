import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from glove import Corpus, Glove

# 准备领域相关和通用语料库的数据

# 领域相关语料库
domain_corpus = [
    ['apple', 'fruit', 'red', 'juice'],
    ['banana', 'fruit', 'yellow', 'smoothie'],
    ['orange', 'fruit', 'citrus', 'vitamin'],
    ['dog', 'pet', 'friendly', 'loyal'],
    ['cat', 'pet', 'independent', 'playful'],
    ['bird', 'pet', 'feathers', 'singing']
]

# 通用语料库
general_corpus = [
    ['apple', 'fruit', 'company', 'technology'],
    ['banana', 'fruit', 'tropical', 'snack'],
    ['orange', 'fruit', 'citrus', 'color'],
    ['dog', 'pet', 'animal', 'companion'],
    ['cat', 'pet', 'feline', 'whiskers'],
    ['bird', 'pet', 'avian', 'wings']
]

# 训练word2vec模型
w2v_model_domain = Word2Vec(domain_corpus, vector_size=100, window=5, min_count=1)
w2v_model_general = Word2Vec(general_corpus, vector_size=100, window=5, min_count=1)

# 训练Glove模型
corpus_domain = Corpus()
corpus_domain.fit(domain_corpus, window=10)
glove_model_domain = Glove(no_components=100, learning_rate=0.05)
glove_model_domain.fit(corpus_domain.matrix, epochs=10, no_threads=4, verbose=True)

corpus_general = Corpus()
corpus_general.fit(general_corpus, window=10)
glove_model_general = Glove(no_components=100, learning_rate=0.05)
glove_model_general.fit(corpus_general.matrix, epochs=10, no_threads=4, verbose=True)

# 获取领域相关和通用语料库的词向量
word_vectors_domain_w2v = w2v_model_domain.wv
word_vectors_general_w2v = w2v_model_general.wv


word_vectors_domain_glove = {word: glove_model_domain.word_vectors[glove_model_domain.dictionary[word]] for word in glove_model_domain.dictionary}
word_vectors_general_glove = {word: glove_model_general.word_vectors[glove_model_general.dictionary[word]] for word in glove_model_general.dictionary}

# 执行聚类算法
k = 2  # 聚类数量

# 使用k-means算法进行聚类
kmeans_w2v_domain = KMeans(n_clusters=k, random_state=42)
kmeans_w2v_domain.fit(word_vectors_domain_w2v.vectors)

kmeans_w2v_general = KMeans(n_clusters=k, random_state=42)
kmeans_w2v_general.fit(word_vectors_general_w2v.vectors)

kmeans_glove_domain = KMeans(n_clusters=k, random_state=42)
kmeans_glove_domain.fit(np.array(list(word_vectors_domain_glove.values())))

kmeans_glove_general = KMeans(n_clusters=k, random_state=42)
kmeans_glove_general.fit(np.array(list(word_vectors_general_glove.values())))

# 输出聚类结果
print("Word2Vec Clustering Results:")
print("Domain Corpus Clusters:")
for word, label in zip(word_vectors_domain_w2v.vocab, kmeans_w2v_domain.labels_):
    print(f"{word}: {label}")

print("General Corpus Clusters:")
for word, label in zip(word_vectors_general_w2v.vocab, kmeans_w2v_general.labels_):
    print(f"{word}: {label}")

print("Glove Clustering Results:")
print("Domain Corpus Clusters:")
for word, label in zip(word_vectors_domain_glove, kmeans_glove_domain.labels_):
    print(f"{word}: {label}")

print("General Corpus Clusters:")
for word, label in zip(word_vectors_general_glove, kmeans_glove_general.labels_):
    print(f"{word}: {label}")
