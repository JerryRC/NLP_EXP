import multiprocessing
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np


WORDS = []
MODEL_FILE = "w2v_model/word2vec_model_f0_e10_legal.bin"
MIN_FREQUENCY = 0
EPOCH = 10
NO_COMPONENTS = 128
# CORPUS = "text8"
CORPUS = "legal_case_reports"


def train():

    # 准备训练文本数据
    sentences = []
    with open(CORPUS, "r") as f:
        for line in f:
            line = line.strip().split()
            sentences.append(line)

    # 配置Word2Vec模型参数
    min_count = MIN_FREQUENCY  # 忽略出现次数低于min_count的词语
    window = 10  # 上下文窗口大小

    # 训练Word2Vec模型
    model = Word2Vec(sentences, vector_size=NO_COMPONENTS, min_count=min_count, window=window, epochs=EPOCH, workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(MODEL_FILE)


def load_words():
    with open("words.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                WORDS.append(line)


def test():
    # 加载训练好的词向量模型
    loaded_model = Word2Vec.load(MODEL_FILE)

    load_words()
    # 获取词语的嵌入向量
    word_embeddings = []
    words = []
    for word in WORDS:
        word = word.lower()
        try:
            embeddings = loaded_model.wv[word]
            words.append(word)
        except KeyError:
            print(f"{word} not in vocabulary, skip it.")
            continue
            embeddings = np.random.uniform(-0.1, 0.1, size=(NO_COMPONENTS,)).astype(np.float32)
            
            # print(f"{word} not in vocabulary, use UNK instead.")
            # embeddings = loaded_word_vectors["UNK"]

        word_embeddings.append(embeddings)
        
    # 使用K-means进行聚类
    num_clusters = 15
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_embeddings)

    # 获取每个词语所属的簇标签
    cluster_labels = kmeans.labels_

    # 将词语和对应的簇标签组合成字典
    word_clusters = {}
    for i, word in enumerate(words):
        word_clusters[word] = cluster_labels[i]

    # 打印每个簇中的词语
    for cluster in range(num_clusters):
        print(f"Cluster {cluster}:")
        words_in_cluster = [word for word, label in word_clusters.items() if label == cluster]
        print(words_in_cluster)
        print()


def traditional_test():
    loaded_model = Word2Vec.load(MODEL_FILE)

    female_king = loaded_model.wv.most_similar_cosmul(positive='king woman'.split(), negative='man'.split(), topn=5)
    for ii, (word, score) in enumerate(female_king):
        print("{}. {} ({:1.2f})".format(ii+1, word, score))


if __name__ == '__main__':
    train()
    test()
    # traditional_test()
