import multiprocessing
import pickle
from glove import Corpus, Glove
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

WORDS = []
PKL_FILE = "glove_model/word_vectors_f0_e10_legal.pkl"
MIN_FREQUENCY = 0
EPOCH = 10
LR = 0.05
NO_COMPONENTS = 128
# CORPUS = "text8"
CORPUS = "legal_case_reports"


def train_glove():

    # # 构建词频字典
    # word_frequency = Counter()
    # with open(CORPUS, "r") as f:
    #     for line in f:
    #         line = line.strip().split()
    #         word_frequency.update(line)
    # # ================== 保存词频字典 ================== #
    # with open('word_frequency_legal.pkl', "wb") as f:
    #     pickle.dump(word_frequency, f)
    # # ================== 只需要做一次 ================== #
    with open('word_frequency_legal.pkl', 'rb') as f:
        word_frequency = pickle.load(f)

    # 替换低频词为UNK标记
    threshold = MIN_FREQUENCY
    unk_token = "UNK"
    sentences = []
    with open(CORPUS, "r") as f:
        for line in f:
            line = line.strip().split()
            for i in range(len(line)):
                if word_frequency[line[i]] < threshold:
                    line[i] = unk_token
            sentences.append(line)

    # 创建GloVe语料库对象
    corpus = Corpus()
    corpus.fit(sentences, window=10)

    # 训练GloVe模型
    glove = Glove(no_components=NO_COMPONENTS, learning_rate=LR)
    glove.fit(corpus.matrix, epochs=EPOCH, no_threads=multiprocessing.cpu_count(), verbose=True)
    glove.add_dictionary(corpus.dictionary)

    # 保存训练好的词向量
    word_vectors = {word: glove.word_vectors[glove.dictionary[word]] for word in glove.dictionary}
    with open(PKL_FILE, "wb") as f:
        pickle.dump(word_vectors, f)


def load_words():
    with open("words.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                WORDS.append(line)


def test_glove():
    # 加载训练好的词向量
    with open(PKL_FILE, "rb") as f:
        loaded_word_vectors = pickle.load(f)

    load_words()
    # 获取词语的嵌入向量
    word_embeddings = []
    words = []
    for word in WORDS:
        word = word.lower()
        try:
            embeddings = loaded_word_vectors[word]
            words.append(word)
        except KeyError:
            print(f"{word} not in vocabulary, skip it.")
            continue
            embeddings = np.random.uniform(-0.1, 0.1, size=(NO_COMPONENTS,))
            
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


if __name__=='__main__':
    train_glove()
    test_glove()
