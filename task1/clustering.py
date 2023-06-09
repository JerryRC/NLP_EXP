import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel


MNAME = "bert-base-uncased"
WORDS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_words():
    with open("words.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                WORDS.append(line)


def main():
    load_words()
    tokenizer = BertTokenizer.from_pretrained(MNAME)
    model = BertModel.from_pretrained(MNAME).to(device)
    # 获取词语的嵌入向量
    word_embeddings = []
    for word in WORDS:
        word = word.lower()
        input_ids = tokenizer.encode(word, add_special_tokens=True, truncation=True, padding="max_length", max_length=16, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        word_embeddings.append(embeddings)


    # 使用K-means进行聚类
    num_clusters = 15
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_embeddings)

    # 获取每个词语所属的簇标签
    cluster_labels = kmeans.labels_

    # 将词语和对应的簇标签组合成字典
    word_clusters = {}
    for i, word in enumerate(WORDS):
        word_clusters[word] = cluster_labels[i]
    print("\n\n\n\n\n\n")
    # 打印每个簇中的词语
    for cluster in range(num_clusters):
        print(f"Cluster {cluster}:")
        words_in_cluster = [word for word, label in word_clusters.items() if label == cluster]
        print(words_in_cluster)
        print()


if __name__ == "__main__":
    main()
