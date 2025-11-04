from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

import argparse
import matplotlib.pyplot as plt
import pickle


def find_sillouette_score(data, min_k, max_k, inc, ss_scores=dict(), print_ss_scores=True):
    for k in range(min_k, max_k, inc):
        print(k)
        kmeans = KMeans(init="k-means++", n_clusters=k, random_state=42)
        kmeans.fit(data)
        ss_scores[k] = silhouette_score(data, kmeans.labels_)
    if print_ss_scores:
        print(ss_scores)
    return ss_scores


def print_graph(data, title, y_label, filename="ss_scores.png"):
    plt.figure(figsize=(8, 5))
    data_keys = list(data.keys())
    k_values = sorted(data_keys)
    scores = list()
    for k in k_values:
        scores.append(data[k])
    plt.plot(k_values, scores, marker='o')
    plt.title(title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default="word_vectors_of_data.pkl", type=str)
    parser.add_argument('--prev_vals', default=None)
    parser.add_argument('--min_k', default=100, type=int)
    parser.add_argument('--max_k', default=2300, type=int)
    parser.add_argument('--inc', default=100, type=int)

    args = parser.parse_args()

    with open(args.file_name, "rb") as f:
        data = pickle.load(f)

    print("finding Silhouette Score")
    ss_scores = dict()
    if args.prev_vals:
        with open(args.file_name, "rb") as f:
            ss_scores = pickle.load(f)

    ss_scores = find_sillouette_score(data, args.min_k, args.max_k, args.inc, ss_scores)
    
    with open("find-opt-cluster-num-ss.pkl", "wb") as f:
        pickle.dump(ss_scores, f)

    print_graph(ss_scores, "Silhouette Score vs # of Clusters", "Silhouette Score")

        

