
from sklearn.metrics.pairwise import cosine_distances
def subset_similarity_metric(canidate_set,target_set,word_model,exclude_words=True, words_to_exclude=[]):
    most_similar_words = list()
    for inferred_word in canidate_set:
        if exclude_words:
            if inferred_word in words_to_exclude:
                continue
        most_similar = 2
        for marked_word in target_set:
            temp = abs(cosine_distances(word_model.wv[inferred_word].reshape(1,-1), word_model.wv[marked_word].reshape(1,-1)))[0][0]
            if most_similar > temp:
                most_similar = temp
        most_similar_words.append(most_similar)
    return sum(most_similar_words)/len(most_similar_words)

def subset_representation_bias_score(c, t1, t2, word_model, exclude_words=True, words_to_exclude=[]):
    return subset_similarity_metric(c, t1, word_model, exclude_words=exclude_words, words_to_exclude=words_to_exclude) - subset_similarity_metric(c, t2, word_model, exclude_words=exclude_words, words_to_exclude=words_to_exclude)