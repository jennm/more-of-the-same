"""
Running this file obtains the words that distinguish a target group from the corresponding
unmarked ones.
Example usage: (To obtain the words that differentiate the 'Asian F' category)
python3 marked_words.py ../generated_personas.csv --target_val 'an Asian' F --target_col race gender --unmarked_val 'a White' M
"""

from collections import Counter, defaultdict
from nltk.corpus import brown
import argparse
import heapq
import math
import nltk
import numpy as np
import pandas as pd

def get_log_odds(df1, df2, df0,lower=True, prior=True, prior_type='hybrid', return_computation=False):
    """Monroe et al. Fightin' Words method to identify top words in df1 and df2
    against df0 as the background corpus"""
    
    overall_common_words = {'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are', 'from', 'at', 'as', 'your', 'am', 'an', 'my', 'are'}#, 't', 'h', '', 'r', 'm', 'y'}

    if lower:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    else:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])

    if '' in counts1:
        del counts1['']
    if '' in counts2:
        del counts2['']
    if '' in prior:
        del prior['']
    c_english = .0001625
    c_topic = 0.005078125
    if prior_type == 'english':
        prior = nltk.FreqDist([w.lower() for w in brown.words()])
        c = .225
    elif prior_type == 'hybrid':
        english_prior = nltk.FreqDist([w.lower() for w in brown.words()])
        p = 0.25
        c_english = .225
        c_topic = .45
        c = p * c_topic + (1 - p) * c_english
        for word in prior.keys():
            prior[word] = int(p * prior[word] + (1 - p) * english_prior[word])
    else:
        c = .45

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    reg1 = sum(prior.values())
    reg2 = sum(prior.values())

    prior_min_heap = []
    counts1_min_heap = []
    counts2_min_heap = []
    num_common_words = 20

    # find most common shared words between df1 and df2
    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)
        if not prior:
            prior[word] = 0
        else:
            heapq.heappush(prior_min_heap, (prior[word], word))
            if len(prior_min_heap) > num_common_words:
                heapq.heappop(prior_min_heap)

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1
        else:
            heapq.heappush(counts1_min_heap, (counts1[word], word))
            if len(counts1_min_heap) > num_common_words:
                heapq.heappop(counts1_min_heap)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1
        else:
            heapq.heappush(counts2_min_heap, (counts2[word], word))
            if len(counts2_min_heap) > num_common_words:
                heapq.heappop(counts2_min_heap)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())
    
    if n1 == 0 or n2 == 0:
        return delta

    prior_top_words_set = set(map(lambda x: x[1], prior_min_heap))
    counts1_top_words_set = set(map(lambda x: x[1], counts1_min_heap))
    counts2_top_words_set = set(map(lambda x: x[1], counts2_min_heap))
    common_words = (prior_top_words_set & counts1_top_words_set & counts2_top_words_set) | overall_common_words

    p_word = 0
    g1_word = 0
    g2_word = 0

    for word in common_words:
        p_word += float(prior[word])
        g1_word += float(counts1[word])
        g2_word += float(counts2[word])

    reg1 = c * p_word / g1_word
    reg2 = c * p_word / g2_word

    display_nums=False
    if display_nums:
        print(f'c: {c}, p: {p_word}, g1: {g1_word}, g2: {g2_word}, r1: {reg1}, r2: {reg2}')

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + float(prior[word])/reg1) / (( n1 + float(nprior)/reg1 ) - (counts1[word] + float(prior[word])/reg1))
            l2 = float(counts2[word] + float(prior[word])/reg2) / (( n2 + float(nprior)/reg2 ) - (counts2[word] + float(prior[word])/reg2))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])/reg1) + 1/(float(counts2[word]) + float(prior[word])/reg2)
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if return_computation:
        return delta, [counts1, counts2, prior, n1, n2, nprior, reg1, reg2]
    else:
        return delta




def calibrated_marked_words(df, target_val, target_col, unmarked_val,occupation=None,return_computation=False):

    """Get words that distinguish the target group (which is defined as having 
    target_group_vals in the target_group_cols column of the dataframe) 
    from all unmarked_attrs (list of values that correspond to the categories 
    in unmarked_attrs)"""

    grams = dict()
    thr = 1.96 #z-score threshold
    if occupation:
        df = df[df.occupation == occupation]

    subdf = df.copy()
    for i in range(len(target_val)):
        subdf = subdf.loc[subdf[target_col[i]]==target_val[i]]


    for i in range(len(unmarked_val)):
        if return_computation:
            delt, computation = get_log_odds(subdf['text'], df.loc[df[target_col[i]]==unmarked_val[i]]['text'],df['text'],return_computation=return_computation)
        else:
            delt = get_log_odds(subdf['text'], df.loc[df[target_col[i]]==unmarked_val[i]]['text'],df['text'],return_computation=return_computation) #first one is the positive-valued one
        
        c1 = []
        c2 = []
        for k,v in delt.items():
            if v > thr:
                c1.append([k,v])
            elif v < -thr:
                c2.append([k,v])

        if 'target' in grams:
            grams['target'].extend(c1)
        else:
            grams['target'] = c1
        if unmarked_val[i] in grams:
            grams[unmarked_val[i]].extend(c2)
        else:
            grams[unmarked_val[i]] = c2
    grams_refine = dict()
    for r in grams.keys():
        temp = []
        thr = len(unmarked_val) 
        for k,v in Counter([word for word, z in grams[r]]).most_common():
            if v >= thr:
                z_score_sum = np.sum([z for word, z in grams[r] if word == k])
                temp.append([k, z_score_sum])

        grams_refine[r] = temp
    if return_computation:
        return grams_refine['target'], computation
    else:
        return grams_refine['target']


def main():
    parser = argparse.ArgumentParser(description="Just an example",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("target_filename", help="Generated personas file")
    parser.add_argument("--target_val",nargs="*", 
    type=str,
    default=[''], help="List of demographic attribute(s) for target group of interest")
    parser.add_argument("--target_col", nargs="*",
    type=str,
    default=[''],help="List of demographic categories that distinguish target group")
    parser.add_argument("--unmarked_val", nargs="*",
    type=str,
    default=[''],help="List of unmarked default values for relevant demographic categories")
    parser.add_argument("--occupation", nargs="*", type=str,default=None)
    parser.add_argument("--verbose", action='store_true',help="If set to true, prints out top words calculated by Fightin' Words")

    args = parser.parse_args()

    filename = args.target_filename
    target_val = args.target_val
    target_col = args.target_col
    unmarked_val = args.unmarked_val

    assert len(target_val) == len(target_col) == len(unmarked_val)
    assert len(target_val) > 0
    df = pd.read_csv(filename)

    # Optional: filter out unwanted prompts
    # df = df.loc[~df['prompt'].str.contains('you like')]
    top_words = calibrated_marked_words(df, target_val, target_col, unmarked_val,occupation=args.occupation,verbose=args.verbose)
    print("Top words:")
    print(top_words)

if __name__ == '__main__':
    
    main()
