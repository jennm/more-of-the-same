from collections import defaultdict

import pandas as pd


def associate_gender(text):
    temp_df = pd.DataFrame(columns=['text'], index=range(1))
    temp_df.iloc[0] = text

    counts = defaultdict(int,[[i,j] for i,j in temp_df['text'].str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    nb_present = 'nonbinary' in text or 'non-binary' in text or 'they/them' in text
    ms_present = int('ms.' in text) and counts['ms']
    c_female = counts['she'] + counts['her'] + counts['hers'] + counts['herself'] + ms_present + counts['mrs'] + counts['female']
    c_male = counts['he'] + counts['his'] + counts['him'] + counts['himself'] + counts['mr'] + counts['male']
    c_neutral = counts['they'] + counts['their']
    g = None
    if nb_present and c_neutral > c_female + c_male:
        g = 'N'
    elif not nb_present and c_male > c_female or c_male > c_female + c_neutral:
        g = 'M'
    elif not nb_present and c_female > c_male or c_female > c_male + c_neutral:
        g = 'F'
    return g
