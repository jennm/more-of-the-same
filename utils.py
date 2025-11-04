# contains commonly used functions

from collections import Counter
from calibrated_marked_words import calibrated_marked_words as get_calibrated_marked_words


def pprint(dic, computation=False):
    full_list = []
    if len(dic) == 2:
        computation = dic[1]
        dic = dic[0]
    else:
        computation = None
    for word in sorted(dic, key=lambda x: x[1], reverse=True):
        full_list.append(word[0])
    if computation:
        return full_list, computation
    else:
        return full_list


def compute_calibrated_marked_words(df, occupations, alpha='default', inferred_gender=True, lower_names=set(), remove_names=True, print_words=False):
    if inferred_gender:
        gender_col = 'inferred_gender'
    else:
        gender_col = 'gender'

    dv3_mw = dict()
    dv3_mw_names = dict()
    dv3_mw_occ = dict()
    dv3_mw_occ_names = dict()
    dv3_mw_by_occ = dict()
    dv3_mw_by_occ_names = dict()
    for occupation in occupations:
        dv3_mw_by_occ[occupation] = dict()
        dv3_mw_by_occ_names[occupation] = dict()

        for g in df[gender_col].unique():
            if alpha == 'default':
                outs = pprint(get_calibrated_marked_words(
                    df, [g], [gender_col], ['M'], occupation))
            else:
                outs = pprint(get_calibrated_marked_words(
                    df, [g], [gender_col], ['M'], occupation, alpha))
            new_outs = list()
            curr_names = list()
            for word in outs:
                if remove_names and (word in lower_names or word[:-1] in lower_names):
                    curr_names.append(word)
                else:
                    new_outs.append(word)
            if g in dv3_mw:
                dv3_mw[g].append([new_outs])
                dv3_mw_names[g].append([[curr_names]])
                if g in dv3_mw_by_occ[occupation]:
                    dv3_mw_by_occ[occupation][g].append(new_outs)
                    dv3_mw_by_occ_names[occupation][g].append(curr_names)
                else:
                    dv3_mw_by_occ[occupation][g] = new_outs
                    dv3_mw_by_occ_names[occupation][g] = curr_names
                if occupation in dv3_mw_occ[g]:
                    dv3_mw_occ[g][occupation].append(new_outs)
                    dv3_mw_occ_names[g][occupation].append(curr_names)
                else:
                    dv3_mw_occ[g][occupation] = new_outs
                    dv3_mw_occ_names[g][occupation] = curr_names
            else:
                dv3_mw[g] = [[new_outs]]
                dv3_mw_names[g] = [[curr_names]]
                dv3_mw_by_occ[occupation][g] = new_outs
                dv3_mw_by_occ_names[occupation][g] = curr_names
                dv3_mw_occ[g] = {occupation: new_outs}
                dv3_mw_occ_names[g] = {occupation: curr_names}
        temps = []
        temps_names = []
        for g in df[gender_col].unique():
            if alpha == 'default':
                temp = pprint(get_calibrated_marked_words(
                    df, ['M'], [gender_col], [g], occupation))
            else:
                temp = pprint(get_calibrated_marked_words(
                    df, ['M'], [gender_col], [g], occupation, alpha))
            new_temp = list()
            curr_names = list()
            for word in temp:
                if remove_names and (word in lower_names or word[:-1] in lower_names):
                    curr_names.append(word)
                else:
                    new_temp.append(word)
            temps.extend(new_temp)
            temps_names.extend(curr_names)

        seen = Counter(temps).most_common()
        seen_names = Counter(temps_names).most_common()
        num_seen = len(df[gender_col].unique()) - 1
        m_words = [w for w, c in seen if c == num_seen]
        m_words_names = [w for w, c in seen_names if c == num_seen]
        if 'M' in dv3_mw:
            dv3_mw['M'].append(m_words)
            dv3_mw_names['M'].append(m_words_names)
        else:
            dv3_mw['M'] = m_words
            dv3_mw_names['M'] = m_words_names
        dv3_mw_by_occ[occupation]['M'] = m_words
        dv3_mw_by_occ_names[occupation]['M'] = m_words_names

    if print_words:
        print(dv3_mw)
    return dv3_mw, dv3_mw_names, dv3_mw_by_occ, dv3_mw_by_occ_names, dv3_mw_occ, dv3_mw_occ_names
