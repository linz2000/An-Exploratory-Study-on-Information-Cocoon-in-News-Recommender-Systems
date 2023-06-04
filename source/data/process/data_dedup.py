
import pandas as pd
import os

def deduplicate(data_path, save_path):  # remove duplicate 'clicked_news' item and 'impression' item
                                        # and make sure that 'impression' item is not in 'clicked_news'

    behaviors = pd.read_table(data_path, header=None, usecols=range(5), names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])
    print("dup before:", len(behaviors))
    behaviors.dropna(inplace=True)
    behaviors.reset_index(drop=True)
    print("drop na:", len(behaviors))
    behaviors.drop_duplicates(inplace=True)
    print("dup drop:", len(behaviors))

    for row in behaviors.itertuples():
        clicked_news = row.clicked_news
        impressions = row.impressions
        news_list = clicked_news.strip().split()
        news_set = set(news_list)
        imp_set = set(impressions.strip().split())

        if len(news_list) > len(news_set):
            new_news_str = ' '.join(list(news_set))
            behaviors.at[row.Index, 'clicked_news'] = new_news_str

        new_imp_list = []
        for imp in imp_set:
            news = imp.split('-')[0]
            if news not in news_set:
                new_imp_list.append(imp)
            else:
                # print("candidate news is already in clicked history.")
                pass
        new_imp_str = ' '.join(new_imp_list)
        behaviors.at[row.Index, 'impressions'] = new_imp_str

    print("dup drop:", len(behaviors))
    behaviors.to_csv(save_path, sep="\t", header=False, index=False)


def mergeRow(data_path, save_path): # merge rows of same user
    behaviors = pd.read_table(data_path, header=None, usecols=range(5), names=[
        'impression_id', 'user', 'time',
        'clicked_news', 'impressions'
    ])
    print("merge before:", len(behaviors))
    # print(behaviors[behaviors.isna().any(axis=1)])
    behaviors.dropna(inplace=True)
    print("drop na:", len(behaviors))
    behaviors.reset_index(drop=True)
    behaviors.drop_duplicates(inplace=True)
    print("merge drop:", len(behaviors))

    user2record = dict()
    user2idx = dict()

    for row in behaviors.itertuples():
        user = row.user
        time = row.time
        clicked_news = row.clicked_news
        impressions = row.impressions

        if user not in user2idx:
            user2idx[user] = user2idx.__len__() + 1
            user2record[user] = [[], [], []]
        user2record[user][0].append(time)
        user2record[user][1].append(clicked_news)
        user2record[user][2].append(impressions)

    for user, record in user2record.items():
        for i in range(len(record)):
            user2record[user][i] = ' '.join(record[i])

    records = []
    for user, record in user2record.items():
        records.append([user2idx[user], user, record[0], record[1], record[2]])
    print("user num:",len(user2idx))
    new_behaviors = pd.DataFrame(records)
    print("merge fin:", len(new_behaviors))

    new_behaviors.to_csv(save_path, sep="\t", header=False, index=False)


def merge_and_dedup(behavior_path, merge_save_path, dedup_save_path):
    # step 1: merge
    # step 2: deduplicate
    mergeRow(behavior_path, merge_save_path)
    deduplicate(merge_save_path, dedup_save_path)





