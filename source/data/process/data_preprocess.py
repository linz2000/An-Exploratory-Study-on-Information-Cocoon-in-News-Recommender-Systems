from config import model_name

import pandas as pd
# import swifter
import json
import math
from tqdm import tqdm
from os import path
from pathlib import Path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import importlib

# import nltk
# nltk.download('punkt')

try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


def parse_behaviors_origin(source, target, user2int_path, mode='train'):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])


    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    if mode == 'train':
        user2int = {}
        for row in behaviors.itertuples(index=False):
            if row.user not in user2int:
                user2int[row.user] = len(user2int) + 1

        pd.DataFrame(user2int.items(), columns=['user',
                                                'int']).to_csv(user2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_users` in `config.py` into 1 + {len(user2int)}'
        )
    else:
        user2int = dict(pd.read_table(user2int_path).values.tolist())

    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]

    for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
        positive = iter([x for x in row.impressions if x.endswith('1')])
        negative = [x for x in row.impressions if x.endswith('0')]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(config.negative_sampling_ratio):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, 'impressions'] = pairs

    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())


    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

def generate_ngcf_behaviors(source, user2int_path, news2int_path, target, mode):
    """
        Parse behaviors file in training set.
        Args:
            source: source behaviors file
            target: target behaviors file
        """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])

    # behaviors.dropna(inplace=True)
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions.fillna(' ', inplace=True)
    behaviors.clicked_news = behaviors.clicked_news.str.split()
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = dict(pd.read_table(user2int_path).values.tolist()) # idx start from 1
    news2int = dict(pd.read_table(news2int_path).values.tolist()) # idx start from 1

    user_num = max(user2int.values())
    user_clicked_news_set = [ set() for i in range(user_num)]

    for row in behaviors.itertuples():
        user_id = user2int[row.user] - 1
        if mode == 'train':
            pos_news_list = [imp.split('-')[0] for imp in row.impressions if imp.split('-')[1] == '1'] + row.clicked_news
        else:
            pos_news_list = [imp.split('-')[0] for imp in row.impressions if imp.split('-')[1] == '1']
        for pos_news in pos_news_list:
            user_clicked_news_set[user_id].add( str(news2int[pos_news]-1 ))

    # save to file
    with open(target, 'w') as outf:
        for uid in range(user_num):
            clicked_news = user_clicked_news_set[uid]
            if clicked_news.__len__() > 0:
                outf.write(str(uid)+' '+ ' '.join(clicked_news) + '\n')


def generate_fm_behaviors(source, news_parsed_path, user2int_path, news2int_path, target, mode):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
    """
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])

    # behaviors.dropna(inplace=True)
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions.fillna(' ', inplace=True)
    behaviors.clicked_news = behaviors.clicked_news.str.split()
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = dict(pd.read_table(user2int_path).values.tolist())
    news2int = dict(pd.read_table(news2int_path).values.tolist())

    news_df = pd.read_table(news_parsed_path,
                            header=0,
                            usecols=[0,1,2])
    news2cat = dict(zip(news_df['id'], news_df['category']))
    news2subcat = dict(zip(news_df['id'], news_df['subcategory']))

    data = []
    for row in behaviors.itertuples():
        user_id = user2int[row.user]

        if mode == 'train':
            pos_news_list = [imp.split('-')[0] for imp in row.impressions if imp.split('-')[1] == '1'] + row.clicked_news
        else:
            pos_news_list = [imp.split('-')[0] for imp in row.impressions if imp.split('-')[1] == '1']
        neg_news_list = [imp.split('-')[0] for imp in row.impressions if imp.split('-')[1] == '0']
        positive = iter(pos_news_list)
        negative = iter(neg_news_list)
        news_list = []
        label_list = []

        try:
            while True:
                pos_news = next(positive)
                news_list.append(pos_news)
                label_list.append(1)
                neg_news = next(negative)
                news_list.append(neg_news)
                label_list.append(0)
                if news_list.__len__() >= 10:
                    break
        except StopIteration:
            pass

        for i in range(len(news_list)):
            news = news_list[i]
            data.append([user_id, news2int[news], news2cat[news] if news in news2cat else 0,
                         news2subcat[news] if news in news2subcat else 0, label_list[i]])

    behaviors = pd.DataFrame(data, columns=['user', 'news', 'category', 'subcategory', 'label'])

    behaviors.to_csv(
        target,
        sep='\t',
        index=False)


def parse_news(source, target, category2int_path, subcategory2int_path, word2int_path,
               entity2int_path, news2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, subcategory2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, subcategory2int_path, word2int_path, entity2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 6, 7],
                         quoting=csv.QUOTE_NONE,
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])  # TODO try to avoid csv.QUOTE_NONE
    news.title_entities.fillna('[]', inplace=True)
    news.abstract_entities.fillna('[]', inplace=True)
    news.fillna(' ', inplace=True)

    def parse_row(row):
        new_row = [
            row.id,
            category2int[row.category] if row.category in category2int else 0,
            subcategory2int[row.subcategory]
            if row.subcategory in subcategory2int else 0,
            [0] * config.num_words_title, [0] * config.num_words_abstract,
            [0] * config.num_words_title, [0] * config.num_words_abstract
        ]

        # Calculate local entity map (map lower single word to entity)
        local_entity_map = {}
        for e in json.loads(row.title_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                    'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]
        for e in json.loads(row.abstract_entities):
            if e['Confidence'] > config.entity_confidence_threshold and e[
                    'WikidataId'] in entity2int:
                for x in ' '.join(e['SurfaceForms']).lower().split():
                    local_entity_map[x] = entity2int[e['WikidataId']]

        try:
            for i, w in enumerate(word_tokenize(row.title.lower())):
                if w in word2int:
                    new_row[3][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[5][i] = local_entity_map[w]
        except IndexError:
            pass

        try:
            for i, w in enumerate(word_tokenize(row.abstract.lower())):
                if w in word2int:
                    new_row[4][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[6][i] = local_entity_map[w]
        except IndexError:
            pass

        return pd.Series(new_row,
                         index=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])

    if mode == 'train':
        category2int = {}
        subcategory2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}
        news2int = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in subcategory2int:
                subcategory2int[row.subcategory] = len(subcategory2int) + 1

            if row.id not in news2int:
                news2int[row.id] = len(news2int) + 1

            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for e in json.loads(row.title_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

            for e in json.loads(row.abstract_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

        for k, v in word2freq.items():
            if v >= config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        parsed_news = news.swifter.apply(parse_row, axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `config.py` into 1 + {len(category2int)}'
        )

        pd.DataFrame(subcategory2int.items(),
                     columns=['subcategory', 'int']).to_csv(subcategory2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_subcategories` in `config.py` into 1 + {len(subcategory2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `config.py` into 1 + {len(word2int)}'
        )

        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
        print(
            f'Please modify `num_entities` in `config.py` into 1 + {len(entity2int)}'
        )

        pd.DataFrame(news2int.items(),
                     columns=['news', 'int']).to_csv(news2int_path,
                                                       sep='\t',
                                                       index=False)

    elif mode == 'test':
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        subcategory2int = dict(pd.read_table(subcategory2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        parsed_news = news.swifter.apply(parse_row, axis=1)
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')

def remove_new_news(behaviors_path, user2int_path, news2int_path, target):

    user2int = dict(pd.read_table(user2int_path).values.tolist())
    news2int = dict(pd.read_table(news2int_path).values.tolist())

    behaviors = pd.read_table(
        behaviors_path,
        header=None,
        names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
    behaviors.dropna(inplace=True)
    behaviors.reset_index(drop=True)
    behaviors.clicked_news = behaviors.clicked_news.str.split()
    behaviors.impressions = behaviors.impressions.str.split()

    del_index_list =[]
    for row in behaviors.itertuples():
        if row.user not in user2int:
            del_index_list.append(row.Index)

    behaviors.drop(index=del_index_list, inplace=True)

    new_records = []
    for row in behaviors.itertuples(index=False):
        try:
            clicked_news_list = [news for news in row.clicked_news if news in news2int]
            impressions_list = [imp for imp in row.impressions if imp.split('-')[0] in news2int]
        except:
            print(row.clicked_news)
            print(row.impressions)
            exit(0)

        new_records.append([' '.join(clicked_news_list) if len(clicked_news_list) > 0 else pd.NA,
                            ' '.join(impressions_list) if len(impressions_list) > 0 else pd.NA])

    behaviors[['clicked_news', 'impressions']] = new_records

    behaviors.dropna(inplace=True)
    behaviors.reset_index(drop=True, inplace=True)
    behaviors['impression_id'] = range(1, 1 + len(behaviors))

    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        header=False)


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    word2int = pd.read_table(word2int_path, na_filter=False, index_col='word')
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(config.word_embedding_dim))
    # word, vector
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
                                merged.index.values)
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), config.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    print(final_embedding.values.shape)
    np.save(target, final_embedding.values)

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}'
    )


def transform_entity_embedding(source, target, entity2int_path):
    """
    Args:
        source: path of embedding file
        target: path of transformed embedding file in numpy format
        entity2int_path
    """
    entity_embedding = pd.read_table(source, header=None)
    entity_embedding['vector'] = entity_embedding.iloc[:,
                                                       1:101].values.tolist()
    entity_embedding = entity_embedding[[0, 'vector'
                                         ]].rename(columns={0: "entity"})

    entity2int = pd.read_table(entity2int_path)
    merged_df = pd.merge(entity_embedding, entity2int,
                         on='entity').sort_values('int')
    entity_embedding_transformed = np.random.normal(
        size=(len(entity2int) + 1, config.entity_embedding_dim))
    for row in merged_df.itertuples(index=False):
        entity_embedding_transformed[row.int] = row.vector
    np.save(target, entity_embedding_transformed)

def process_ngcf_data(train_dir, val_dir, test_dir):

    print("generate ngcf data")

    generate_ngcf_behaviors(path.join(train_dir, 'behaviors.tsv'),
                            path.join(train_dir, 'user2int.tsv'),
                            path.join(train_dir, 'news2int.tsv'),
                            path.join(train_dir, 'ngcf_behaviors.txt'),
                            mode='train')

    generate_ngcf_behaviors(path.join(val_dir, 'behaviors_no_new.tsv'),
                            path.join(train_dir, 'user2int.tsv'),
                            path.join(train_dir, 'news2int.tsv'),
                            path.join(val_dir, 'ngcf_behaviors.tsv'),
                            mode='val')

    generate_ngcf_behaviors(path.join(test_dir, 'behaviors_no_new.tsv'),
                            path.join(train_dir, 'user2int.tsv'),
                            path.join(train_dir, 'news2int.tsv'),
                            path.join(test_dir, 'ngcf_behaviors.tsv'),
                            mode='test')


def process_fm_data(train_dir, val_dir, test_dir):

    print("generate fm data")
    generate_fm_behaviors(path.join(train_dir, 'behaviors.tsv'),
                          path.join(train_dir, 'news_parsed.tsv'),
                          path.join(train_dir, 'user2int.tsv'),
                          path.join(train_dir, 'news2int.tsv'),
                          path.join(train_dir, 'fm_behaviors.tsv'),
                          mode='train')

    generate_fm_behaviors(path.join(val_dir, 'behaviors_no_new.tsv'),
                          path.join(train_dir, 'news_parsed.tsv'),
                          path.join(train_dir, 'user2int.tsv'),
                          path.join(train_dir, 'news2int.tsv'),
                          path.join(val_dir, 'fm_behaviors.tsv'),
                          mode='val')

    generate_fm_behaviors(path.join(test_dir, 'behaviors_no_new.tsv'),
                          path.join(train_dir, 'news_parsed.tsv'),
                          path.join(train_dir, 'user2int.tsv'),
                          path.join(train_dir, 'news2int.tsv'),
                          path.join(test_dir, 'fm_behaviors.tsv'),
                          mode='test')

def process_data():
    train_dir = '../MIND/train'
    val_dir = '../MIND/val'
    test_dir = '../MIND/test'

    print('Process data for training')

    print('Parse behaviors')
    parse_behaviors_origin(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'behaviors_parsed.tsv'),
                    path.join(train_dir, 'user2int.tsv'))

    print('Parse news')
    parse_news(path.join(train_dir, 'news.tsv'),
               path.join(train_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'subcategory2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               path.join(train_dir, 'news2int.tsv'),
               mode='train')

    print('Generate word embedding')
    generate_word_embedding(
        f'../MIND/glove/glove.840B.{config.word_embedding_dim}d.txt',
        path.join(train_dir, 'pretrained_word_embedding.npy'),
        path.join(train_dir, 'word2int.tsv'))

    print('Transform entity embeddings')
    transform_entity_embedding(
        path.join(train_dir, 'entity_embedding.vec'),
        path.join(train_dir, 'pretrained_entity_embedding.npy'),
        path.join(train_dir, 'entity2int.tsv'))


    print('\nProcess data for validation')

    remove_new_news(path.join(val_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'user2int.tsv'),
                    path.join(train_dir, 'news2int.tsv'),
                    path.join(val_dir, 'behaviors_no_new.tsv'))


    print('\nProcess data for test')

    remove_new_news(path.join(test_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'user2int.tsv'),
                    path.join(train_dir, 'news2int.tsv'),
                    path.join(test_dir, 'behaviors_no_new.tsv'))

    process_fm_data(train_dir, val_dir, test_dir)

    process_ngcf_data(train_dir, val_dir, test_dir)


# process_data()


# Please modify `num_categories` in `config.py` into 1 + 18
# Please modify `num_subcategories` in `config.py` into 1 + 285
# Please modify `num_words` in `config.py` into 1 + 101232
# Please modify `num_entities` in `config.py` into 1 + 21842