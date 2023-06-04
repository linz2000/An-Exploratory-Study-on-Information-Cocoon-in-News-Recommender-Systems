from model_interface import Content_Rec_Interface, FM_Rec_Interface, FM_Content_Rec_Interface, NGCF_Rec_Interface
from config import BaseConfig, MINDConfig, model_name, dataset_name, device
from dataset import BaseDataset, FMDataset, FM_Content_Dataset, NGCF_Dataset
from config import emb_model

import pandas as pd
import numpy as np
import math
import torch
import faiss
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import random

def getTopKSim(ue, ie, k):
    index = faiss.IndexFlatIP(ue.shape[1])
    index.add(ie)
    D, I = index.search(ue, k) 
    
    return D, I

def getRandomBeta(u, s=0.05):
    try:
        a = ((1 - u) / (s * s) - 1 / u) * u * u
        b = a * (1 / u - 1)
        return np.random.beta(a, b)
    except:
        return u

class NewsRec():
    def __init__(self, news_path, behaviors_path, test_behaviors_path, test_fm_behaviors_path,
                 train_ngcf_behaviors_dir, train_ngcf_behaviors_path, val_ngcf_behaviors_path, test_ngcf_behaviors_path,
                 user2int_path, news2int_path, news2emb_path, checkpoint_dir):
        self.news_path = news_path
        self.behaviors_path = behaviors_path  
        self.test_behaviors_path = test_behaviors_path
        self.test_fm_behaviors_path = test_fm_behaviors_path
        self.train_ngcf_behaviors_dir = train_ngcf_behaviors_dir
        self.train_ngcf_behaviors_path = train_ngcf_behaviors_path
        self.val_ngcf_behaviors_path = val_ngcf_behaviors_path
        self.test_ngcf_behaviors_path = test_ngcf_behaviors_path
        self.user2int_path = user2int_path
        self.news2int_path = news2int_path
        self.news2emb_path = news2emb_path

        self.behaviors, self.test_behaviors, self.user2news_set, self.user_list = self.__load_behaviors() 
        self.origin_behaviors = self.behaviors.copy()
        self.news, self.news_list = self.__load_news()
        self.user2int, self.news2int, self.user2feat_id_list, self.news2feat_id_list = self.__gen_fm_data()
        if model_name == 'DFM_CONTENT'  or model_name == 'DFM_CONTENT_ONLY' :
            self.news_int2emb = self.test_behaviors.get_news_int2emb()

        
        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            self.model = Content_Rec_Interface(news_path, behaviors_path, user2int_path, checkpoint_dir)
        elif model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_ID' :
            self.model = FM_Rec_Interface(checkpoint_dir)
        elif model_name == 'DFM_CONTENT'  or model_name == 'DFM_CONTENT_ONLY' :
            self.model = FM_Content_Rec_Interface(checkpoint_dir)
        elif model_name == 'NGCF':
            self.ngcf_data_generator = NGCF_Dataset(train_ngcf_behaviors_dir, train_ngcf_behaviors_path,
                                               val_ngcf_behaviors_path, test_ngcf_behaviors_path)
            self.test_behaviors = self.ngcf_data_generator
            self.model = NGCF_Rec_Interface(self.ngcf_data_generator, checkpoint_dir)
        else:
            
            print(f"{model_name} not included!")
            exit()
        self.model_acc = self.model.evaluate(self.test_behaviors)[1]
        print("init acc:", self.model_acc)

    def __load_behaviors(self):
        behaviors = pd.read_table(
            self.behaviors_path,
            header=None,
            usecols=[1, 3],
            names=['user', 'clicked_news'],
            index_col=0)

        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            test_behaviors = BaseDataset(self.test_behaviors_path, self.news_path)
        elif model_name == 'DFM':
            test_behaviors = FMDataset.init_from_path(self.test_fm_behaviors_path)
        elif model_name == 'NCF' or model_name == 'DFM_ID' :
            test_behaviors = FMDataset.init_from_path(self.test_fm_behaviors_path, id_only=True)
        elif model_name == 'DFM_CONTENT' :
            test_behaviors = FM_Content_Dataset.init_from_path(self.test_fm_behaviors_path, self.news2int_path, self.news2emb_path)
        elif model_name == 'DFM_CONTENT_ONLY' :
            test_behaviors = FM_Content_Dataset.init_from_path(self.test_fm_behaviors_path, self.news2int_path, self.news2emb_path,
                                                               feat_idx_list=MINDConfig.user_item_idx)
        elif model_name == 'NGCF':
            test_behaviors = None
        else:
            
            print(f"{model_name} not included!")
            exit()

        
        user2news_set = dict()
        for row in behaviors.itertuples():
            user = row.Index
            clicked_news_list = row.clicked_news.strip().split()
            user2news_set[user] = set(clicked_news_list)

        return behaviors, test_behaviors, user2news_set, behaviors.index.tolist()

    def __load_news(self):
        news = pd.read_table(self.news_path, header=0, index_col=0)

        return news, news.index.tolist()

    def __gen_fm_data(self):

        user2feat_id_list = {}
        news2feat_id_list = {}

        user2int = dict(pd.read_table(self.user2int_path).values.tolist())
        for user, id in user2int.items():
            user2feat_id_list[user] = [id]

        news2int = dict(pd.read_table(self.news2int_path).values.tolist())
        news_df = pd.read_table(self.news_path,
                                header=0,
                                usecols=[0, 1, 2])
        for row in news_df.itertuples():
            news = row.id
            cat = row.category
            subcat = row.subcategory

            if model_name == 'NCF' or model_name == 'DFM_ID'  or model_name == 'DFM_CONTENT_ONLY' :
                news2feat_id_list[news] = [news2int[news]]
            else:
                news2feat_id_list[news] = [news2int[news], cat, subcat]

        return user2int, news2int, user2feat_id_list, news2feat_id_list

    def __getRecList(self, click_prob, I, users, rec_num):
        if click_prob.is_cuda:
            click_prob = click_prob.cpu().numpy()
        else:
            click_prob = click_prob.numpy()

        idx = np.argsort(click_prob, axis=-1 )
        idx_arrays = idx[:, ::-1]

        news_index = [[row_I[i] for i in row_idx]
                      for row_idx, row_I in zip(idx_arrays, I)]

        rec_news_list = []
        rec_prob_list = []
        for i, user in enumerate(users):
            news_list_tmp = []
            prob_list_tmp = []

            for j, news_idx in enumerate(news_index[i]):
                news = self.news_list[news_idx]
                prob = click_prob[i][idx_arrays[i][j]]

                if news not in self.user2news_set[user]:
                    news_list_tmp.append(news)
                    prob_list_tmp.append(prob)
                    if news_list_tmp.__len__() >= rec_num:
                        break
            rec_news_list.append(news_list_tmp)
            rec_prob_list.append(prob_list_tmp)

        return rec_news_list, rec_prob_list

    def __simulateClick(self, rec_news_list, rec_prob_list):

        click_news_list = []

        for i in range(rec_news_list.__len__()):
            tmp = []
            for j, prob in enumerate(rec_prob_list[i]):
                click_prob = getRandomBeta(prob) 
                sim_click_prob = self.model_acc * click_prob 
                random_p = np.random.random()
                if random_p < sim_click_prob:
                    tmp.append(rec_news_list[i][j])
            click_news_list.append(tmp)

        return click_news_list

    def updateClickHistory(self, user2added_news):

        for user in self.behaviors.index:
            if user in user2added_news:
                added_news = user2added_news[user]
                if added_news.__len__() > 0:
                    added_news_str = ' '.join(added_news)
                    self.behaviors.loc[user, 'clicked_news'] = added_news_str + ' ' +\
                                                self.behaviors.loc[user, 'clicked_news']

        for user in self.user2news_set:
            if user in user2added_news:
                self.user2news_set[user] = self.user2news_set[user].union(
                    set(user2added_news[user]) )

    def __get_neg_news(self, user, neg_num):
        neg_news_list = []

        news_num = len(self.news_list)
        while len(neg_news_list) < neg_num:
            news = self.news_list[random.randint(0, news_num - 1)]
            if news not in neg_news_list and news not in self.user2news_set[user]:
                neg_news_list.append(news)

        return neg_news_list

    def __get_ngcf_train_data(self, user2added_news):

        added_users = []
        added_pos_items = []
        added_neg_items = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            for i in range(neg_num):
                added_users.append(self.user2int[user]-1)
                added_pos_items.append(self.news2int[added_news[i]]-1)
                added_neg_items.append(self.news2int[neg_news_list[i]]-1)

        return added_users, added_pos_items, added_neg_items

    def __get_fm_train_data(self, user2added_news):

        target = []
        feat_id_list = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            for i in range(neg_num):
                feat_id_list.append(self.user2feat_id_list[user] + self.news2feat_id_list[added_news[i]] )
                feat_id_list.append(self.user2feat_id_list[user] + self.news2feat_id_list[neg_news_list[i]])
                target.append(1)
                target.append(0)

        return feat_id_list, target

    def __get_content_rec_train_data(self, added_news_history, user2added_news):
        data = []

        for user, added_news in user2added_news.items():
            neg_num = len(added_news)
            neg_news_list = self.__get_neg_news(user, neg_num)

            user_id = self.user2int[user]
            origin_clicked_news = self.origin_behaviors.loc[user, 'clicked_news']
            added_news_list = []
            for record in added_news_history[::-1]:
                if user in record:
                    added_news_list.extend(record[user])
            clicked_news = origin_clicked_news
            if added_news_list.__len__() > 0:
                clicked_news =  ' '.join(added_news_list) + ' ' + clicked_news

            for i in range(neg_num):
                candidate_news = [added_news[i], neg_news_list[i]]
                clicked = ['1', '0']
                data.append([user_id, clicked_news, ' '.join(candidate_news), ' '.join(clicked)])

        behaviors_parsed = pd.DataFrame(data, columns=['user', 'clicked_news', 'candidate_news', 'clicked'])

        return behaviors_parsed

    def rec(self, recall_num=500, rec_num=50, rec_round=6, update_interval=1, sample=True):
        batch_size = BaseConfig.batch_size
        user_num = len(self.user_list)
        news_num = len(self.news_list)
        print("user num:", user_num)
        print("news num:", news_num)

        if not os.path.exists(f"res/{dataset_name}"):
            os.makedirs(f"res/{dataset_name}")
        if not os.path.exists(f"res/{dataset_name}/{model_name}"):
            os.makedirs(f"res/{dataset_name}/{model_name}")

        added_news_records = []
        complete_news_records = []

        for round_i in range(rec_round):
            print("rec round:", round_i + 1)

            
            news2emb_save_path = f"res/{dataset_name}/{model_name}/news2emb_{round_i+1}.pkl"
            if os.path.exists(news2emb_save_path):
                with open(news2emb_save_path, 'rb') as embf:
                    news2emb_arr = pkl.load(embf)
                news2emb = {}
                for news, emb_arr in news2emb_arr.items():
                    news2emb[news] = torch.from_numpy(emb_arr).to(device)
            else:
                if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
                    news2emb = self.model.getNews2Emb()
                elif model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_ID' :
                    news2emb = self.model.getNews2Emb(self.news2feat_id_list)
                elif model_name == 'DFM_CONTENT' or model_name == 'DFM_CONTENT_ONLY' :
                    news2emb = self.model.getNews2Emb(self.news2feat_id_list, self.news_int2emb)
                elif model_name == 'NGCF':
                    news2emb = self.model.getNews2Emb(self.news2int)
                else:
                    
                    print(f"{model_name} not included!")
                    exit()

                news2emb_arr = {}
                for news, emb in news2emb.items():
                    if emb.is_cuda:
                        news2emb_arr[news] = emb.cpu().numpy()
                    else:
                        news2emb_arr[news] = emb.numpy()

                with open(news2emb_save_path, 'wb') as outf:
                    pkl.dump(news2emb_arr, outf)
                print("save news2emb done.")

            news_tensors = torch.stack([news2emb[news] for news in self.news_list], dim=0)
            if news_tensors.is_cuda:
                news_arrays = news_tensors.cpu().numpy()
            else:
                news_arrays = news_tensors.numpy()
            news_arrays = news_arrays.astype('float32')

            
            if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
                user2emb = self.model.getUser2Emb(news2emb)
            elif model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_ID' :
                user2emb = self.model.getUser2Emb(self.user2feat_id_list)
            elif model_name == 'DFM_CONTENT' or model_name == 'DFM_CONTENT_ONLY' :
                user2emb = self.model.getUser2Emb(self.user2feat_id_list)
            elif model_name == 'NGCF':
                user2emb = self.model.getUser2Emb(self.user2int)
            else:
                
                print(f"{model_name} not included!")
                exit()

            user2added_news = dict()

            for i in range(math.ceil(user_num/batch_size)):
                users = self.user_list[i*batch_size: (i+1)*batch_size]

                user_tensors = torch.stack([user2emb[u] for u in users], dim=0)
                if model_name != 'DKN':
                    if user_tensors.is_cuda:
                        user_arrays = user_tensors.cpu().numpy()
                    else:
                        user_arrays = user_tensors.numpy()
                else:
                    mean_user_tensors = torch.stack([torch.mean(user2emb[u], dim=0) for u in users], dim=0)
                    if mean_user_tensors.is_cuda:
                        user_arrays = mean_user_tensors.cpu().numpy()
                    else:
                        user_arrays = mean_user_tensors.numpy()
                user_arrays = user_arrays.astype('float32')

                D, I = getTopKSim(user_arrays, news_arrays, recall_num)

                
                if model_name == 'NRMS' or model_name == 'NGCF' or model_name == 'NAML' or model_name == 'DKN':

                    
                    recall_news_tensors = torch.stack([ torch.stack([news_tensors[i] for i in row], dim=0)
                                                        for row in I], dim=0)
                    
                    click_prob = self.model.getChickProb(recall_news_tensors, user_tensors)

                elif model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_CONTENT' or \
                        model_name == 'DFM_ID' or model_name == 'DFM_CONTENT_ONLY':
                    user_feat = [] 
                    for user in users:
                        user_feat.append(self.user2feat_id_list[user])

                    recall_news_feat = [] 
                    for row in I:
                        tmp = []
                        for news_id in row:
                            tmp.append( self.news2feat_id_list[self.news_list[news_id]])
                        recall_news_feat.append(tmp)

                    if model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_ID' :
                        
                        click_prob = self.model.getChickProb(user_feat, recall_news_feat)
                    else:
                        click_prob = self.model.getChickProb(user_feat, recall_news_feat, self.news_int2emb)

                else:
                    
                    print(f"{model_name} not included!")
                    exit()

                
                rec_news_list, rec_prob_list = self.__getRecList(click_prob, I, users, rec_num)

                click_news_list = self.__simulateClick(rec_news_list, rec_prob_list)

                
                user2added_news.update(dict(zip(users, click_news_list)))
            print("done")
            
            self.updateClickHistory(user2added_news)
            added_news_records.append(user2added_news)
            complete_news_records.append(user2added_news)

            if (round_i+1) % update_interval == 0:  
                if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
                    for i, record in enumerate(added_news_records):
                        self.model.updateUserDataset(record)
                        new_behaviors = self.__get_content_rec_train_data(complete_news_records[:(-1)*update_interval + i], record)
                        self.model.retrain(new_behaviors)
                elif model_name == 'DFM' or model_name == 'NCF' or model_name == 'DFM_ID' :
                    for record in added_news_records:
                        feat_id_list, target = self.__get_fm_train_data(record)
                        self.model.retrain(feat_id_list, target)
                elif model_name == 'DFM_CONTENT' or model_name == 'DFM_CONTENT_ONLY' :
                    for record in added_news_records:
                        feat_id_list, target = self.__get_fm_train_data(record)
                        self.model.retrain(feat_id_list, target, self.news_int2emb)
                elif model_name == 'NGCF':
                    for record in added_news_records:
                        added_data = self.__get_ngcf_train_data(record)
                        self.model.retrain(added_data)

                
                auc, acc, macro_f1, micro_f1 = self.model.evaluate(self.test_behaviors)
                print(f"rec round {round_i + 1} res:", auc, acc, macro_f1, micro_f1)
                self.model_acc = acc

                added_news_records = []

            
            if sample:
                save_path = f"res/{dataset_name}/{model_name}/behaviors_sample_recall_{recall_num}_rec_{rec_num}_interval_{update_interval}_{round_i+1}.tsv"
            else:
                save_path = f"res/{dataset_name}/{model_name}/behaviors_recall_{recall_num}_rec_{rec_num}_interval_{update_interval}_{round_i+1}.tsv"
            self.behaviors.to_csv(save_path, sep="\t", header=False, index=True)

def main():
    sample = False
    if sample:
        behaviors_path = f"data/{dataset_name}/test/behaviors_sample.tsv"
    else:
        behaviors_path = f"data/{dataset_name}/test/behaviors_merge_dedup.tsv"

    test_behaviors_path = f"data/{dataset_name}/test/behaviors_merge_dedup_parsed.tsv"
    test_fm_behaviors_path = f"data/{dataset_name}/test/fm_behaviors.tsv"
    test_ngcf_behaviors_path = f"data/{dataset_name}/test/ngcf_behaviors.txt"

    train_ngcf_behaviors_dir = f"data/{dataset_name}/train"
    train_ngcf_behaviors_path = f"data/{dataset_name}/train/ngcf_behaviors.txt"
    val_ngcf_behaviors_path = f"data/{dataset_name}/val/ngcf_behaviors.txt"

    news_path = f"data/{dataset_name}/train/news_parsed.tsv"
    news2int_path = f"data/{dataset_name}/train/news2int.tsv"
    user2int_path = f"data/{dataset_name}/train/user2int.tsv"

    checkpoint_dir = f"checkpoint/{dataset_name}/{model_name}"

    news2emb_path = f"res/{dataset_name}/{emb_model}/news2emb.pkl"
    if model_name[:12] == 'DFM_CONTENT' :
        if not os.path.exists(news2emb_path):
            print("news embedding file does not exist!")
            exit()

    news_recommender = NewsRec(news_path, behaviors_path, test_behaviors_path, test_fm_behaviors_path,
                               train_ngcf_behaviors_dir, train_ngcf_behaviors_path, val_ngcf_behaviors_path, test_ngcf_behaviors_path,
                               user2int_path, news2int_path, news2emb_path, checkpoint_dir)
    news_recommender.rec(sample=sample)

if __name__ == '__main__':
    main()