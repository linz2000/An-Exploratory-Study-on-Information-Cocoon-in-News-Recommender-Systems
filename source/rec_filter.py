from config import model_name, dataset_name
from model_interface import Content_Rec_Interface, FM_Rec_Interface
from model.DPP.dpp import dpp, dpp_filter
from model.MMR.mmr import mmr, mmr_filter

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
import time
import csv
import os
import random

def entropy(px):
    return np.sum(- (px * np.log2(px)))

def gini(px):
    return 1- np.sum(px**2)

def getListGini(type_list: list):
    type2freq = dict()
    for t in type_list:
        if t not in type2freq:
            type2freq[t] = 1
        else:
            type2freq[t] += 1

    sum = type_list.__len__()
    freq_list = np.array([num/sum for num in type2freq.values()] )

    return gini(freq_list)

def getListNormalizedEntory(type_list: list):
    type2freq = dict()
    for t in type_list:
        if t not in type2freq:
            type2freq[t] = 1
        else:
            type2freq[t] += 1

    sum = type_list.__len__()
    freq_list = np.array([num/sum for num in type2freq.values()] )
    type_num = len(type2freq)

    if type_num <= 1:
        return entropy(freq_list)

    return entropy(freq_list) / np.log2(type_num)

def calcuDis(user_news_list: list, news2vector):  

    n = len(user_news_list)
    if n == 0:    
        return 0, 0, 0
    elif n == 1:
        return 0, 0, 1
    news_vector_puser = torch.stack([news2vector[i] for i in user_news_list], dim=0)
    dis_sum = torch.sum(1 - torch.cosine_similarity(news_vector_puser.unsqueeze(dim=1),
                            news_vector_puser.unsqueeze(dim=0), dim=-1)).item()
    dis_avg = dis_sum / (n* (n-1) )

    return dis_avg, dis_sum, n 

def calcuAvgDis(user2distance: dict, multi_value=False):  
    num = user2distance.__len__()
    if num == 0:
        return 0
    sum = 0.
    for d in user2distance.values():
        if multi_value:
            sum += d[0]
        else:
            sum += d
    return sum / num

def calculate_single_user_metric(tasks):
    y_true = tasks[0]
    y_score = tasks[1]

    try:
        auc = roc_auc_score(y_true, y_score)
        ndcg = ndcg_score([y_true], [y_score])

        return [auc, ndcg]

    except Exception as e:
        print(e)

        return [np.nan] * 2

class Recommender():
    def __init__(self, news_path, behaviors_path, user2int_path, news2int_path, checkpoint_dir):
        self.behaviors_path = behaviors_path
        self.news_path = news_path
        self.user2int_path = user2int_path
        self.news2int_path = news2int_path

        self.behaviors, self.news2cat, self.user2feat_id_list, self.news2feat_id_list, self.news_list = self.__load_data()

        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            self.model = Content_Rec_Interface(news_path, behaviors_path, user2int_path, checkpoint_dir)
            self.news2emb = self.model.getNews2Emb()
            self.user2emb = self.model.getUser2Emb(self.news2emb)
        elif model_name == 'DFM':
            self.model = FM_Rec_Interface(checkpoint_dir)
            self.news2emb = self.model.getNews2Emb(self.news2feat_id_list)
            self.user2emb = self.model.getUser2Emb(self.user2feat_id_list)

        else:
            
            print(f"{model_name} not included!")
            exit()

        self.user2radius_tuple = self.get_user_initial_radius()

    def __load_data(self):
        behaviors = pd.read_table(self.behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=[
                                           'impression_id', 'user', 'time',
                                           'clicked_news', 'impressions'
                                       ])

        news_df = pd.read_table(self.news_path,
                                header=0,
                                usecols=[0, 1, 2])

        news2cat = {}
        user2feat_id_list = {}
        news2feat_id_list = {}

        user2int = dict(pd.read_table(self.user2int_path).values.tolist())
        for user, id in user2int.items():
            user2feat_id_list[user] = [id]

        news2int = dict(pd.read_table(self.news2int_path).values.tolist())

        for row in news_df.itertuples():
            news = row.id
            cat = row.category
            subcat = row.subcategory
            news2cat[news] = cat

            news2feat_id_list[news] = [news2int[news], cat, subcat]

        news_list = list(news2cat.keys())

        return behaviors, news2cat, user2feat_id_list, news2feat_id_list, news_list


    def get_user_initial_radius(self):
        user2radius_tuple = {}
        for row in self.behaviors.itertuples():
            user = row.user
            clicked_news_string = row.clicked_news
            clicked_news_list = clicked_news_string.strip().split()

            dis_avg, dis_sum, p = calcuDis(clicked_news_list, self.news2emb)
            user2radius_tuple[user] = (dis_avg, dis_sum, p)

        return user2radius_tuple

    def get_new_user_radius(self, user, clicked_news_list, added_news_list):
        dis_avg, dis_sum, p = self.user2radius_tuple[user]
        q = len(added_news_list)

        if q ==0:
            return dis_avg, dis_sum, p

        news_vector_puser = torch.stack([self.news2emb[i] for i in clicked_news_list], dim=0)
        added_news_vector = torch.stack([self.news2emb[i] for i in added_news_list], dim=0)

        part1 = 2 * torch.sum(1 - torch.cosine_similarity(added_news_vector.unsqueeze(1), news_vector_puser, dim=-1)).item()
        _, part2, _ = calcuDis(added_news_list, self.news2emb)
        new_dis_sum = dis_sum + part1 + part2
        new_radius = new_dis_sum / ((p+q)*(p+q-1))

        return new_radius, new_dis_sum, p+q


    def get_rec_idx(self, user:str, candidate_news_list, rec_num):

        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()
        order = np.argsort(click_prob_list)[::-1]  
        rec_idx_list = order[:rec_num]

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def get_rec_idx_filter(self, user: str, clicked_news_list, candidate_news_list, rec_num, padding=True):

        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()
        order = np.argsort(click_prob_list)[::-1]  

        news_vector_puser = torch.stack([self.news2emb[i] for i in clicked_news_list], dim=0)
        dis_avg, dis_sum, p = self.user2radius_tuple[user]

        rec_idx_list = []
        for idx in order:
            news_vector = self.news2emb[candidate_news_list[idx]]
            add_dis_sum = torch.sum(1 - torch.cosine_similarity(news_vector.unsqueeze(dim=0),
                                                                news_vector_puser, dim=-1)).item()
            if add_dis_sum > dis_avg * p:
                rec_idx_list.append(idx)
                if len(rec_idx_list) >= rec_num:
                    break

        if padding and len(rec_idx_list) < rec_num: 
            rec_idx_set = set(rec_idx_list)
            for idx in order[:rec_num]:
                if idx not in rec_idx_set:
                    rec_idx_list.append(idx)
                    if len(rec_idx_list) >= rec_num:
                        break
        if len(rec_idx_list) != rec_num:
            print("rec num:", len(rec_idx_list))

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def get_rec_idx_mrr(self, user: str, candidate_news_list, rec_num):

        candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()

        if candidate_news_vector.is_cuda:
            np_feats = candidate_news_vector.cpu().numpy()
        else:
            np_feats = candidate_news_vector.numpy()
        np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
        similarities = np.dot(np_feats, np_feats.T)
        rec_idx_list = mmr(click_prob_list, similarities, rec_num, lambda_constant=0.5)

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def get_rec_idx_mrr_filter(self, user: str, clicked_news_list, candidate_news_list, rec_num, padding=True):

        candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()

        if candidate_news_vector.is_cuda:
            np_feats = candidate_news_vector.cpu().numpy()
        else:
            np_feats = candidate_news_vector.numpy()
        np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
        similarities = np.dot(np_feats, np_feats.T)
        user_radius, _, _ = self.user2radius_tuple[user]
        rec_idx_list = mmr_filter(clicked_news_list, candidate_news_list, self.news2emb, user_radius, click_prob_list,
                                  similarities, rec_num, lambda_constant=0.5)

        if padding and len(rec_idx_list) < rec_num: 
            order = np.argsort(click_prob_list)[::-1]  
            rec_idx_set = set(rec_idx_list)
            for idx in order[:rec_num]:
                if idx not in rec_idx_set:
                    rec_idx_list.append(idx)
                    if len(rec_idx_list) >= rec_num:
                        break
        if len(rec_idx_list) != rec_num:
            print("rec num:", len(rec_idx_list))

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def get_rec_idx_dpp(self, user: str, candidate_news_list, rec_num):

        candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()

        np_scores = np.array(click_prob_list)
        if candidate_news_vector.is_cuda:
            np_feats = candidate_news_vector.cpu().numpy()
        else:
            np_feats = candidate_news_vector.numpy()
        np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
        similarities = np.dot(np_feats, np_feats.T)
        candidate_num = len(candidate_news_list)
        kernel_matrix = np_scores.reshape((candidate_num, 1)) * similarities * np_scores.reshape((1, candidate_num))
        rec_idx_list = dpp(kernel_matrix, rec_num)

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def get_rec_idx_dpp_filter(self, user: str, clicked_news_list, candidate_news_list, rec_num, padding=True):

        candidate_news_vector = torch.stack([self.news2emb[i] for i in candidate_news_list], dim=0)
        if model_name == 'NRMS' or model_name == 'NAML' or model_name == 'DKN':
            user_vector = self.user2emb[user]
            click_prob = self.model.getChickProb(candidate_news_vector.unsqueeze(0), user_vector.unsqueeze(0)).squeeze(0)
        else:
            user_feat = [self.user2feat_id_list[user]]
            news_feat, tmp = [], []
            for news in candidate_news_list:
                tmp.append(self.news2feat_id_list[news])
            news_feat.append(tmp)
            click_prob = self.model.getChickProb(user_feat, news_feat).squeeze(0)

        click_prob_list = click_prob.tolist()

        np_scores = np.array(click_prob_list)
        if candidate_news_vector.is_cuda:
            np_feats = candidate_news_vector.cpu().numpy()
        else:
            np_feats = candidate_news_vector.numpy()
        np_feats /= np.linalg.norm(np_feats, axis=1, keepdims=True)
        similarities = np.dot(np_feats, np_feats.T)
        candidate_num = len(candidate_news_list)
        kernel_matrix = np_scores.reshape((candidate_num, 1)) * similarities * np_scores.reshape((1, candidate_num))
        user_radius, _, _ = self.user2radius_tuple[user]
        rec_idx_list = dpp_filter(clicked_news_list, candidate_news_list, self.news2emb, user_radius, kernel_matrix, rec_num)

        if padding and len(rec_idx_list) < rec_num: 
            order = np.argsort(click_prob_list)[::-1]  
            rec_idx_set = set(rec_idx_list)
            for idx in order[:rec_num]:
                if idx not in rec_idx_set:
                    rec_idx_list.append(idx)
                    if len(rec_idx_list) >= rec_num:
                        break
        if len(rec_idx_list) != 10:
            print("rec num:", len(rec_idx_list))

        rec_y_pred = [click_prob_list[i] for i in rec_idx_list]

        return rec_idx_list, rec_y_pred

    def test_time(self, candidate_news_num, rec_num=10, min_clicked_news=10, mode='none'):
        """
        mode:   none(no filter), filter, dpp, dpp_filter, mrr, mrr_filter
        """
        sum, t_sum = 0, 0.
        for row in self.behaviors.itertuples():
            t_start = time.time()
            user = row.user
            clicked_news_string = row.clicked_news
            clicked_news_list = clicked_news_string.strip().split()

            if len(clicked_news_list) < min_clicked_news:
                continue

            sum += 1
            candidate_news_list = random.sample(self.news_list, candidate_news_num) 

            if mode == 'none':
                rec_idx_list, rec_y_pred = self.get_rec_idx(user, candidate_news_list, rec_num)
            elif mode == 'filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            elif mode == 'dpp':
                rec_idx_list, rec_y_pred = self.get_rec_idx_dpp(user, candidate_news_list, rec_num)
            elif mode == 'dpp_filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_dpp_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            elif mode == 'mrr':
                rec_idx_list, rec_y_pred = self.get_rec_idx_mrr(user, candidate_news_list, rec_num)
            elif mode == 'mrr_filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_mrr_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            else:
                print(f"{mode} not included!")
                exit()

            added_news_list = [candidate_news_list[i] for i in rec_idx_list] 

            t_end = time.time() 
            t_sum += t_end - t_start

        avg_t = t_sum / sum

        save_dir = "res/rec_filter/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"test_time_rec_num_{rec_num}_min_clicked_news_{min_clicked_news}_2.csv")
        flag = os.path.exists(save_path)

        with open(save_path, 'a', newline="") as outf:
            writer = csv.writer(outf)
            if not flag:
                writer.writerow(['model name', 'mode', 'candidate news num', 'avg. running time (s)'])
            writer.writerow([model_name, mode, str(candidate_news_num), str(avg_t)])


    def rec(self, rec_num=10, min_clicked_news=10, min_impressions=100, mode='none'):
        """
        mode:   none(no filter), filter, dpp, dpp_filter, mrr, mrr_filter
        """
        tasks = [[], []]
        sum, count, t_sum = 0, 0, 0.
        new_user2radius_tuple = {}
        user2cat_gini = {}
        user2normalized_cat_entropy = {}

        for row in self.behaviors.itertuples():
            t_start = time.time()
            user = row.user
            clicked_news_string = row.clicked_news
            clicked_news_list = clicked_news_string.strip().split()
            impressions_string = row.impressions
            impressions_list = impressions_string.strip().split()

            if len(clicked_news_list) < min_clicked_news or len(impressions_list) < min_impressions:
                continue

            sum += 1
            candidate_news_list = [imp.split('-')[0] for imp in impressions_list]
            candidate_labels = [int(imp.split('-')[1]) for imp in impressions_list]

            if mode == 'none':
                rec_idx_list, rec_y_pred = self.get_rec_idx(user, candidate_news_list, rec_num)
            elif mode == 'filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            elif mode == 'dpp':
                rec_idx_list, rec_y_pred = self.get_rec_idx_dpp(user, candidate_news_list, rec_num)
            elif mode == 'dpp_filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_dpp_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            elif mode == 'mrr':
                rec_idx_list, rec_y_pred = self.get_rec_idx_mrr(user, candidate_news_list, rec_num)
            elif mode == 'mrr_filter':
                rec_idx_list, rec_y_pred = self.get_rec_idx_mrr_filter(user, clicked_news_list, candidate_news_list, rec_num, padding=True)
            else:
                print(f"{mode} not included!")
                exit()

            rec_y_true = [candidate_labels[i] for i in rec_idx_list]

            tasks[0].extend(rec_y_true)
            tasks[1].extend(rec_y_pred)

            
            added_news_list = [candidate_news_list[i] for i in rec_idx_list] 

            t_end = time.time() 
            t_sum += t_end - t_start

            clicked_news_cat_list = [self.news2cat[news] for news in clicked_news_list]
            added_news_cat_list = [self.news2cat[news] for news in added_news_list]
            news_cat_list = clicked_news_cat_list + added_news_cat_list
            user2cat_gini[user] = getListGini(news_cat_list)
            user2normalized_cat_entropy[user] = getListNormalizedEntory(news_cat_list)

            dis_avg, dis_sum, p = self.get_new_user_radius(user, clicked_news_list, added_news_list)
            new_user2radius_tuple[user] = (dis_avg, dis_sum, p)

            if dis_avg < self.user2radius_tuple[user][0]:
                count += 1

        auc, ndcg = calculate_single_user_metric(tasks)
        cat_gini = calcuAvgDis(user2cat_gini, multi_value=False)
        normalized_cat_entropy = calcuAvgDis(user2normalized_cat_entropy, multi_value=False)
        initial_radius = calcuAvgDis(self.user2radius_tuple, multi_value=True)
        new_radius = calcuAvgDis(new_user2radius_tuple, multi_value=True)
        des_rate = count/sum
        avg_t = t_sum/sum
        print(f"auc:", auc)
        print(f"ndcg:", ndcg)
        print("cat gini:", cat_gini)
        print("normalized cat entropy:", normalized_cat_entropy)
        print("initial radius:", initial_radius)
        print("new radius:", new_radius)
        print("des rate:", des_rate)
        print("avg. running time (s):", avg_t)
        print("user num:", sum)

        save_dir = "res/rec_filter/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"rec_num_{rec_num}_min_clicked_news_{min_clicked_news}_min_impressions_{min_impressions}_new.csv")
        flag = os.path.exists(save_path)

        with open(save_path, 'a', newline="") as outf:
            writer = csv.writer(outf)
            if not flag:
                writer.writerow(['model name', 'mode', 'auc', 'ndcg', 'cat gini', 'normalized cat entropy', 'initial radius', 'new radius', 'des rate', 'avg. running time (s)', 'user num'])
            writer.writerow([model_name, mode, str(auc), str(ndcg), str(cat_gini), str(normalized_cat_entropy), str(initial_radius), str(new_radius), str(des_rate), str(avg_t), str(sum)])



def main():
    behaviors_path = f"data/{dataset_name}/test/behaviors_merge_dedup.tsv"
    news_path = f"data/{dataset_name}/train/news_parsed.tsv"
    user2int_path = f"data/{dataset_name}/train/user2int.tsv"
    news2int_path = f"data/{dataset_name}/train/news2int.tsv"
    checkpoint_dir = f"checkpoint/{dataset_name}/{model_name}"

    news_recommender = Recommender(news_path, behaviors_path, user2int_path, news2int_path, checkpoint_dir)

    mode_list = ['none', 'filter', 'dpp', 'dpp_filter', 'mrr', 'mrr_filter']

    
    for mode in mode_list:
        print("mode", mode)
        news_recommender.rec(rec_num=10, mode=mode)

    # # test time
    # candidate_news_num_list = [100, 200, 300, 400, 500]
    # for candidate_news_num in candidate_news_num_list:
    #     for mode in mode_list:
    #         print(f"candidate news_num: {candidate_news_num}   mode: {mode}")
    #         news_recommender.test_time(candidate_news_num, mode=mode) # default: rec_num=10

if __name__ == '__main__':
    main()