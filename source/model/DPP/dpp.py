import numpy as np
import math
import torch

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items

def dpp_filter(clicked_news_list: list, candidate_news_list: list, news2vector: dict, user_radius, kernel_matrix, max_length, epsilon=1E-10):
    clicked_news_num = len(clicked_news_list)
    candidate_news_num = len(candidate_news_list)
    news_vector_puser = torch.stack([news2vector[i] for i in clicked_news_list], dim=0)
    flag_list = []

    item_size = kernel_matrix.shape[0]
    cis = np.zeros((candidate_news_num, item_size)) # (max_length, item_size)
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    selected_items.append(selected_item)

    news_vector = news2vector[candidate_news_list[selected_item]]
    add_dis_sum = torch.sum(1 - torch.cosine_similarity(news_vector.unsqueeze(dim=0), news_vector_puser, dim=-1)).item()
    if add_dis_sum > user_radius * clicked_news_num:
        flag_list.append(True)
    else:
        flag_list.append(False)
        max_length += 1

    while len(selected_items) < candidate_news_num and len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        di2s[selected_item] = -np.inf
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

        news_vector = news2vector[candidate_news_list[selected_item]]
        add_dis_sum = torch.sum(1 - torch.cosine_similarity(news_vector.unsqueeze(dim=0), news_vector_puser, dim=-1)).item()
        if add_dis_sum > user_radius * clicked_news_num:
            flag_list.append(True)
        else:
            flag_list.append(False)
            max_length += 1

    true_selected_items = [idx for flag, idx in zip(flag_list, selected_items) if flag]

    return true_selected_items