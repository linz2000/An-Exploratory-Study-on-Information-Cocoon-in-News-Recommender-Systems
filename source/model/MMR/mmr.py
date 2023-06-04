import torch

def mmr(click_prob_list, similarity_matrix, rec_num, lambda_constant=0.5):
    candidate_num = len(click_prob_list)
    sel_idx_list, candidate_idx_set = [], set([i for i in range(candidate_num)])

    while len(candidate_idx_set) > 0:
        highest_score = None
        new_select_idx = None

        for idx in candidate_idx_set:
            part1 = click_prob_list[idx]

            part2 = None
            for sel_idx in sel_idx_list:
                sim_score = similarity_matrix[idx][sel_idx]
                if part2 == None or sim_score > part2:
                    part2 = sim_score
            if part2 == None: part2=0

            score = lambda_constant * (part1 - (1-lambda_constant)*part2)
            if highest_score == None or score > highest_score:
                highest_score = score
                new_select_idx = idx

        candidate_idx_set.remove(new_select_idx)
        sel_idx_list.append(new_select_idx)

        if len(sel_idx_list) >= rec_num:
            break

    return sel_idx_list

def mmr_filter(clicked_news_list: list, candidate_news_list: list, news2vector: dict,
               user_radius, click_prob_list, similarity_matrix, rec_num, lambda_constant=0.5):

    clicked_news_num = len(clicked_news_list)
    news_vector_puser = torch.stack([news2vector[i] for i in clicked_news_list], dim=0)
    candidate_num = len(click_prob_list)
    sel_idx_list, candidate_idx_set = [], set([i for i in range(candidate_num)])

    while len(candidate_idx_set) > 0:
        highest_score = None
        new_select_idx = None

        for idx in candidate_idx_set:
            part1 = click_prob_list[idx]

            part2 = None
            for sel_idx in sel_idx_list:
                sim_score = similarity_matrix[idx][sel_idx]
                if part2 == None or sim_score > part2:
                    part2 = sim_score
            if part2 == None: part2 = 0

            score = lambda_constant * (part1 - (1 - lambda_constant) * part2)
            if highest_score == None or score > highest_score:
                highest_score = score
                new_select_idx = idx

        candidate_idx_set.remove(new_select_idx)

        news_vector = news2vector[candidate_news_list[new_select_idx]]
        add_dis_sum = torch.sum(
            1 - torch.cosine_similarity(news_vector.unsqueeze(dim=0), news_vector_puser, dim=-1)).item()
        if add_dis_sum > user_radius * clicked_news_num:
            sel_idx_list.append(new_select_idx)

            if len(sel_idx_list) >= rec_num:
                break

    return sel_idx_list