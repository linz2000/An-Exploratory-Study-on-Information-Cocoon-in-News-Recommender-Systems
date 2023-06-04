
from config import model_name, dataset_name, device
from dataset import FM_Content_Dataset
from config import FMConfig, MINDConfig
from config import emb_model

import torch
from torch.utils.data import DataLoader
import importlib
from pathlib import Path
import os
import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

def get_model(field_dims):
    model_name_tmp = model_name
    if model_name_tmp.endswith('_ONLY'):
        model_name_tmp = model_name_tmp[:-5]

    Model = getattr(importlib.import_module(f"model.FM.{model_name_tmp.lower()}"), model_name_tmp)
    if model_name_tmp == 'DFM_CONTENT':
        return Model(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2, conti_dim=MINDConfig.news_emb_dim)
    else:
        raise ValueError('unknown model name: ' + model_name)

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, conti_x, target) in enumerate(tk0):
        fields, conti_x, target = fields.to(device), conti_x.to(device), target.to(device)
        y = model(fields, conti_x)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, conti_x, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, conti_x, target = fields.to(device), conti_x.to(device), target.to(device)
            y = model(fields, conti_x)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            
    return roc_auc_score(targets, predicts)

def test_fm_content_model(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, conti_x, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, conti_x, target = fields.to(device), conti_x.to(device), target.to(device)
            y = model(fields, conti_x)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    predicts_l = [1 if s>0.5 else 0 for s in predicts]

    auc = roc_auc_score(targets, predicts)
    acc = accuracy_score(targets, predicts_l)
    macro_f1 = f1_score(targets, predicts_l, average='macro')
    micro_f1 = f1_score(targets, predicts_l, average='micro')

    return auc, acc, macro_f1, micro_f1


def main():

    train_behaviors_path = f"data/{dataset_name}/train/fm_behaviors.tsv"
    val_behaviors_path = f"data/{dataset_name}/val/fm_behaviors.tsv"
    test_behaviors_path = f"data/{dataset_name}/test/fm_behaviors.tsv"
    model_savepath = f"checkpoint/{dataset_name}/{model_name}/ckpt.pth"

    news2int_path = f"data/{dataset_name}/train/news2int.tsv"
    news2emb_path = f"res/{dataset_name}/{emb_model}/news2emb.pkl"

    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists(f'checkpoint/{dataset_name}'):
        os.makedirs(f'checkpoint/{dataset_name}')
    checkpoint_dir = os.path.join(f'./checkpoint/{dataset_name}', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if model_name == 'DFM_CONTENT':
        train_dataset = FM_Content_Dataset.init_from_path(train_behaviors_path, news2int_path, news2emb_path)
        val_dataset = FM_Content_Dataset.init_from_path(val_behaviors_path, news2int_path, news2emb_path)
        test_dataset = FM_Content_Dataset.init_from_path(test_behaviors_path, news2int_path, news2emb_path)

        model = get_model(np.array(MINDConfig.field_dims)).to(device)
    elif model_name == 'DFM_CONTENT_ONLY':
        train_dataset = FM_Content_Dataset.init_from_path(train_behaviors_path, news2int_path, news2emb_path, feat_idx_list=MINDConfig.user_item_idx)
        val_dataset = FM_Content_Dataset.init_from_path(val_behaviors_path, news2int_path, news2emb_path, feat_idx_list=MINDConfig.user_item_idx)
        test_dataset = FM_Content_Dataset.init_from_path(test_behaviors_path, news2int_path, news2emb_path, feat_idx_list=MINDConfig.user_item_idx)

        model = get_model(np.array(MINDConfig.id_field_dims)).to(device)
    else:
        raise ValueError('unknown model name: ' + model_name)

    train_data_loader = DataLoader(train_dataset, batch_size=FMConfig.batch_size, num_workers=8)
    val_data_loader = DataLoader(val_dataset, batch_size=FMConfig.batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=FMConfig.batch_size, num_workers=8)

    if os.path.exists(model_savepath):
        model.load_state_dict(torch.load(model_savepath, map_location=device))
        model.eval()
    else:
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=FMConfig.learning_rate, weight_decay=FMConfig.weight_decay)
        early_stopper = EarlyStopper(num_trials=5, save_path=model_savepath)
        for epoch_i in range(FMConfig.epoch):
            train(model, optimizer, train_data_loader, criterion, device)
            auc = test(model, val_data_loader, device)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            if not early_stopper.is_continuable(model, auc):
                print(f'validation: best auc: {early_stopper.best_accuracy}')
                break

    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == '__main__':
    main()