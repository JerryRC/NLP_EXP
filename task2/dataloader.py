"""This module defines all the downstream loader."""

import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn
import multiprocessing


R2IDFILE = "semeval/semeval_rel2id.json"

class RelationExtractLoader(Dataset):

    def __init__(self, tokenizer, file):
        
        with open(file, 'r', encoding='utf-8') as json_file:
            # json obj format: 
            # {"token": ["the", "original", "play", "was", "filled", "with", "very", "topical", "humor", ",", "so", "the", "director", "felt", "free", "to", "add", "current", "topical", "humor", "to", "the", "script", "."], 
            # "h": {"name": "play", "pos": [2, 3]}, 
            # "t": {"name": "humor", "pos": [8, 9]}, 
            # "relation": "Component-Whole(e2,e1)"}
            self.data = [json.loads(line) for line in json_file.readlines()]
        
        with open(R2IDFILE, 'r', encoding='utf-8') as json_file:
            self.rel2id = json.load(json_file)
        # self.id2rel = {v: k for k, v in self.rel2id.items()}
        
        self.max_len = 128
        self.classes = len(self.rel2id)
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.e1s = '[E1]'
        self.e1e = '[/E1]'
        self.e2s = '[E2]'
        self.e2e = '[/E2]'
        tokenizer.add_tokens([self.e2s, self.e2e, self.e1s, self.e1e])
        

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        obj = self.data[index]
        # 取出label
        y = obj['relation']
        y = self.rel2id[y]
        # 取出token
        X = obj['token']
        # 找到e1和e2的位置
        e1_pos = obj['h']['pos']
        e2_pos = obj['t']['pos']

        X = X[:e1_pos[0]] + [self.e1s] + X[e1_pos[0]:e1_pos[1]] + [self.e1e] + X[e1_pos[1]:e2_pos[0]] + [self.e2s] + X[e2_pos[0]:e2_pos[1]] + [self.e2e] + X[e2_pos[1]:]

        encode_x = self.tokenizer.encode(X, add_special_tokens=True, return_tensors="pt").squeeze(0)[:self.max_len]
        if encode_x.size(0) < self.max_len:
            pad_x = torch.tensor([self.pad_id]*(self.max_len-encode_x.size(0)))
            encode_x = torch.cat((encode_x, pad_x), dim=-1)

        y = torch.tensor([y]).long()
        return {'sent': encode_x, 'label': y}


def downstream_collate_fn(data):
    batch_data = {'sent': [], 'label': []}
    for data_item in data:
        for k, v in batch_data.items():
            tmp = data_item[k]
            batch_data[k].append(tmp)
            
    batch_data['sent'] = torch.stack(batch_data['sent'])   
    batch_data['label'] = torch.stack(batch_data['label']).squeeze()
    return batch_data


def get_loader(tokenizer, file, num_workers=multiprocessing.cpu_count()):
    dataset = RelationExtractLoader(tokenizer, file)
    data_loader = DataLoader(dataset=dataset,
                                    batch_size=128,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=downstream_collate_fn,
                                 )
    
    return data_loader, dataset.classes
