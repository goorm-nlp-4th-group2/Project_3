import pandas as pd
import json
import datasets
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter

def load_parallel(path) :
    path_list = glob(path + "/*.json")
    kor2eng_list = defaultdict(list)
    eng2kor_list = defaultdict(list)

    # 아래에는 한영 데이터의 경로를 넣어주시면 됩니다.

    for p in path_list :
        with open(p) as f :
            json_data = json.load(f)
        if "kor2eng" in p :
            for item in tqdm(json_data["data"]) :
                kor2eng_list["domain"].append(item["domain"])
                kor2eng_list["ko"].append(item["ko_original"])
                kor2eng_list["en"].append(item["en"])
                kor2eng_list["type"].append(p.split("/")[-1].split("_")[0])
        else :
            for item in tqdm(json_data["data"]) :
                eng2kor_list["domain"].append(item["domain"])
                eng2kor_list["en"].append(item["en_original"])
                eng2kor_list["ko"].append(item["ko"])
                eng2kor_list["type"].append(p.split("/")[-1].split("_")[0])
    kor2eng_df = pd.DataFrame.from_dict(kor2eng_list)
    eng2kor_df = pd.DataFrame.from_dict(eng2kor_list)

    original_train = pd.concat([kor2eng_df, eng2kor_df])
    original_train = original_train.drop_duplicates().reset_index(drop = True)
    original_train.loc[original_train.ko.str.startswith(">"), "ko"] = original_train.loc[original_train.ko.str.startswith(">"), "ko"].apply(lambda x : x[1:])
    original_train.loc[original_train.en.str.startswith(">"), "en"] = original_train.loc[original_train.en.str.startswith(">"), "en"].apply(lambda x : x[1:])
    kr_counted = Counter(original_train.loc[original_train.ko.str.contains(">") & original_train.en.str.contains(">"), "ko"].str.split(">").explode().index)
    en_counted = Counter(original_train.loc[original_train.ko.str.contains(">") & original_train.en.str.contains(">"), "en"].str.split(">").explode().index)

    nested_data = pd.DataFrame(kr_counted.items()).loc[(pd.DataFrame(en_counted.items()) == pd.DataFrame(kr_counted.items())).loc[:, 1], :]
    nested_data = nested_data.loc[nested_data.loc[:, 1] == 2, :]
    nested_idx = nested_data.loc[:, 0].unique()
    flatten_data = original_train.loc[nested_idx, ["ko", "en"]].apply(lambda x : x.str.split(">")).explode(["ko", "en"])
    flatten_data.loc[:, "domain"] = original_train.loc[nested_idx, "domain"]
    flatten_data.loc[:, "type"] = original_train.loc[nested_idx, "type"]
    flatten_data = flatten_data.reset_index(drop = True)
    original_train = original_train.drop(nested_idx, axis = "rows")
    original_train = pd.concat([original_train, flatten_data])
    original_train = original_train.loc[original_train.domain == "일상생활", :]
    original_train = original_train.loc[original_train.ko.str.len() < 100, :].reset_index(drop = True)

    return original_train

def __tokenizing(inputs, tokenizer, training, src_lang, tgt_lang):
    model_inputs = tokenizer(inputs[src_lang])
    if training :
        with tokenizer.as_target_tokenizer() :
            model_inputs["labels"] = tokenizer(inputs[tgt_lang])["input_ids"]        
    return model_inputs

def get_dataset(inputs, tokenizer, collator, batch_size, training, src_lang, tgt_lang) :
    inputs = datasets.Dataset.from_pandas(inputs)
    tokenized_inputs = inputs.map(__tokenizing,
                                  batched = True,
                                  fn_kwargs = {"training" : training,
                                               "tokenizer" : tokenizer,
                                               "src_lang" : src_lang,
                                               "tgt_lang" : tgt_lang})
    
    if training :
        columns = tokenizer.model_input_names + ["labels"]
    else :
        columns = tokenizer.model_input_names

    tokenized_inputs.set_format("torch", columns = columns)
    train_dataset = torch.utils.data.DataLoader(tokenized_inputs,
                                                batch_size = batch_size,
                                                shuffle = training,
                                                collate_fn = collator)
    return train_dataset

class EarlyStopping:
    def __init__(self, path, patience = 1, verbose=False, delta = 0):
        """
        Args:
            path (str): checkpoint저장 경로
                            Default: None
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 1
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_pretrained(self.path)
        self.val_loss_min = val_loss