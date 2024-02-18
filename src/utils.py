import os
import pickle
import random
from typing import List, Any
import pandas as pd
from faker import Faker
import torch
import sys
import numpy as np
import json
from lxml import etree
from rdflib import Graph
import inspect

### set to true if HP tuning
HP_TUNING_BOOL = False

### Torch settings
torch.cuda.is_available()
torch.cuda.device_count()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.cuda.set_device(0)

### Set constants and seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

### BERT max sequence length constant
BERT_MAX_SEQUENCE_LENGTH = 100

### false/true constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/false_true_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
FALSE_TRUE_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
FALSE_TRUE_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/false_true_relations.txt', header=None)[0])

### ESCO-CNL constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/ESCO_CNL_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
ESCO_CNL_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
ESCO_CNL_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/ESCO_CNL_relations.txt', header=None)[0])

### WN18RR constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/WN18RR_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
WN18RR_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
WN18RR_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/WN18RR_relations.txt', header=None)[0])

### FB15K constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/FB15k_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
FB15k_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
FB15k_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/FB15k_relations.txt', header=None)[0])

### STROMA datasets constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/STROMA_data_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
STROMA_DATA_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
STROMA_DATA_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/STROMA_data_relations.txt', header=None)[0])

### STROMA model constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/STROMA_model_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
STROMA_MODEL_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
STROMA_MODEL_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/STROMA_model_relations.txt', header=None)[0])

### TaSeR model constants
with open('/home/lucas.snijder/internship_lucas/data/processed/relation_information/taser_label_dict.json', "r") as f:
   loaded_dict = json.load(f)
TASER_LABEL_DICT = {int(k): v for k, v in loaded_dict.items()}
TASER_RELATIONS = list(pd.read_csv('/home/lucas.snijder/internship_lucas/data/processed/relation_information/taser_relations.txt', header=None)[0])

def read_raw_csv(file_name: str, delimiter: str = ';', encoding: str = 'latin-1', header: str = 'infer') -> pd.DataFrame:
    filepath = f'../data/raw/{file_name}.csv'
    raw_csv: pd.DataFrame = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding, header=header)
    print(f"succesfully read csv {file_name} from ../data/raw")
    return raw_csv

def read_raw_rdf(file_name: str, returns: str):
    if returns == "tree":
        tree = etree.parse(f'../data/raw/{file_name}.rdf')
        print(f"succesfully read rdf {file_name} from ../data/raw")
        return tree
    elif returns == "graph":
        g = Graph()
        g.parse(f'../data/raw/{file_name}.rdf', format="application/rdf+xml")
        print(f"succesfully read rdf {file_name} from ../data/raw")
        return g
    else:
        sys.exit("Error in read raw rdf function in")

def read_txt_to_df(file_name: str, source_folder: str, columns: list = None) -> pd.DataFrame:
    with open(f'../data/{source_folder}/{file_name}.txt', 'r', encoding='utf-8') as txt_file:
        rows = txt_file.readlines()
        data = [row.strip().split('\t') for row in rows]
        if columns is not None:
            df = pd.DataFrame(data, columns=columns)
        else:
            df = pd.DataFrame(data)
    return df

def read_processed_csv(file_name: str, type_of_file: str) -> pd.DataFrame:
    filepath = f'../data/processed/{type_of_file}/{file_name}.csv'
    processed_csv: pd.DataFrame = pd.read_csv(filepath, delimiter=';', encoding="utf-8")
    print(f"succesfully read csv {file_name} from ../data/processed/{type_of_file}")
    return processed_csv

def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_processed_df_to_csv(processed_df: pd.DataFrame, file_name: str, type_of_file: str, sep:str = ";") -> None:
    folder_path = f'../data/processed/{type_of_file}'
    create_folder_if_not_exists(folder_path)
    processed_df.to_csv(f'{folder_path}/{file_name}.csv', index=False, sep=sep, encoding="utf-8")
    print(f"succesfully saved df {file_name} to {folder_path}")

def save_processed_var_to_pickle(list_to_save: List[Any], file_name: str, type_of_file: str) -> None:
    folder_path = f'../data/processed/{type_of_file}'
    create_folder_if_not_exists(folder_path)
    with open(f'{folder_path}/{file_name}.pkl', 'wb') as f:
        pickle.dump(list_to_save, f)
    print(f"succesfully saved pickle file {file_name} to {folder_path}")
    
def read_processed_var_from_pickle(file_to_read: str, type_of_file: str) -> List[Any]:
    with open(f'../data/processed/{type_of_file}/{file_to_read}.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    print(f"succesfully read pickle file {file_to_read} from ../data/processed/{type_of_file}")
    return loaded_data

def save_model_to_pth(safe_model_variant: str, model_name: str, classifier) -> None:
    os.makedirs(f"../models/{safe_model_variant}", exist_ok=True)
    weights_filename: str = f"../models/{safe_model_variant}/{model_name}_weights.pth"
    torch.save(classifier.model.state_dict(), weights_filename)
    print(f"succesfully saved model {model_name} to ../models/{safe_model_variant}/{model_name}_weights.pth")

def save_printed_output_to_file(list_vars_to_print, file_name, file_type, append_or_write="w"):
    with open(f"../prints/{file_type}/{file_name}.txt", append_or_write) as file:
        original_stdout = sys.stdout
        sys.stdout = file

        for var in list_vars_to_print:
            print(var)

        sys.stdout = original_stdout    

    print(f"succesfully saved printed output to ../prints/{file_type}/{file_name}.txt")



def random_name_generator() -> str:
    fake: Faker = Faker()
    name: str = fake.first_name().lower()
    name_gen_random = random.Random()
    number: str = str(name_gen_random.randint(100,999))
    print(f"Random name generated: {name + number}")
    return name + number

def left_fill_str_with_zeroes(series: pd.Series, total_length: int) -> pd.Series:
    filled_series = series.apply(lambda x: str(x).rjust(total_length, '0'))

    return filled_series

def create_CNL_occupation_label_dict(df_ontology: pd.DataFrame) -> dict:
    column_pairs = [
        ('BEROEPS_CODE', 'OMSCHRIJVING_BEROEP'),
        ('code 5e laag', 'naam 5e laag'),
        ('isco code UG', 'isco EN unit group'),
        ('isco code MiG', 'isco EN minor group'),
        ('isco code sub MG', 'isco EN sub-major group'),
        ('isco code MG', 'isco EN major group')
    ]
    label_dict = {}
    for code_column, label_column in column_pairs:
        label_dict.update(dict(zip(df_ontology[code_column], df_ontology[label_column])))

    return label_dict

def softmax(logits):
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / sum_exp_logits
    return probabilities

def sort_words_str(s):
    s_list = s.split(',')
    sorted_s_list = sorted(s_list)
    sorted_s = ''
    for word in sorted_s_list:
        sorted_s =  sorted_s + word + "_"
    sorted_s = sorted_s[:-1]  
    return sorted_s

def is_nested(my_list):
    if my_list is not None:
        return any(isinstance(i, list) for i in my_list)

def scale_list_by_mean(my_list):
    mean = sum(my_list) / len(my_list) 
    scaled_list =  [x / mean for x in my_list]
    return scaled_list  

def print_with_line_number(*args):
    frame = inspect.currentframe().f_back
    current_line = frame.f_lineno
    print(f"-- Line {current_line}: ", end="")
    for arg in args:
        print(arg, end=" ")
    print()