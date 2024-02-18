import subprocess
import argparse
import sys
import re
import optuna
import wandb
from datetime import datetime
from utils import *

torch.cuda.set_device(1)

def run_script(script, args):
    command = ["python3", script] + args
    subprocess.run(command)

def run_ESCO_ONET_case():
    print("**********************************")
    print("    create train sets ESCO,ONET        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "ONET_ontology_to_triples"])
    run_script("data_preparation.py", ["--action", "ESCO_ontology_to_triples"])

    run_script("data_preparation.py", ["--action", "train_set_converter_matcher", "--data_sets", "ONET,ESCO"])

    print("**********************************")
    print("    create test sets ESCO,ONET        ")
    print("**********************************")
    run_script("data_preparation.py", ["--action", "ESCO-ONET_mapping_to_triples"])
    run_script("data_preparation.py", ["--action", "test_set_converter_matcher", "--data_sets", "ESCO-ONET", "--task_type", "RP"]) 


def run_training():
#GroNLP/bert-base-dutch-cased / bert-base-uncased / dwsunimannheim/TaSeR
    run_script("train.py", ["--model_variant","bert-base-uncased",
                            "--classifier_variant","default",
                            "--task_type","RP",
                            "--train_set_names","ESCO,ONET_matcher", 
                            "--train_set_balanced","F",
                            "--anchors", "matcher",
                            "--num_labels","2",
                            "--lr","0.00001",
                            "--epochs","4",
                            "--batch_size","64",
                            "--parallelize","F",
                            "--balance_class_weights","T",
                            ])
    
def run_model_test():
    run_script("evaluate.py", ["--model_variant", "bert-base-uncased",
                            "--classifier_variant", "default",
                            "--eval_type", "test",
                            "--dataset_name", "ESCO-ONET_matcher",
                            "--anchors", "matcher",
                            "--task_type", "RP",
                            "--model_name", "ashley150",
                            "--num_labels", "2",
                            "--matcher","F"
                            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', required=True)
    args = parser.parse_args()

    if args.exp == 'data_ESCO-ONET':
        run_ESCO_ONET_case()
    elif args.exp == "train":
        run_training()
    elif args.exp == "test":
        run_model_test()
    else:
        sys.exit(f"Unknown experiment: {args.exp}. Try again")
