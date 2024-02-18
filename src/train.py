import numpy 
from typing import List
from torch.optim import AdamW
import wandb
import torch
import argparse
from trainer import BertClassifier_relation_prediction, BertClassifier_triple_classification, BERTClassifierNewRP, BERTClassifierNewTC
from utils import *
from sklearn.utils.class_weight import compute_class_weight
import torch.backends.cudnn as cudnn
import os

def initialize_classifier(args) -> object:
    common_args = {
        'model_variant': args.model_variant,
        'num_labels': args.num_labels,
        'parallelize': args.parallelize,
        'model_to_be_loaded': args.general_model,
        'classifier_variant': args.classifier_variant
    }
    classifier = None
    if args.task_type == "RP":
        classifier = BertClassifier_relation_prediction(**common_args)
    elif args.task_type == "TC":
        classifier = BertClassifier_triple_classification(**common_args)
    elif args.task_type == "NewRP":
        classifier = BERTClassifierNewRP(**common_args)
    elif args.task_type == "NewTC":
        classifier = BERTClassifierNewTC(**common_args)
    return classifier


def set_wandb_config(classifier, args, sorted_train_set_names, custom_name):
    safe_model_variant = args.model_variant.replace("/", "-")
    classifier.wandb = wandb.init(project="exploratory_experiments2",
                                  name=custom_name,
                                  entity="lucas-snijder-tno")
    wandb.config.update({
        'task': args.task_type,
        'model': safe_model_variant,
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'parallelized': args.parallelize,
        'train_set': sorted_train_set_names,
        'anchors' : args.anchors,
        'max_sequence_length':BERT_MAX_SEQUENCE_LENGTH,
    })


def main(args) -> None:
    cudnn.benchmark = True

    classifier = initialize_classifier(args)
    sorted_train_set_names = sort_words_str(args.train_set_names)

    if args.train_set_balanced == 'T':
        sorted_train_set_names += '_BA'
    elif args.train_set_balanced == 'F':
        sorted_train_set_names += '_UB'
    else:
        sys.exit('Unknown value for train_set_balanced. Choose from T or F.')

    custom_name = random_name_generator()
    if not HP_TUNING_BOOL:
        set_wandb_config(classifier, args, sorted_train_set_names, custom_name)

    train_sentences, train_labels, train_relations = read_data(args, sorted_train_set_names)

    class_weights = handle_class_weights(args, train_sentences, train_labels)

    classifier.train(train_sentences,
                     train_labels,
                     lr=args.lr,
                     epochs=args.epochs,
                     batch_size=args.batch_size,
                     class_weights=class_weights,
                     relations=train_relations)

    print(args.model_variant.replace("/", "-"), custom_name, classifier)
    save_model_to_pth(args.model_variant.replace("/", "-"), custom_name, classifier)

    if not HP_TUNING_BOOL:
        # the test script needs to know the name of the model, so in order to run training and evaluating sequentially the name needs to be available for the next script
        with open("../prints/miscellaneous/run_info.txt", "a") as file:
            file.write(custom_name + ";" + classifier.wandb.id + "\n")
        
        classifier.wandb.finish()
    else:
        # the test script needs to know the name of the model, so in order to run training and evaluating sequentially the name needs to be available for the next script
        with open("../prints/miscellaneous/run_info.txt", "a") as file:
            file.write(custom_name + ";" + "NO_ID_HP_TUNING" + "\n") 

def read_data(args, sorted_train_set_names):
    data_type = args.task_type
    if data_type == "NewRP":
        data_type = "RP"
    # elif data_type == "NewTC":
    #     data_type = "TC"
        
    if data_type == "TC" or data_type == "NewTC":
        item_type = "triples"
    elif data_type == "RP" or data_type == "NewRP":
        item_type = "pairs"
    else:
        sys.exit("problem with reading data in train.py script")

    train_sentences = read_processed_var_from_pickle(f"{data_type}_train_{item_type}_{sorted_train_set_names}", f"train/{args.anchors}/{sorted_train_set_names[:-3]}")
    train_labels = read_processed_var_from_pickle(f"{data_type}_train_labels_{sorted_train_set_names}", f"train/{args.anchors}/{sorted_train_set_names[:-3]}")
    
    if data_type == "TC":
        train_relations = read_processed_var_from_pickle(f"{data_type}_train_relations_{sorted_train_set_names}", f"train/{args.anchors}/{sorted_train_set_names[:-3]}")
    else:
        train_relations = None

    return train_sentences, train_labels, train_relations


def get_relations_class_weight_matrix(args, train_sentences, train_labels):
    t_rel_dict = {}
    for t, l in zip(train_sentences, train_labels):
        if l == 1:
            if t[1] in t_rel_dict:
                t_rel_dict[t[1]] += 1
            else:
                t_rel_dict[t[1]] = 1

    f_rel_dict = {}
    for t, l in zip(train_sentences, train_labels):
        if l == 0:
            if t[1] in f_rel_dict:
                f_rel_dict[t[1]] += 1
            else:
                f_rel_dict[t[1]] = 1

    total_t = np.sum(list(t_rel_dict.values()))
    total_f = np.sum(list(f_rel_dict.values()))
    total = total_t + total_f

    for key in t_rel_dict.keys():
        t_rel_dict[key] = total_t/t_rel_dict[key]

    for key in f_rel_dict.keys():
        f_rel_dict[key] = total_f/f_rel_dict[key]
    
    replacement_dict = {'broadMatch':0, 'exactMatch':1, 'narrowMatch':2}

    t_rel_dict = {replacement_dict.get(key, key): value for key, value in t_rel_dict.items()}
    f_rel_dict = {replacement_dict.get(key, key): value for key, value in f_rel_dict.items()}

    t_rel_dict = [value for key, value in sorted(t_rel_dict.items())]
    f_rel_dict = [value for key, value in sorted(f_rel_dict.items())]
                         
    class_weights_matrix = [f_rel_dict, t_rel_dict]

    # divide by mean of list for easier weights
    for i in range(len(class_weights_matrix)):
        class_weights_matrix[i] = scale_list_by_mean(class_weights_matrix[i])

    return(class_weights_matrix)


def handle_class_weights(args, train_sentences, train_labels):
    if args.balance_class_weights == 'T':
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    elif args.balance_class_weights == "matrix":
        class_weights = get_relations_class_weight_matrix(args, train_sentences, train_labels)
    else:
        class_weights = None
    return class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_variant", required=True, type=str, help="Choose which HF model variant to use.")
    parser.add_argument("--classifier_variant", required=True, type=str, help="Choose whether to use BERTForSequenceClassification (default) or Demir's classifier (demir).")
    parser.add_argument("--task_type", required=True, type=str, help="Choose which task type to train the model for. Choices are RP, NewRP, TC and NewTC.")
    parser.add_argument("--train_set_names", required=True, type=str, help="Indicate which data sets to use for training. Specify using a comma without spaces between them.")
    parser.add_argument("--train_set_balanced", required=True, type=str, help="Indicate wheter you want to use a train set of which the classes have been balanced. For TC this concerns the relations, not the labels.")
    parser.add_argument("--anchors", required=True, type=str, help="Indicate the percentage of anchors to use, for no anchors set to default")    
    parser.add_argument("--num_labels", required=True, type=int, help="Indicate how many labels the data for the problem at hand has. For TC, the number is 2.")
    parser.add_argument("--balance_class_weights", required=False, type=str, help="Indicate whether you want the model to take into account class imbalances.")
    parser.add_argument("--lr", required=True, type=float, help="Specify the learning rate.")
    parser.add_argument("--epochs", required=True, type=int, help="Specify the number of epochs.")
    parser.add_argument("--batch_size", required=True, type=int, help="Specify the batch size.")
    parser.add_argument("--parallelize", required=True, type=str, help="Indicate whether you want to use parallel computing across multiple GPUs. The visible GPUs are specified in the train.py file.")
    parser.add_argument("--general_model", required=False, type=str, default=None, help="If you want to further fine-tune an already fine-tuned model, specify the model-name here.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
