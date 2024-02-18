import argparse
from typing import List, Tuple, Dict
from trainer import BertClassifier_relation_prediction, BertClassifier_triple_classification, BERTClassifierNewRP, BERTClassifierNewTC
import warnings
from utils import *

def get_label_dict(args: argparse.Namespace) -> Dict[int, str]:
    # label_mapping = {
    #     "ESCO-CNL": ESCO_CNL_LABEL_DICT,
    #     "handcrafted": ESCO_CNL_LABEL_DICT,
    #     "UI": ESCO_CNL_LABEL_DICT,
    #     "FB15k_test": FB15k_LABEL_DICT,
    #     "FB15k_valid": FB15k_LABEL_DICT,
    #     "WN18RR_test": WN18RR_LABEL_DICT,
    #     "WN18RR_valid": WN18RR_LABEL_DICT,
    #     "stroma_g2_reference": STROMA_G2_REFERENCE_LABEL_DICT, 
    # }

    if "WN18RR" in args.dataset_name:
        label_dict = WN18RR_LABEL_DICT
    elif "FB15k" in args.dataset_name:
        label_dict = FB15k_LABEL_DICT
    elif "CNL" in args.dataset_name or "ESCO" in args.dataset_name or "handcrafted" in args.dataset_name:
        label_dict = ESCO_CNL_LABEL_DICT
    elif "stroma" in args.dataset_name:
        label_dict = STROMA_DATA_LABEL_DICT
    else:
        sys.exit(
            "Could not select relation list, check function "
            "'get_label_dict' if still up to date with the used datasets"
        )

    if (args.num_labels == 2) and (args.task_type == "TC" or args.matcher == "T"):
        return FALSE_TRUE_LABEL_DICT
    
    return label_dict

def get_classifier_and_test_data(args: argparse.Namespace) -> Tuple:
    classifiers = {
        "RP": BertClassifier_relation_prediction,
        "TC": BertClassifier_triple_classification,
        "NewRP": BERTClassifierNewRP,
        "NewTC": BERTClassifierNewTC
    }
    dict_task_type_data = {
        "TC":"TC",
        "RP":"RP",
        "NewRP":"RP",
        "NewTC":"TC"
    }
    dict_data_type = {
        "TC":"triples",
        "RP":"pairs",
        "NewRP":"pairs",
        "NewTC":"triples"
    }
    dict_dataset_names = {
        "ESCO-CNL":"CNL_ESCO",
        "stroma_g2_reference":"stroma_g2_source_stroma_g2_target",
        "stroma_g4_reference":"stroma_g4_source_stroma_g4_target",
        "stroma_g5_reference":"stroma_g5_source_stroma_g5_target",
        "stroma_g6_reference":"stroma_g6_source_stroma_g6_target",
        "stroma_g7_reference":"stroma_g7_source_stroma_g7_target",
    }

    if args.eval_type == "validation":
        args.dataset_name = dict_dataset_names[args.dataset_name]

    task_type_data = dict_task_type_data[args.task_type]
    data_type = dict_data_type[args.task_type]

    classifier = classifiers[args.task_type](model_variant=args.model_variant, num_labels=args.num_labels, model_to_be_loaded=args.model_name, classifier_variant=args.classifier_variant)
    test_pairs = read_processed_var_from_pickle(f"{task_type_data}_{args.eval_type}_{data_type}_{args.dataset_name}", f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
    test_labels = read_processed_var_from_pickle(f"{task_type_data}_{args.eval_type}_labels_{args.dataset_name}", f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
    test_data = None
    
    if args.task_type == "TC":
        test_data = read_processed_csv(f"TC_{args.eval_type}_{args.dataset_name}", f"{args.eval_type}/{args.anchors}/{args.dataset_name}")
        
    return classifier, test_pairs, test_labels, test_data

def main(args: argparse.Namespace) -> None:
    label_dict = get_label_dict(args)
    if label_dict is None:
        sys.exit("Unknown dataset, check main function in evaluate.py")
        
    classifier, test_pairs, test_labels, test_data = get_classifier_and_test_data(args)
    
    full_dataset_name = args.dataset_name + "_" + args.eval_type

    if args.task_type == "TC":
        classifier.evaluate(test_data, test_pairs, test_labels, full_dataset_name, args.model_name, batch_size=1024)
    else:
        if args.num_labels == 2:
            warnings.warn("The task type is RP and number of labels is set to 2, is this correct?")
        classifier.evaluate(test_pairs, test_labels, label_dict, full_dataset_name, args.model_name, batch_size=1024)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model_variant", type=str, required=True, help="Choose which HF model variant to use.")
    parser.add_argument("--classifier_variant", type=str, required=True, help="Choose whether to use BERTForSequenceClassification (default) or Demir's classifier (demir).")
    parser.add_argument("--model_name", type=str, required=True, help="Specify which model should be evaluated")
    parser.add_argument("--eval_type", type=str, required=True, help="Specify whether you want to use a validation or a test set")
    parser.add_argument("--dataset_name", type=str, required=True, help="Give the name of the data set that should be used for evaluation.")
    parser.add_argument("--anchors", required=True, type=str, help="Indicate the percentage of anchors used, for no anchors set to default")    
    parser.add_argument("--num_labels", type=int, required=True, help="Specify the number of labels. If task_type TC or NewTC is used this should be set to 2.")
    parser.add_argument("--task_type", type=str, required=True, help="Specify what task type the to-be-evaluatd model is trained for. Choices are RP, NewRP, TC, NewTC")
    parser.add_argument("--matcher", type=str, required=True, help="Is the evaluated case a match case (=T) or a refinement case (=F)")
    args = parser.parse_args()

    main(args)
