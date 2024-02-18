import argparse
from utils import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_predictions(args):
    pred_df = read_processed_csv(f"{args.model}_{args.test_set}_matcher", "predictions_matcher")

    print(pred_df.columns)

    pred_df = pred_df.drop_duplicates(['head','tail'])

    if args.model == "bert_gpt" or args.model == "fixed_spacy_gpt" or args.model == "fixed_bert_gpt":
        pred_df['label'] = 1

    return pred_df

def get_ground_truth(args):
    ground_truth = read_processed_csv(f"RP_test_{args.test_set}_matcher", f"test/matcher/{args.test_set}_matcher")
    
    ground_truth = ground_truth.drop_duplicates(['head','tail'])

    print(ground_truth[ground_truth['label']==1])

    return ground_truth

def merge_gt_with_pred(ground_truth, predictions):
    predictions['pred_label'] = predictions['label']
    predictions =  predictions.drop('label',axis=1)

    print(sum(predictions['pred_label']))

    ground_truth['head'] = ground_truth['head'].str.lower()
    ground_truth['tail'] = ground_truth['tail'].str.lower()
    predictions['head'] = predictions['head'].str.lower()
    predictions['tail'] = predictions['tail'].str.lower()

    combined = pd.merge(ground_truth, predictions, on=['head','tail'] ,how='left')
    combined['pred_label'] = combined['pred_label'].fillna(0.0)

    print(sum(combined['pred_label']))

    test_merge = pd.merge(predictions, ground_truth, how='left', on=['head','tail'])
    print(test_merge[test_merge['pred_label']==float("nan")])

    return combined

def evaluate(combined):
    relations = sorted(pd.concat([combined['pred_label'], combined['label']]).drop_duplicates().reset_index(drop=True))
    relations = [str(r) for r in relations]

    accuracy = accuracy_score(combined['label'], combined['pred_label'])
    print('Overall Accuracy: %d %%' % (100 * accuracy))

    cm: np.ndarray = confusion_matrix(combined['label'], combined['pred_label'])
    cm_df: pd.DataFrame = pd.DataFrame(cm, index=relations, columns=relations)
    print("Confusion Matrix:")
    print(cm_df)

    print("Classification Report:")
    print(classification_report(combined['label'], combined['pred_label'], target_names=relations))

def main(args: argparse.Namespace) -> None:
    pred = load_predictions(args)

    ground_truth = get_ground_truth(args)

    combined = merge_gt_with_pred(ground_truth, pred)

    evaluate(combined)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--test_set", type=str, required=True, help="Choose which test set to evaluate")
    parser.add_argument("--anchors", type=str, required=True, help="Choose how many achors were used")
    parser.add_argument("--model", type=str, required=True, help="Choose which model was used")
    args = parser.parse_args()

    main(args)
