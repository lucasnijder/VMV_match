import pandas as pd
import openai
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from utils import *

os.environ["OPENAI_API_KEY"] = "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"



from openai import OpenAI
client = OpenAI()

def get_best_matches(formatted_string):
    message = f"From this list of matched occupations, decide which of the pairs is the most correct. Pick at most one pair. Only output the pair without any extra words, characters or symbols. The candidate pairs are: \n {formatted_string}"
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a occupational research expert. You know what tasks and responsibilities every occupation has."},
            {"role": "user", "content": message}
            ]
        )
    answer = completion.choices[0].message.content
    return answer

def create_formatted_string(df_group):
    string = ""
    for _,row in df_group.iterrows():
        string =  string + f"'{row['head']}'-'{row['tail']}' \n"

    return string
    
def pred_and_save(args):
    if 'spacy' in args.model:
        pred = pd.read_csv(f'../data/processed/predictions_matcher/{args.model}_{args.test_set}.csv', sep=';')
        pred = pred.drop(pred.index)
    else:
        pred = pd.read_csv(f'../data/processed/predictions/RP_{args.model}_{args.test_set}_matcher_test_prediction.csv',sep=';')
        pred = pred[pred['prediction']==1]

    grouped = pred.groupby('head')
    preds = list()
    res_df = pd.DataFrame()
    res_df['pred'] = ['nan']*len(grouped.groups)
    res_df['head'] = list(grouped.groups)
    count = 0

    for group_name in grouped.groups:
        formatted_string = create_formatted_string(grouped.get_group(group_name))
        best_matches = get_best_matches(formatted_string)
        preds.append((group_name, best_matches))
        print(count, "out of", len(grouped.groups))
        count+=1
        res_df.loc[res_df['head']==group_name, "pred"] = best_matches
        res_df.to_csv(f'../data/processed/predictions_matcher/{args.model}_GPT-4_{args.test_set}.csv', sep=';', index=False)
    res_df.to_csv(f'../data/processed/predictions_matcher/{args.model}_GPT-4_{args.test_set}.csv', sep=';', index=False)

    df_preds = pd.DataFrame(preds, columns=['head','tail'])

    for idx,row in df_preds.iterrows():
        try:
            df_preds.iloc[idx, 1] = df_preds.iloc[idx, 1].split("-")[1].replace("'","").strip()
        except:
            df_preds.iloc[idx, 1] = df_preds.iloc[idx, 1].replace("'","").strip()

def main(args):
    pred_and_save(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='specify which candidate generator to use (spacy_top5 or WANDB name)')
    parser.add_argument('--test_set', required=True, help='select test set')
    args = parser.parse_args()

    main(args)
