import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

import argparse

model_spacy = spacy.load('en_core_web_lg') # TODO: download the model using python -m spacy download en_core_web_lg

def calculate_spacy_vector(string):
    emb = model_spacy(string)
    return emb.vector

def calculate_embeddings(df, column, filepath_out):
        # Make sure all rows are strings
        df[column] = df[column].astype(str)

        # Make into list, To avoid double work, make set to get unique values
        sentences_unique = list(set(df[column]))

        # Create df of unique sentences
        df_embeddings = pd.DataFrame({'sentences': sentences_unique})
        df_embeddings['vector'] = df_embeddings['sentences'].apply(lambda word: calculate_spacy_vector(word))

        # Write df with sentences and embeddings to pickle file
        df_embeddings.to_pickle(filepath_out)   

        print(f'Wrote embeddings to {filepath_out}')

        return filepath_out

def get_most_similar(vector_a, vector_b):
       return cosine_similarity([vector_a.values[0]], [vector_b.values[0]])[0][0]

def calculate_similarity(df, path_ontA, path_ontB):
    df_A = pd.read_pickle(path_ontA)
    df_B = pd.read_pickle(path_ontB)

    # get_most_similar(vector_a, vector_b)
    df['sims'] = df.apply(lambda row: get_most_similar(df_A.loc[df_A['sentences'] == row['head']]['vector'],df_B.loc[df_B['sentences'] == row['tail']]['vector']), axis=1)
    #df['results'] = df['sims'].apply(lambda x: 0.0 if float(df['sims']) >= 0.8 else 0.0 )
    return df

def main(args):
    # TODO: change the defined paths
    # first file is the test pairs as csv, with column head and tail
    fle = f'../data/processed/test/matcher/{args.test_set}_matcher/RP_test_{args.test_set}_matcher.csv'
    df = pd.read_csv(fle, sep=';')[['head','tail']]
    print("calculating embeddings heads")
    # calculate the embeddings file, to have faster code
    # TODO: only the first time you should do calculate embeddings, after that you can just load them, need addition of a if file exists
    fileout1 = f'../data/processed/misc/embeddings_{args.test_set}_spacy_head.pickle' #
    calculate_embeddings(df, 'head', fileout1)
    print("calculating embeddings tails")
    fileout2 = f'../data/processed/misc/embeddings_{args.test_set}_spacy_tail.pickle' #
    calculate_embeddings(df, 'tail', fileout2)
    # the main part, loads the files in the function
    print("calculating similarity")
    df_with_results = calculate_similarity(df, fileout1, fileout2)
    df_with_results.to_csv(f'../data/processed/predictions_matcher/spacy_{args.test_set}.csv', sep=';')

    print("getting top 5")
    top_k_df = df_with_results.groupby('head').apply(lambda x: x.nlargest(5, 'sims')).reset_index(drop=True)
    top_k_df.to_csv(f'../data/processed/predictions_matcher/spacy_top5_{args.test_set}.csv', sep=';')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data preparations, there are various options.")
    parser.add_argument("--test_set", required=True, type=str, help="test set to predict matches of")
    args = parser.parse_args()
    main(args)