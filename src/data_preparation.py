import argparse
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from utils import *
import random
import re
import itertools
from sklearn.model_selection import train_test_split

class MappingPreprocessor:
    def __init__(self):
        self.mapping = None
        self.case_name = None

    def process_raw_mapping(self):
        raise NotImplementedError

    def save_mapping_triples(self):
        save_processed_df_to_csv(
            self.mapping, f"{self.case_name}_triples", "triples"
        )

    def drop_duplicate_rows(self):
        duplicates = self.mapping[
            self.mapping.duplicated(subset=['head', 'tail'], keep='first') |
            self.mapping.duplicated(subset=['head', 'tail'], keep='last')
        ]
        print(f"{len(duplicates)} duplicate rows are dropped")
        self.mapping = self.mapping.drop_duplicates(
            subset=['head', 'tail'], keep='first'
        )

    
class OntologyPreprocessor:
    """
    Goal is to go from raw data to pkl of pair/triple and pkl of label.
    """

    def __init__(self):
        self.case_name = None
        self.ontology = None
        self.graph = None
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])

    def convert_to_graph(self):
        raise NotImplementedError

    def generate_subsumption_triples(self) -> pd.DataFrame:
        triples = []
        for edge in self.graph.edges():
            parent, child = edge
            if not parent.lower() == child.lower():
                triples.append((parent.lower(), 'narrowMatch', child.lower()))
                triples.append((child.lower(), 'broadMatch', parent.lower()))

        df_triples = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])

        self.triples = pd.concat([self.triples, df_triples], ignore_index=True)
        print(f"number of narrowMatch triples = {len(self.triples[self.triples['relation']=='narrowMatch'])}")
        print(f"number of broadMatch triples = {len(self.triples[self.triples['relation']=='broadMatch'])}")

    def generate_selfloop_exactMatch_triples(self):
        selfloop_exactMatches = self.triples.copy()
        selfloop_exactMatches['tail'] = selfloop_exactMatches['head']
        selfloop_exactMatches['relation'] = 'exactMatch'
        selfloop_exactMatches = selfloop_exactMatches.drop_duplicates()
        
        self.triples = pd.concat([self.triples, selfloop_exactMatches], ignore_index=True)

    def save_triples(self):
        save_processed_df_to_csv(
            self.triples, f"{self.case_name}_triples", "triples"
        )

    def get_triples(self):
        return self.triples
    
    def remove_duplicate_pairs(self):
        triples = self.triples
        len_before = len(triples)
                
        triples['is_exactMatch'] = triples['relation'] == 'exactMatch'
        triples = triples.sort_values(by='is_exactMatch')
        triples = triples.drop('is_exactMatch', axis=1)

        triples = triples.drop_duplicates(['head','tail'], keep='first')
        len_after = len(triples)
        len_diff = len_before - len_after

        self.triples = triples

        print(f"Removed {len_diff} duplicate pairs")

    def print_triples_statistics(self) -> None:
        print("-------------- Triple statistics ------------------")
        print(f'Number of triples: {len(self.triples)}')
        # print("Triples:")
        # for i, row in self.triples.iterrows():
        #     print(tuple(row))
        #     if i > 2:
        #         break

        relation_counts = self.triples['relation'].value_counts().to_dict()
        print("Relation Counts Top 20:")
        counter = 0
        for relation, count in relation_counts.items():
            print(f"{relation}: {counter}")
            if counter > 18:
                break



class StromaExternalDataPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        self.case_name = case_name
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])

        if self.case_name == 'wikidata':
            self.triples = read_raw_csv(
                f"stroma_train_data_{self.case_name}",
                delimiter=',',
                encoding='utf-8'
            )
        else:
            self.triples = read_raw_csv(f"stroma_train_data_{self.case_name}")

        # there were some ; in the strings so they are replaced by ',' to counter delim errors
        self.triples = self.triples.replace({';': ','}, regex=True)

    def convert_ints_to_labels(self):
        self.triples = self.triples.replace(STROMA_MODEL_LABEL_DICT)

class TaSeRPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        self.case_name = case_name
        self.triples = read_raw_csv("taser_trainset", delimiter=",", header=None)
        self.triples.columns = ['head','tail','relation']

        self.triples['relation'] = self.triples['relation'].map(TASER_LABEL_DICT)


class WN18RRDataPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        self.case_name = case_name
        columns = ['head', 'relation', 'tail']
        self.triples = pd.DataFrame(columns=columns)
        dirty_csv = read_txt_to_df(case_name, source_folder="raw", columns=columns)

        dirty_csv['head'] = dirty_csv['head'].apply(
            lambda x: re.sub(r'\.\w\.\d{2}', '', x)
        )
        dirty_csv['relation'] = dirty_csv['relation'].apply(
            lambda x: x.lstrip('_')
        )
        dirty_csv['tail'] = dirty_csv['tail'].apply(
            lambda x: re.sub(r'\.\w\.\d{2}', '', x)
        )

        self.triples = dirty_csv
        self.triples = self.triples[
            self.triples['relation'].isin(WN18RR_RELATIONS)
        ]

        duplicates = self.triples[
            self.triples.duplicated(subset=['head', 'tail'], keep='first') |
            self.triples.duplicated(subset=['head', 'tail'], keep='last')
        ]
        print(f"{len(duplicates)} duplicate rows are dropped")
        self.triples = self.triples.drop_duplicates(
            subset=['head', 'tail'], keep='first'
        )

class FB15kDataPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        self.case_name = case_name
        self.triples = read_raw_csv(case_name)


class DataSetConverter:
    def __init__(self, set_name, case_name):
        self.set_name = set_name
        self.case_name = case_name
        self.balanced_bool = False 
        self.triples = None
        self.validation_triples = None
        self.test_triples = None
        self.test_case_name = None

    def add_pos_label_to_df(self, set_name):

        if set_name == "train":
            self.triples['label'] = 1
        elif set_name == "validation":
            self.validation_triples['label'] = 1
        elif set_name == "test":
            self.test_triples['label'] = 1
        else:
            sys.exit("something is wrong in add_pos_label_to_dict")

    def shuffle_triples_random(self):
        self.triples = self.triples.sample(frac=1, random_state=234).reset_index(drop=True)

    def add_inverse_subsumptions(self):
        inverse_triples = self.triples[
            self.triples['relation'].isin(['narrowMatch', 'broadMatch'])
        ].copy()
        
        inverse_triples[['head', 'tail']] = inverse_triples[['tail', 'head']]
        inverse_triples['relation'] = inverse_triples['relation'].replace(
            {'narrowMatch': 'broadMatch', 'broadMatch': 'narrowMatch'}
        )
        
        self.triples = pd.concat([self.triples, inverse_triples], ignore_index=True)

    def drop_rows_with_na(self, set_name):
        set_types = {"train": self.triples,
                        "test": self.test_triples,
                        "validation": self.validation_triples}
        triples = set_types[set_name]

        mask = triples[['head', 'relation', 'tail']].isna().any(axis=1)
        rows_without_na = triples[~mask]
        print(f"{len(triples[mask])} rows contained NA's and were dropped")

        if set_name == "train":
            self.triples = rows_without_na
        elif set_name == "validation":
            self.validation_triples = rows_without_na
        elif set_name == "test":
            self.test_triples = rows_without_na

    def generate_all_negatives(self, set_name):

        set_types = {"train": self.triples,
                        "test": self.test_triples,
                        "validation": self.validation_triples}
        true_triples = set_types[set_name]

        if hasattr(self, 'case_name'):
            case_name_str = self.case_name
        elif hasattr(self, 'case_name'):
            case_name_str = self.case_name

        if "WN18RR" in case_name_str:
            relation_list = WN18RR_RELATIONS
        elif "CNL" in case_name_str or "ESCO" in case_name_str or "handcrafted" in case_name_str:
            relation_list = ESCO_CNL_RELATIONS
        elif "stroma" in case_name_str:
            relation_list = STROMA_DATA_RELATIONS
        else:
            sys.exit(
                "Could not select relation list, check function "
                "'generate_all_negatives' if still up to date with the used datasets"
            )

        false_triples = []
        for _, row in true_triples.iterrows():
            for relation in relation_list:
                if relation != row['relation']:
                    false_triples.append([row['head'], relation, row['tail'], 0])

        false_triples_df = pd.DataFrame(
            false_triples, columns=['head', 'relation', 'tail', 'label']
        )

        true_triples = pd.concat([true_triples, false_triples_df], ignore_index=True)
    
        if set_name == "train":
            self.triples = true_triples
        elif set_name == "test":
            self.test_triples = true_triples
        elif set_name == "validation":
            self.validation_triples = true_triples

    def drop_duplicate_rows(self, set_name):
        set_types = {"train": self.triples,
                    "validation": self.validation_triples,
                    "test": self.test_triples}
        triples = set_types[set_name]

        duplicates = triples[
            triples.duplicated(subset=['head', 'tail'], keep='first') |
            triples.duplicated(subset=['head', 'tail'], keep='last')
        ]
        print(f"{len(duplicates)} duplicate rows are dropped")

        # Directly modify the attribute in the self object
        if set_name == "train":
            self.triples = triples.drop_duplicates(subset=['head', 'tail'], keep='first')
        elif set_name == "validation":
            self.validation_triples = triples.drop_duplicates(subset=['head', 'tail'], keep='first')
        elif set_name == "test":
            self.test_triples = triples.drop_duplicates(subset=['head', 'tail'], keep='first')


    def save_to_triples_TC(self, set_name, anchor_folder):
        set_types = {"train": self.triples,
                     "validation": self.validation_triples,
                     "test": self.test_triples}
        triples = set_types[set_name]

        if (set_name == "test") and (anchor_folder != "default"):
            self.case_name = self.test_case_name
        
        triples_list = list(
            triples[['head', 'relation', 'tail']].itertuples(
                index=False, name=None)
        )
        labels = list(triples['label'])
        le = LabelEncoder()
        int_relations = le.fit_transform(triples['relation'])
        
        if self.balanced_bool == True:
            balanced_str = "_BA"
        elif self.balanced_bool == False:
            balanced_str = "_UB"
        elif self.balanced_bool == None:
            balanced_str = ""
        else:
            sys.exit("Something is wrong with you entry for balancing the trainset, check the save_to_triples_TC function")

        if set_name in ['validation','test']:
            balanced_str = ""

        save_processed_df_to_csv(
            triples, f"TC_{set_name}_{self.case_name}{balanced_str}", f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            triples_list, f'TC_{set_name}_triples_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            labels, f'TC_{set_name}_labels_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

        if set_name == "train":
            save_processed_var_to_pickle(
                int_relations, f'TC_{set_name}_relations_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

    def save_to_pairs_RP(self, set_name, anchor_folder):

        set_types = {"train": self.triples,
                     "validation": self.validation_triples,
                     "test": self.test_triples}
        triples = set_types[set_name]

        if (set_name == "test") and (anchor_folder != "default"):
            self.case_name = self.test_case_name

        pairs_list = list(
            triples[['head', 'tail']].itertuples(index=False, name=None)
        )
        le = LabelEncoder()
        integer_labels = le.fit_transform(triples['relation'])

        if self.balanced_bool == True:
            balanced_str = "_BA"
        elif self.balanced_bool == False:
            balanced_str = "_UB"

        if set_name in ['validation','test']:
            balanced_str = ""

        save_processed_df_to_csv(
            triples, f"RP_{set_name}_{self.case_name}{balanced_str}", f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            pairs_list, f'RP_{set_name}_pairs_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")
        save_processed_var_to_pickle(
            integer_labels, f'RP_{set_name}_labels_{self.case_name}{balanced_str}', f"{set_name}/{anchor_folder}/{self.case_name}")

class TrainSetConverter(DataSetConverter):
    def __init__(self, case_name):
        super().__init__("train", case_name) 
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])
        ontology_list = sorted(case_name.split(","))
        self.case_name = sort_words_str(case_name)
        print(self.case_name)
        for ontology in ontology_list:
            self.triples = pd.concat(
                [self.triples, read_processed_csv(f"{ontology}_triples", "triples")],
                ignore_index=True
            )


    def split_into_train_and_val(self):
        # Calculate 20% count for the split
        total_count = len(self.triples)
        count_20_percent = total_count // 5

        # Calculate the distribution of relations
        relation_counts = self.triples['relation'].value_counts()
        relation_proportions = relation_counts / total_count

        validation_triples = pd.DataFrame(columns=self.triples.columns)
        train_triples = pd.DataFrame(columns=self.triples.columns)

        # Stratified sampling based on relations
        for relation, count in relation_counts.items():
            relation_triples = self.triples[self.triples['relation'] == relation]

            # Number of samples for this relation in validation set
            val_count = int(relation_proportions[relation] * count_20_percent)

            # Split for this relation
            val_relation_triples = relation_triples.sample(n=val_count)
            train_relation_triples = relation_triples.drop(val_relation_triples.index)

            # Concatenate with the overall train and validation sets
            validation_triples = pd.concat([validation_triples, val_relation_triples])
            train_triples = pd.concat([train_triples, train_relation_triples])

        # Remove duplicates if necessary
        validation_triples = validation_triples.drop_duplicates(['head','tail'])

        # Assign back to the class attributes
        self.triples = train_triples
        self.validation_triples = validation_triples
            
        # self.triples, self.validation_triples = train_test_split(self.triples, 
        #                                                               test_size=0.2,
        #                                                               stratify=self.triples['relation']) 
        
    def undersample_classes(self):
        self.triples = self.triples.iloc[::-1].reset_index(drop=True)
        count_min_class = self.triples['relation'].value_counts().min()
        dfs = []
        for label in self.triples['relation'].unique():
            df_class = self.triples[self.triples['relation'] == label]
            df_class_downsampled = resample(
                df_class, replace=False, n_samples=count_min_class, random_state=234
            )
            dfs.append(df_class_downsampled)
        df_balanced = pd.concat(dfs, ignore_index=True)
        self.triples = df_balanced
        self.balanced_bool = True

    def balance_relations(self) -> None:
        subsumption_mask = self.triples['relation'].isin(['broadMatch','narrowMatch'])
        subsumption_relation_counts = self.triples[subsumption_mask]['relation'].value_counts()
        max_count = np.max(subsumption_relation_counts)
        exactMatch_count = len(self.triples[self.triples['relation']=='exactMatch'])

        if max_count > exactMatch_count:
            taser_trainset = read_raw_csv("wordnet_triples")
            taser_trainset = taser_trainset.dropna()
            taser_trainset_equivalence = taser_trainset[taser_trainset['relation']=="exactMatch"]
            samples_equivalence = taser_trainset_equivalence.sample(n=(max_count-exactMatch_count+1))
            self.triples = pd.concat([self.triples, samples_equivalence])
        else:
            print("There are more exactMatch triples than subsumptions, so no extra exactMatches are added")

    # def over_under_sample_classes(self):
    #     self.balanced_bool = True

    #     self.triples = self.triples.iloc[::-1].reset_index(drop=True)
    #     count_min_class = self.triples['relation'].value_counts().min()
    #     count_max_class = self.triples['relation'].value_counts().max()
    #     middle_point = 10000#int((count_min_class+count_max_class)/3)

    #     print(len(self.triples))

    #     sampled_df = pd.DataFrame()
    #     for value in self.triples['relation'].unique():
    #         subset = self.triples[self.triples['relation'] == value]
            
    #         if len(subset) > middle_point:
    #             r = False
    #         else:
    #             r= True

    #         subset_sampled = subset.sample(n=middle_point, replace=r)
            
    #         sampled_df = pd.concat([sampled_df, subset_sampled], ignore_index=True)

    #     self.triples = sampled_df      

    def shuffle_triples_grouped(self, batch_size=16):
        grouped = self.triples.groupby(['head', 'tail'])
        group_keys = list(grouped.groups.keys())
        random.Random(3).shuffle(group_keys)
        batches = []
        current_batch = []

        for key in group_keys:
            group = grouped.get_group(key)
            indexes = group.index.tolist()
            random.Random(3).shuffle(indexes)
            current_batch.extend(indexes)

            if len(current_batch) >= batch_size:
                batches.append(current_batch[:batch_size])
                current_batch = current_batch[batch_size:]

        if current_batch:
            batches.append(current_batch)

        self.triples = pd.concat(
            [self.triples.loc[batches[i]] for i in range(len(batches))],
            ignore_index=True
        )

    # def convert_triples_to_triples_and_labels(self, args):
    #     triples = list(self.triples[['head', 'relation', 'tail']].itertuples(index=False, name=None))
    #     labels = list(self.triples['label'])

    #     le = LabelEncoder()
    #     relations = le.fit_transform(self.triples['relation'])

    #     if self.balanced_bool:
    #         save_processed_df_to_csv(self.triples, f"{args.task_type}_train_{self.case_name}_BA", f"train/{self.case_name}")
    #         save_processed_var_to_pickle(triples, f'{args.task_type}_train_triples_{self.case_name}_BA', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(labels, f'{args.task_type}_train_labels_{self.case_name}_BA', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(relations, f'{args.task_type}_train_relations_{self.case_name}_BA', f"train/{self.case_name}")
    #     else:
    #         save_processed_df_to_csv(self.triples, f"{args.task_type}_train_{self.case_name}_UB", f"train/{self.case_name}")
    #         save_processed_var_to_pickle(triples, f'{args.task_type}_train_triples_{self.case_name}_UB', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(labels, f'{args.task_type}_train_labels_{self.case_name}_UB', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(relations, f'{args.task_type}_train_relations_{self.case_name}_UB', f"train/{self.case_name}")

    # def convert_triples_to_pairs_and_labels(self):
    #     pairs = list(self.triples[['head', 'tail']].itertuples(index=False, name=None))
    #     le = LabelEncoder()
    #     integer_labels = le.fit_transform(self.triples['relation'])
    #     label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    #     flipped_label_mapping = {v: k for k, v in label_mapping.items()}

    #     if self.balanced_bool:
    #         save_processed_df_to_csv(self.triples, f"RP_train_{self.case_name}_BA", f"train/{self.case_name}")
    #         save_processed_var_to_pickle(pairs, f'RP_train_pairs_{self.case_name}_BA', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(integer_labels, f'RP_train_labels_{self.case_name}_BA', f"train/{self.case_name}")
    #     else:
    #         save_processed_df_to_csv(self.triples, f"RP_train_{self.case_name}_UB", f"train/{self.case_name}")
    #         save_processed_var_to_pickle(pairs, f'RP_train_pairs_{self.case_name}_UB', f"train/{self.case_name}")
    #         save_processed_var_to_pickle(integer_labels, f'RP_train_labels_{self.case_name}_UB', f"train/{self.case_name}")

    def print_triples_statistics(self) -> None:
        print("-------------- Triple statistics ------------------")
        print(f'Number of triples: {len(self.triples)}')
        # print("Triples:")
        # for i, row in self.triples.iterrows():
        #     print(tuple(row))
        #     if i > 2:
        #         break

        relation_counts = self.triples['relation'].value_counts().to_dict()
        print("Relation Counts Top 20:")
        counter = 0
        for relation, count in relation_counts.items():
            print(f"{relation}: {count}")
            if counter >= 19:
                break
            counter += 1

        if "label" in self.triples.columns:
            label_counts = self.triples['label'].value_counts().to_dict()
            print("Label Counts:")
            for label, count in label_counts.items():
                print(f"{label}: {count}")


class ONETOntologyPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name

    def read_SOC(self):
        soc = read_raw_csv("SOC")

        soc_list = []
        for row_idx in range(len(soc)):
            for column_idx in range(4):
                if isinstance(soc.iloc[row_idx,column_idx], str):
                    soc_list.append((soc.iloc[row_idx,column_idx], soc.iloc[row_idx,4]))

        df_soc = pd.DataFrame(soc_list, columns=['code','label'])

        df_soc['code'] = df_soc['code'].apply(lambda x: x[:6] + '-' + x[6:] + "-00")
        df_soc['code'] = df_soc['code'].apply(lambda x: x[:4] + '-' + x[4:])
 
        self.soc = df_soc

    def read_ONET(self):
        df_onet = read_raw_csv("ONET")
        df_onet = df_onet[["ï»¿O*NET-SOC 2019 Code", "O*NET-SOC 2019 Title"]]
        df_onet.columns = ['code','label']

        df_onet['code'] = df_onet['code'].apply(lambda x: x.replace('.','-'))
        df_onet['code'] = df_onet['code'].apply(lambda x: x[:6] + '-' + x[6:])
        df_onet['code'] = df_onet['code'].apply(lambda x: x[:4] + '-' + x[4:])

        self.onet = df_onet
    
    # def merge_SOC_and_ONET(self):
    #     #self.ontology = pd.concat([self.soc, self.onet])
    #     pass

    def convert_to_graph(self):
        G= nx.DiGraph()
        for code in self.onet['code']:
            # Split the code into its components
            parts = code.split('-')

            # Add edges for each parent-child pair
            first = parts[0] + '-0-00-0-00'
            second = parts[0] + '-' + parts[1] + '-00-0-00'
            third = parts[0] + '-' + parts[1] + '-' + parts[2] + '-0-00'
            fourth = parts[0] + '-' + parts[1] + '-' + parts[2] + '-' + parts[3] + '-00'
            fifth = parts[0] + '-' + parts[1] + '-' + parts[2] + '-' + parts[3] + '-' + parts[4]

            G.add_edge(first, second)
            G.add_edge(second, third)
            G.add_edge(third, fourth)
            G.add_edge(fourth, fifth)

                # parent = parts[i]
                # current_full_code = self.get_full_code(i, first, second, third, fourth)

                # child = parts[i+1]
                # child_full_code = self.get_full_code(i+1, child)



                # # Avoid adding empty nodes and self-loops
                # if current_full_code and current_full_code != child_full_code:
                #     print(current_full_code, child_full_code)
                #     G.add_edge(current_full_code, child_full_code)


        soc_label_dict = self.soc.set_index('code')['label'].to_dict()
        onet_label_dict = self.onet.set_index('code')['label'].to_dict()
        for node in G.nodes:
            if len(node) == 2:
                G = nx.relabel_nodes(G, {node: node})
            if len(node) == 6:
                G = nx.relabel_nodes(G, {node: node})
            if len(node) == 8:
                G = nx.relabel_nodes(G, {node: node})

        H = nx.relabel_nodes(G, soc_label_dict)
        H2 = nx.relabel_nodes(H, onet_label_dict)

        self.graph = H2
        
    def fix_inconsistencies(self):
        triples = self.triples

        replace_dict = {"51-5-00-0-00":"Printing Workers",
                        "15-1-00-0-00":"Computer Occupations",
                        "29-1-22-0-00":"Physicians",
                        "31-1-00-0-00":"Home Health and Personal Care Aides; and Nursing Assistants, Orderlies, and Psychiatric Aides"}
        triples = triples.replace(replace_dict)

        self.triples = triples

class ESCOOntologyPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name
        self.ontology = read_raw_csv("ESCO_engels", encoding="utf-8")
        self.ontology = self.ontology[['parentLabel','childLabel']]
        self.ontology = self.ontology.rename({"parentLabel":'head','childLabel':'tail'}, axis=1)

    def generate_exactMatch_ESCO(self) -> None:
        raw_altLabels = read_raw_csv('raw_translated_synonyms')
        raw_altLabels = raw_altLabels.drop_duplicates('preferredLabel', keep='first')
        raw_altLabels = raw_altLabels.dropna()

        altlabel_exactMatches = raw_altLabels[['preferredLabel','altLabels']].rename({'preferredLabel':'head','altLabels':'tail'}, axis=1)
        altlabel_exactMatches['head'] = altlabel_exactMatches['head'].str.lower()
        altlabel_exactMatches['tail'] = altlabel_exactMatches['tail'].str.lower()
        altlabel_exactMatches['relation'] = 'exactMatch'

        # inverse_exactMatches = pd.DataFrame()
        # inverse_exactMatches[['head','tail']] = exactMatches[['tail','head']]
        # inverse_exactMatches['relation'] = 'exactMatch'

        self.triples = pd.concat([self.triples, altlabel_exactMatches], ignore_index=True) #, inverse_exactMatches

    # def preprocess_raw_ontology(self):
        
    #     print(self.ontology.columns)

    #     self.ontology.columns = ['head_URI', 'relation_URI', 'tail_URI', 'head', 'tail']

    def convert_to_graph(self):
        G = nx.DiGraph()

        for index, row in self.ontology.iterrows():
            G.add_edge(row['head'], row['tail'])

        self.graph = G
    
    
class CNLOntologyPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        super().__init__()
        self.case_name = case_name
        self.ontology = read_raw_csv("CNL_engels", encoding='cp1252')

    def preprocess_raw_ontology(self) -> pd.DataFrame:
        self.ontology = self.ontology.astype(str)
        columns_and_lengths = [
            ('code 5e laag', 5),
            ('isco code UG', 4),
            ('isco code MiG', 3),
            ('isco code sub MG', 2)
        ]
        for column, length in columns_and_lengths:
            self.ontology[column] = left_fill_str_with_zeroes(
                self.ontology[column], length)
            
    def convert_to_graph(self) -> nx.DiGraph:
        label_dict = create_CNL_occupation_label_dict(self.ontology)
        self.ontology['fullPath'] = self.ontology[
            ['isco code MG', 'isco code sub MG', 'isco code MiG',
             'isco code UG', 'code 5e laag', 'BEROEPS_CODE']
        ].agg('-'.join, axis=1)

        G = nx.DiGraph()

        for path in self.ontology['fullPath']:
            nodes = path.split('-')
            for i in range(len(nodes) - 1):
                G.add_edge(nodes[i], nodes[i + 1])

        nx.set_node_attributes(G, label_dict, 'label')

        named_G = nx.DiGraph()

        for node in G.nodes():
            node_label = G.nodes[node]['label']
            named_G.add_node(node_label, id=node)

        for edge in G.edges():
            parent, child = edge
            parent_label = G.nodes[parent]['label']
            child_label = G.nodes[child]['label']
            named_G.add_edge(parent_label, child_label)

        self_loops = list(nx.selfloop_edges(named_G))
        named_G.remove_edges_from(self_loops)

        self.graph = named_G

    def remove_plural_subsumptions(self):
        """
        Removes plural subsumptions from the DataFrame 'triples' where either the head is the plural of the tail
        or vice versa.
        """
        len_before = len(self.triples)

        mask = ~((self.triples['head'] == self.triples['tail'] + 's') | (self.triples['tail'] == self.triples['head'] + 's'))
        
        self.triples = self.triples[mask]
        len_after = len(self.triples)
        
        diff = len_before - len_after
        print(f'Number of plural subsumptions dropped: {diff}')


class TestSetConverter(DataSetConverter):
    def __init__(self, case_name):
        super().__init__("test", case_name) 
        self.test_triples = read_processed_csv(f"{case_name}_triples", "triples")
        self.balanced_bool = None

    # def save_to_pairs_RP(self, set_name: str):

    #     set_types = {"test": self.test_triples,
    #                  "validation": self.validation_triples}
    #     triples = set_types[set_name]

    #     pairs = list(
    #         triples[['head', 'tail']].itertuples(index=False, name=None)
    #     )
    #     le = LabelEncoder()
    #     integer_labels = le.fit_transform(triples['relation'])
    #     label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    #     save_processed_df_to_csv(
    #         triples, f"RP_{self.case_name}_{set_name}", f"{set_name}/{self.case_name}"
    #     )
    #     save_processed_var_to_pickle(
    #         pairs, f'RP_{self.case_name}_{set_name}_pairs', f"{set_name}/{self.case_name}"
    #     )
    #     save_processed_var_to_pickle(
    #         integer_labels, f'RP_{self.case_name}_{set_name}_labels', f"{set_name}/{self.case_name}"
    #     )

    # def save_to_triples_TC(self, set_name):
    #     set_types = {"test": self.test_triples,
    #                  "validation": self.validation_triples}
    #     triples = set_types[set_name]
        
    #     triples_list = list(
    #         triples[['head', 'relation', 'tail']].itertuples(
    #             index=False, name=None)
    #     )
    #     labels = list(triples['label'])

    #     save_processed_df_to_csv(
    #         triples, f"TC_{self.case_name}_{set_name}", f"{set_name}/{self.case_name}"
    #     )
    #     save_processed_var_to_pickle(
    #         triples_list, f'TC_{self.case_name}_{set_name}_triples', f"{set_name}/{self.case_name}"
    #     )
    #     save_processed_var_to_pickle(
    #         labels, f'TC_{self.case_name}_{set_name}_labels', f"{set_name}/{self.case_name}"
    #     )

    def convert_triples_to_stroma_format(self) -> None:

        only_pair = self.test_triples[['head','tail']]
        list_of_strings = only_pair.apply(lambda x: ' :: '.join(x.astype(str)), axis=1).tolist()

        with open(f'../data/processed/test/default/{self.case_name}/STROMA_test_{self.case_name}.txt', 'w') as file:
            for line in list_of_strings:
                file.write(line + '\n')

class ESCOCNLMappingPreprocessor(MappingPreprocessor):
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv("ESCO-CNL_engels_compleet", encoding='cp1252')
        self.case_name = "ESCO-CNL"

    def process_raw_mapping(self):
        rename_dict = {
            "Classification_2_PrefLabel": "tail",
            'Classification_1_PrefLabel': 'head',
            'Mapping_relation': 'relation'
        }

        self.mapping = (
            self.mapping[['Classification_2_PrefLabel', 'Classification_1_PrefLabel', 'Mapping_relation']]
            .rename(columns=rename_dict)
            .dropna()
            .loc[lambda df: ~(df == 'nan').any(axis=1)]
        )

        self.mapping['relation'] = self.mapping['relation'].str[5:]


    # def process_raw_mapping(self) -> pd.DataFrame:
    #     rename_dict = {
    #         "Classification_2_PrefLabel2": "tail",
    #         'preferredLabel': 'head',
    #         'Mapping_relation': 'relation'
    #     }

    #     self.mapping = (
    #         self.mapping[['Classification_2_PrefLabel2', 'preferredLabel', 'Mapping_relation']]
    #         .rename(columns=rename_dict)
    #         .dropna()
    #         .loc[lambda df: ~(df == 'nan').any(axis=1)]
    #     )

    #     self.mapping['relation'] = self.mapping['relation'].str[5:]

    def remove_closeMatch_rows(self):
        self.mapping = self.mapping[self.mapping['relation'] != 'closeMatch']

class ESCOONETMappingPreprocessor(MappingPreprocessor):
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv('ESCO-ONET')
        self.case_name = "ESCO-ONET"

    def process_raw_mapping(self):
        self.mapping = self.mapping[["O*NET Title", "ESCO or ISCO Title","Type of Match"]]
        self.mapping.columns = ['tail','head','relation']


class handcrafted_MappingPreprocessor(MappingPreprocessor):
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv("handcrafted")
        self.case_name = "handcrafted"


class UI_MappingPreprocessor(MappingPreprocessor):
    def __init__(self):
        super().__init__()
        self.mapping = read_processed_csv("UI", "triples")
        
        # need to add relations for TC to work, also need to have all three bc of label ints in eval
        self.mapping['relation'] = "exactMatch"
        self.mapping['relation'][0] = "narrowMatch"
        self.mapping['relation'][1] = "broadMatch"

        self.case_name = "UI"


class ESCOBLMMappingPreprocessor(MappingPreprocessor):
    def __init__(self):
        super().__init__()
        self.mapping = read_raw_csv("ESCO-BLM_mapping")
        self.case_name = "ESCO-BLM"

    def process_raw_mapping(self) -> pd.DataFrame:
        rename_dict = {
            'Classification 1 PrefLabel': "head",
            'Classification 2 PrefLabel NL': "tail",
            'Mapping relation': "relation"
        }

        self.mapping = (
            self.mapping[
                ['Classification 1 PrefLabel', 'Classification 2 PrefLabel NL', 'Mapping relation']
            ]
            .rename(columns=rename_dict)
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.mapping['relation'] = self.mapping['relation'].str[5:]


    def remove_closeMatch_rows(self):
        self.mapping = self.mapping[self.mapping['relation'] != 'closeMatch']

    def add_inverse_subsumptions(self):
        inverse_triples = self.mapping[
            self.mapping['relation'].isin(['narrowMatch', 'broadMatch'])
        ].copy()
        
        inverse_triples[['head', 'tail']] = inverse_triples[['tail', 'head']]
        inverse_triples['relation'] = inverse_triples['relation'].replace(
            {'narrowMatch': 'broadMatch', 'broadMatch': 'narrowMatch'}
        )
        
        self.mapping = pd.concat([self.mapping, inverse_triples], ignore_index=True)


class RDFMappingPreprocessor(MappingPreprocessor):
    def __init__(self, dataset_id):
        super().__init__()
        self.tree = read_raw_rdf(f"stroma_{dataset_id}_reference", "tree")
        self.case_name = f"stroma_{dataset_id}_reference"

    def RDF_to_df(self):
        tree = self.tree

        namespaces = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                    'align': 'http://knowledgeweb.semanticweb.org/heterogeneity/alignment'}

        entity1_list = []
        entity2_list = []
        relation_list = []

        for cell in tree.xpath('//align:map/align:Cell', namespaces=namespaces):
            entity1 = cell.xpath('./align:entity1/@rdf:resource', namespaces=namespaces)
            entity2 = cell.xpath('./align:entity2/@rdf:resource', namespaces=namespaces)
            relation = cell.xpath('./align:relation/text()', namespaces=namespaces)
            
            entity1_list.append(entity1[0] if entity1 else None)
            entity2_list.append(entity2[0] if entity2 else None)
            relation_list.append(relation[0] if relation else None)

        df = pd.DataFrame({
            'head': entity1_list,
            'tail': entity2_list,
            'relation': relation_list,
        })
        
        df['relation'] = df['relation'].replace({'=':'exactMatch','>':'narrowMatch','<':'broadMatch'})

        # code for removing the links and underscores in each entity label
        for column in ['head', 'tail']:
            links = df[column]
            entities = [link.rsplit('/', 1)[-1] for link in links]
            entities = [x.replace("_", " ") for x in entities]
            df[column] = entities

        self.mapping = df

    def generate_inverse_triples(self):
        broad = self.mapping[self.mapping['relation'] == 'broadMatch']
        narrow = self.mapping[self.mapping['relation'] == 'narrowMatch']
        exact = self.mapping[self.mapping['relation'] == 'exactMatch']

        inverse_broad = pd.DataFrame()
        inverse_broad[['head','tail']] = broad[['tail','head']]
        inverse_broad['relation'] = 'narrowMatch'

        inverse_narrow = pd.DataFrame()
        inverse_narrow[['head','tail']] = narrow[['tail','head']]
        inverse_narrow['relation'] = 'broadMatch'

        inverse_exact = pd.DataFrame()
        inverse_exact[['head','relation','tail']] = exact[['tail','relation','head']]
        
        self.mapping = pd.concat([self.mapping, inverse_broad, inverse_narrow, inverse_exact])

class STROMAOntologyPreprocessor(OntologyPreprocessor):
    def __init__(self, case_name):
        super().__init__()
        self.graph = read_raw_rdf(f"stroma_{case_name}", "graph")
        self.case_name = f"stroma_{case_name}"

    def RDF_to_df(self):
        if self.case_name == "stroma_g7_source":
            return None

        standard_q = """
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>

                SELECT ?class ?label ?subClassOf
                WHERE {
                ?class rdf:type owl:Class .
                ?class rdfs:label ?label .
                ?class rdfs:subClassOf ?subClassOf .
                }
            """
        no_label_q = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?class ?subClassOf
            WHERE {
            ?class rdf:type owl:Class .
            ?class rdfs:subClassOf ?subClassOf .
            }
            """

        queries ={
            "g1":standard_q,
            "g2":standard_q,
            "g3":standard_q,
            "g4":standard_q,
            "g5":no_label_q,
            "g6":no_label_q,
            "g7":no_label_q
            }
        
        q_key = self.case_name[7:9]
        qres = self.graph.query(queries[q_key])

        df = pd.DataFrame(qres)

        if len(df.columns) > 2:
            df = df.drop(0, axis=1)

        df.columns = ['head','tail']
        df['relation'] = "broadMatch"

        #code for transforming the links into just names, worked worse for stroma #HERE
        for column in ['head', 'tail']:
            links = df[column]
            entities = [link.rsplit('/', 1)[-1] for link in links]
            entities = [x.replace("_", " ") for x in entities]
            df[column] = entities

        self.triples = df

    # def get_selfloop_exactMatches(self):
    #     selfloop_exactMatches = self.triples.copy()
    #     selfloop_exactMatches['tail'] = selfloop_exactMatches['head']
    #     selfloop_exactMatches['relation'] = 'exactMatch'
    #     selfloop_exactMatches = selfloop_exactMatches.drop_duplicates()

    def get_inverse_triples(self):
        inverse_df = pd.DataFrame()
        inverse_df['head'] = self.triples['tail']
        inverse_df['tail'] = self.triples['head']
        inverse_df['relation'] = 'narrowMatch'

        self.triples = pd.concat([self.triples, inverse_df])

    # def balance_relations(self) -> None:
    #     relation_counts = self.triples['relation'].value_counts()
    #     max_count = np.max(relation_counts)

    #     taser_trainset = read_processed_csv("taser_trainset_triples","triples")
    #     taser_trainset = taser_trainset.dropna()
    #     taser_trainset_equivalence = taser_trainset[taser_trainset['relation']=="exactMatch"]

    #     samples_equivalence = taser_trainset_equivalence.sample(n=max_count)

    #     self.triples = pd.concat([self.triples, samples_equivalence])
        
class AnchorSetsCreator(DataSetConverter):
    def __init__(self, case_name, anchor_percentage):
        super().__init__(None, case_name) 
        self.balanced_bool = False
        self.triples = read_processed_csv(f"RP_train_{case_name}_UB", f"train/default/{case_name}")

        test_case_name_dict = {"stroma_g2_source_stroma_g2_target":"stroma_g2_reference",
                        "stroma_g4_source_stroma_g4_target":"stroma_g4_reference",
                        "stroma_g5_source_stroma_g5_target":"stroma_g5_reference",
                        "stroma_g6_source_stroma_g6_target":"stroma_g6_reference",
                        "stroma_g7_source_stroma_g7_target":"stroma_g7_reference",
                        "CNL_ESCO":"ESCO-CNL",
                        "ESCO_ONET":"ESCO-ONET"}
        self.test_case_name = test_case_name_dict[case_name]

        self.test_triples = read_processed_csv(f"RP_test_{self.test_case_name}", f"test/default/{self.test_case_name}")
        self.anchor_percentage = anchor_percentage

    def generate_train_test_sets(self):
        test_triples = self.test_triples.copy()

        total_anchors = int(len(self.test_triples)*self.anchor_percentage)
        anchors = test_triples.sample(n=total_anchors)

        print(len(self.test_triples))
        print(total_anchors)

        # anchors.to_csv('anchors_ad_hoc.csv', index=False, sep=';')

        self.test_triples = self.test_triples.drop(anchors.index)

        print(len(self.test_triples))
        print(self.test_triples['relation'].value_counts())
        print(len(anchors))
        print(anchors['relation'].value_counts())

        self.triples = pd.concat([self.triples, anchors])

    def convert_triples_to_stroma_format(self) -> None:

        only_pair = self.test_triples[['head','tail']]
        list_of_strings = only_pair.apply(lambda x: ' :: '.join(x.astype(str)), axis=1).tolist()

        with open(f'../data/processed/test/anchors_{int(self.anchor_percentage*100)}/{self.test_case_name}/STROMA_test_{self.test_case_name}.txt', 'w') as file:
            for line in list_of_strings:
                file.write(line + '\n')

class MatcherTrainConverter(DataSetConverter):
    def __init__(self, case_name):
        super().__init__("train", case_name) 
        self.triples = pd.DataFrame(columns=['head', 'relation', 'tail'])
        ontology_list = sorted(case_name.split(","))
        self.case_name = sort_words_str(case_name)
        print(self.case_name)
        for ontology in ontology_list:
            self.triples = pd.concat(
                [self.triples, read_processed_csv(f"{ontology}_triples", "triples")],
                ignore_index=True
            )

    def from_relations_to_label(self):
        self.triples['label'] = 1

    def add_negatives(self):
        shifted_triples = self.triples.copy()
        n=300
        bottom_part_triples = shifted_triples['tail'][-n:]
        top_part_triples = shifted_triples['tail'][:-n]
        shifted_triples['tail'] = pd.concat([bottom_part_triples, top_part_triples]).reset_index(drop=True)
        shifted_triples['label'] = 0

        self.triples = pd.concat([self.triples, shifted_triples])

    def add_test_triples(self):
        data_with_anchors = read_processed_csv(f"RP_train_{self.case_name}_matcher_UB",f"train/matcher/{self.case_name}_matcher")
        combined = pd.concat([self.triples, data_with_anchors])
        combined_without_duplicates = combined.drop_duplicates(subset=['head','relation','tail'], keep='first')

        result = combined_without_duplicates.merge(self.triples, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
        result = result.drop(columns=['_merge'])

        result['label'] = 1

        train_data_with_anchors = pd.concat([result, self.triples])

        self.triples = train_data_with_anchors
        

    def drop_relation_column(self):
        self.triples = self.triples.drop('relation', axis=1)

    def save_to_pairs_matcher(self):
        pairs_list = list(
            self.triples[['head', 'tail']].itertuples(index=False, name=None)
        )
        labels_list = list(self.triples['label'])

        save_processed_df_to_csv(
            self.triples, f"RP_train_{self.case_name}_matcher_UB", f"train/matcher/{self.case_name}_matcher")
        save_processed_var_to_pickle(
            pairs_list, f'RP_train_pairs_{self.case_name}_matcher_UB', f"train/matcher/{self.case_name}_matcher")
        save_processed_var_to_pickle(
            labels_list, f'RP_train_labels_{self.case_name}_matcher_UB', f"train/matcher/{self.case_name}_matcher")
        
class MatcherTestConverter(DataSetConverter):
    def __init__(self, case_name):
        super().__init__("test", case_name) 
        self.test_triples = read_processed_csv(f"RP_test_{case_name}", f"test/matcher/{case_name}")
    
    def add_labels(self):
        self.test_triples['label'] = 1

    def get_all_possible_matches(self):
        if "ONET" in self.case_name:
            possible_occupations = read_raw_csv("ONET")["O*NET-SOC 2019 Title"]
        elif "CNL" in self.case_name:
            possible_occupations = read_raw_csv("CNL_engels")["OMSCHRIJVING_BEROEP"]

        test_occupations = self.test_triples['head']

        combinations = itertools.product(test_occupations, possible_occupations)
        df_possibilities = pd.DataFrame(combinations, columns=['head', 'tail'])
        df_possibilities['label'] = 0

        df_possibilities = pd.concat([self.test_triples, df_possibilities]).drop_duplicates(['head','tail','relation'], keep='first')

        print(df_possibilities)

        self.test_triples = df_possibilities

    def remove_relations_column(self):
        self.test_triples = self.test_triples.drop('relation', axis=1)

    def save_to_pairs_matcher(self):
        pairs_list = list(
            self.test_triples[['head', 'tail']].itertuples(index=False, name=None)
        )
        labels_list = list(self.test_triples['label'])

        save_processed_df_to_csv(
            self.test_triples, f"RP_test_{self.case_name}_matcher", f"test/matcher/{self.case_name}_matcher")
        save_processed_var_to_pickle(
            pairs_list, f'RP_test_pairs_{self.case_name}_matcher', f"test/matcher/{self.case_name}_matcher")
        save_processed_var_to_pickle(
            labels_list, f'RP_test_labels_{self.case_name}_matcher', f"test/matcher/{self.case_name}_matcher")
        


def main(args):
    if args.action == "CNL_ontology_to_triples":
        run_cnl_ontology()
    elif args.action == "ESCO_ontology_to_triples":
        run_esco_ontology()
    elif args.action == "ONET_ontology_to_triples":
        run_onet_ontology()
    elif args.action == "ESCO-CNL_mapping_to_triples":
        run_esco_cnl_mapping()
    elif args.action == "ESCO-ONET_mapping_to_triples":
        run_esco_onet_mapping()
    elif args.action == 'ESCO-BLM_mapping_to_triples':
        run_esco_blm_mapping()
    elif args.action == "handcrafted_to_triples":
        run_handcrafted_mapping()
    elif args.action == "UI_to_triples":
        run_ui_mapping()
    elif args.action == "stroma_ontologies_to_triples":
        run_stromas_ontologies()
    elif args.action == "stroma_mappings_to_triples":
        run_stromas_mapping()
    elif args.action == "stroma_external_data_sets_to_triples":
        run_stroma_external()
    elif args.action == "taser_trainset":
        run_taser_trainset()
    elif args.action == "WN18RR_to_triples":
        run_wn18rr()
    elif args.action == "FB15k_to_triples":
        run_fb15k()
    elif args.action == 'train_set_converter':
        run_train_set_converter(args)
    elif args.action == "test_set_converter":
        run_test_set_converter(args)
    elif args.action == "anchor_creator":
        run_anchor_creator(args)
    elif args.action == "train_set_converter_matcher":
        run_train_set_converter_matcher(args)
    elif args.action == "test_set_converter_matcher":
        run_test_set_converter_matcher(args)
    else:
        raise ValueError(f"Unknown case: {args.action}. Try again.")

def run_cnl_ontology():
    preprocessor = CNLOntologyPreprocessor("CNL")
    preprocessor.preprocess_raw_ontology()
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.remove_plural_subsumptions()
    preprocessor.generate_selfloop_exactMatch_triples()
    preprocessor.remove_duplicate_pairs()
    preprocessor.save_triples()

def run_esco_ontology():
    preprocessor = ESCOOntologyPreprocessor("ESCO")
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.generate_exactMatch_ESCO()
    preprocessor.generate_selfloop_exactMatch_triples()
    preprocessor.remove_duplicate_pairs()
    preprocessor.save_triples()

def run_onet_ontology():
    preprocessor = ONETOntologyPreprocessor("ONET")
    preprocessor.read_SOC()
    preprocessor.read_ONET()
    preprocessor.convert_to_graph()
    preprocessor.generate_subsumption_triples()
    preprocessor.generate_selfloop_exactMatch_triples()
    preprocessor.remove_duplicate_pairs()
    preprocessor.fix_inconsistencies()
    preprocessor.save_triples()

def run_esco_cnl_mapping():
    preprocessor = ESCOCNLMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    preprocessor.remove_closeMatch_rows()
    preprocessor.save_mapping_triples()

def run_esco_onet_mapping():
    preprocessor = ESCOONETMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    # preprocessor.remove_closeMatch_rows()
    preprocessor.save_mapping_triples()

def run_esco_blm_mapping():
    preprocessor = ESCOBLMMappingPreprocessor()
    run_common_mapping_steps(preprocessor)
    #preprocessor.add_inverse_subsumptions()
    preprocessor.remove_closeMatch_rows()
    preprocessor.save_mapping_triples()

def run_handcrafted_mapping():
    preprocessor = handcrafted_MappingPreprocessor()
    preprocessor.save_mapping_triples()

def run_ui_mapping():
    # return NotImplementedError
    preprocessor = UI_MappingPreprocessor()
    preprocessor.save_mapping_triples()

def run_common_mapping_steps(preprocessor):
    preprocessor.process_raw_mapping()
    preprocessor.drop_duplicate_rows()

def run_stromas_ontologies():
    dataset_ids = ['g2', 'g4', 'g5', 'g6', 'g7'] #'g3',
    files = ['source','target']
    for dataset_id in dataset_ids:
        for file in files:
            # since g7 source is empty, we skip it
            if (dataset_id == 'g7') & (file == 'source'):
                continue
            case_name = dataset_id + "_" + file
            preprocessor = STROMAOntologyPreprocessor(case_name)
            preprocessor.RDF_to_df()
            preprocessor.get_inverse_triples()
            preprocessor.generate_selfloop_exactMatch_triples()
            # preprocessor.balance_relations()
            preprocessor.save_triples()

def run_stromas_mapping():
    dataset_ids = ['g2', 'g4', 'g5', 'g6', 'g7']
    for dataset_id in dataset_ids:
        preprocessor = RDFMappingPreprocessor(dataset_id)
        preprocessor.RDF_to_df()
        #preprocessor.generate_inverse_triples()
        preprocessor.drop_duplicate_rows()
        preprocessor.save_mapping_triples()

def run_stroma_external():
    dataset_names = ['dbpedia', 'schema_org', 'wordnet', 'wikidata']
    for dataset_name in dataset_names:
        preprocessor = StromaExternalDataPreprocessor(dataset_name)
        preprocessor.convert_ints_to_labels()
        preprocessor.save_triples()

def run_taser_trainset():
    preprocessor = TaSeRPreprocessor("taser_trainset")
    preprocessor.save_triples()

def run_wn18rr():
    subset_names = ['WN18RR_train', 'WN18RR_test', 'WN18RR_valid']
    for subset_name in subset_names:
        preprocessor = WN18RRDataPreprocessor(subset_name)
        preprocessor.save_triples()

def run_fb15k():
    subset_names = ['FB15k_train', 'FB15k_test']
    for subset_name in subset_names:
        preprocessor = FB15kDataPreprocessor(subset_name)
        preprocessor.save_triples()

def run_train_set_converter(args):
    check_converter_args(args)
    preprocessor = TrainSetConverter(args.data_sets)

    preprocessor.drop_rows_with_na("train")
    
    preprocessor.drop_duplicate_rows("train")

    ### specific for test on 6-12 with exactMatches from TaSeR dataset
    preprocessor.balance_relations()
    # preprocessor.print_triples_statistics()

    if args.task_type == "RP":
        #preprocessor.shuffle_triples_random()
        #preprocessor.split_into_train_and_val()
        #preprocessor.save_to_pairs_RP("validation", "default")
        if args.balanced == "T":
            preprocessor.undersample_classes()
        #preprocessor.shuffle_triples_random()
        preprocessor.save_to_pairs_RP("train", "default")

    elif args.task_type == "TC":
        # preprocessor.split_into_train_and_val()
        # preprocessor.add_pos_label_to_df("validation")
        # preprocessor.generate_all_negatives("validation")
        # preprocessor.save_to_triples_TC("validation", "default")
        if args.balanced == "T":
            preprocessor.undersample_classes()
        preprocessor.add_pos_label_to_df("train")
        preprocessor.generate_all_negatives("train")
        preprocessor.shuffle_triples_grouped()
        preprocessor.save_to_triples_TC("train", "default")
    else:
        sys.exit(f"Unknown task type: {args.task_type}. Try again")

    preprocessor.print_triples_statistics()

def run_test_set_converter(args):
    check_converter_args(args)
    preprocessor = TestSetConverter(args.data_sets)
    preprocessor.drop_rows_with_na("test")
    preprocessor.drop_duplicate_rows("test")
    if args.task_type == "RP":
        preprocessor.save_to_pairs_RP("test", "default")
        preprocessor.convert_triples_to_stroma_format()
    elif args.task_type == "TC":
        preprocessor.add_pos_label_to_df("test")
        preprocessor.generate_all_negatives("test")
        preprocessor.save_to_triples_TC("test", "default")
    else:
        sys.exit(f"Unknown task type: {args.task_type}. Try again")

def run_anchor_creator(args):
    percentages = [0.80]
    set_names = ['train','test']
    for percentage in percentages:
        preprocessor = AnchorSetsCreator(args.data_sets, percentage)
        preprocessor.generate_train_test_sets()
        #preprocessor.add_inverse_subsumptions()
        preprocessor.drop_duplicate_rows("train")
        preprocessor.shuffle_triples_random()

        for set_name in set_names:
            readable_percentage = "anchors_" + str(int(percentage*100))
            preprocessor.save_to_pairs_RP(set_name, readable_percentage)
            preprocessor.convert_triples_to_stroma_format()
            preprocessor.add_pos_label_to_df(set_name)
            preprocessor.generate_all_negatives(set_name)
            preprocessor.save_to_triples_TC(set_name, readable_percentage)

def run_train_set_converter_matcher(args):
    check_converter_args(args)
    preprocessor = MatcherTrainConverter(args.data_sets)
    preprocessor.from_relations_to_label()
    preprocessor.add_negatives()
    preprocessor.add_test_triples()
    preprocessor.drop_relation_column()
    preprocessor.save_to_pairs_matcher()

def run_test_set_converter_matcher(args):
    check_converter_args(args)
    preprocessor = MatcherTestConverter(args.data_sets)
    preprocessor.add_labels()
    preprocessor.get_all_possible_matches()
    preprocessor.remove_relations_column()
    preprocessor.save_to_pairs_matcher()


def check_converter_args(args):
    if args.action == "train_set_converter":    
        if (args.data_sets is None) or (args.task_type is None) or (args.task_type == "RP" and args.balanced is None):
            sys.exit("Missing arguments for converter, check if you need to add data_sets, task_type or balanced")
    elif args.action == "test_set_converter":
        if (args.data_sets is None) or (args.task_type is None):
            sys.exit("Missing arguments for converter, make sure you have specified data_sets and task_type")
    elif (args.action == "train_set_converter_matcher") or (args.action == "test_set_converter_matcher"):
        if (args.data_sets is None):
            sys.exit("Missing arguments for matcher converter, make sure you have specified data_sets")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run data preparations, there are various options.")
    parser.add_argument("--action", required=True, type=str, help="action to be completed")
    parser.add_argument("--data_sets", required=False, type=str, help="datasets to be converted")
    parser.add_argument("--task_type", required=False, type=str, help="choose the task type")
    parser.add_argument("--balanced", required=False, type=str, help="choose whether to balance the classes")
    args = parser.parse_args()
    main(args)