import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler, Trainer, TrainingArguments, BertModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.special import softmax
from typing import List, Tuple, Dict
from utils import *
import wandb
from collections import namedtuple

torch.cuda.empty_cache()

class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length, relations=None, labels=None):
        self.sentences = sentences
        self.labels = labels
        self.relations = relations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def encode(self, sentence):
        raise NotImplementedError("This method should be overridden in subclass")

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoding = self.encode(sentence)

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
        
        if self.labels is not None:
            target = self.labels[idx]
            item['labels'] = torch.tensor(target, dtype=torch.long)

        if self.relations is not None:
            relation = self.relations[idx]
            item['relations'] = torch.tensor(relation, dtype=torch.long)

        return item


class SentencePairDataset(SentenceDataset):
    def encode(self, sentence_pair):
        head, tail = sentence_pair
        formatted_sentence = f"{head} [SEP] {tail}"
        return self.tokenizer.encode_plus(
            formatted_sentence, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )


class SentenceTripleDataset(SentenceDataset):
    def encode(self, sentence_triple):
        head, relation, tail = sentence_triple
        formatted_sentence = f"{head} [SEP] {relation} [SEP] {tail}"
        return self.tokenizer.encode_plus(
            # ' '.join(sentence_triple),
            formatted_sentence,
            add_special_tokens=True, 
            max_length=self.max_length, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt'
        )


class BertClassifier():
    def __init__(self, model_variant, num_labels, parallelize="F", model_to_be_loaded=None,
                 classifier_variant="default"):
        self.initialize_model(model_variant, num_labels, classifier_variant)
        self.initialize_device()
        print(model_to_be_loaded)
        if model_to_be_loaded:
            self.load_model_weights(model_to_be_loaded)
        self.set_parallelization(parallelize)

    def initialize_model(self, model_variant, num_labels, classifier_variant):
        self.model_variant = model_variant
        self.classifier_variant = classifier_variant
        self.num_labels = num_labels
        self.safe_model_variant = model_variant.replace("/", "-")

        # TaSeR uses a different tokenizer, so give the option to pick a different one
        tokenizer_options = {
            "default": lambda: BertTokenizer.from_pretrained(model_variant),
            "taser": lambda: AutoTokenizer.from_pretrained("dwsunimannheim/TaSeR"),
        }
        self.tokenizer = tokenizer_options[classifier_variant]()

        # for most classifiers the underlying model can be varied, but for taser it is always that one
        model_options = {
            "default": lambda: BertForSequenceClassification.from_pretrained(model_variant, num_labels=num_labels),
            "taser": lambda: AutoModelForSequenceClassification.from_pretrained("dwsunimannheim/TaSeR"),#, num_labels=num_labels),
        }
        self.model = model_options[classifier_variant]()

    def initialize_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def load_model_weights(self, model_to_be_loaded):
        weights_filename = f"../models/{self.safe_model_variant}/{model_to_be_loaded}_weights.pth"
        if os.path.exists(weights_filename):
            state_dict = torch.load(weights_filename)
            is_parallel = all([k.startswith('module.') for k in state_dict.keys()])
            if is_parallel:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            sys.exit(f"No saved model weights found at {weights_filename}")

    def set_parallelization(self, parallelize):
        if parallelize == "T":
            self.model = nn.DataParallel(self.model)


    def custom_weighted_cross_entropy(self, logits, labels, relationship_labels):
        # Compute the unweighted cross-entropy loss, without reduction
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        class_weights = torch.tensor(self.class_weights).to(self.device)
        sample_weights = class_weights[labels, relationship_labels]
        
        # Weight the loss
        weighted_loss = ce_loss * sample_weights
        
        # Average the weighted loss
        return weighted_loss.mean()

    def _train_epoch(self, data_loader):
        total_loss = 0
        total_steps = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            # because the relation&class weights are only for TC, exclude this for RP
            if self.task_type == "TC":
                relations = batch['relations'].to(self.device)

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}
            
            # this is alsof the the relation&class weights that are only for TC
            if self.task_type == "TC":
                batch['relations'] = relations

            self.optimizer.zero_grad()

            expected_args = self.model.forward.__code__.co_varnames
            filtered_batch = {k: v for k, v in batch.items() if k in expected_args}
            outputs = self.model(**filtered_batch)
            
            if is_nested(self.class_weights) is False or self.class_weights is None:
                loss_obj = CrossEntropyLoss()
                loss = loss_obj(outputs.logits, batch['labels'])
            elif is_nested(self.class_weights) is True:
                loss = self.custom_weighted_cross_entropy(outputs.logits, batch['labels'], batch['relations'])
            else:
                sys.exit("something is wrong with the weights")

            loss.backward()

            self.optimizer.step()
            #self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1

            if not HP_TUNING_BOOL:
                wandb.log({"step_loss": loss.item()})

        return total_loss / total_steps

    # def _get_loss(self, class_weights):
    #     print(class_weights)
    #     if is_nested(class_weights) is False:
    #         assert len(class_weights) == self.num_labels
    #         weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)
    #         return CrossEntropyLoss(weight=weight_tensor)
    #     elif class_weights is None:
    #         return CrossEntropyLoss(weight=None)
    #     elif is_nested(class_weights) is True:
    #         return weighted_cross_entropy_loss()
    #     else:
    #         sys.exit("something is wrong with the weights")


    
    def train(self, data, labels, lr, epochs, batch_size, relations, class_weights=None):
        self.model.train()
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.class_weights = class_weights

        data_loader = self.create_train_data_loader(data, labels, shuffle=True, relations=relations)

        # scheduler_steps = len(data_loader) * self.epochs
        # warmup_steps = 0  # int(scheduler_steps * 0.15)

        # self.scheduler = get_scheduler("linear", self.optimizer, warmup_steps, scheduler_steps)
        # self.loss = self._get_loss(class_weights)
        print(f"Model uses class weights {class_weights}")

        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch(data_loader)

            if not HP_TUNING_BOOL:
                wandb.log({"epoch_loss": epoch_loss})


class BertClassifier_relation_prediction(BertClassifier):
    def __init__(self, model_variant, num_labels, parallelize = "F",model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, model_to_be_loaded = model_to_be_loaded, classifier_variant=classifier_variant)
        self.task_type = "RP"

    def create_train_data_loader(self, sentence_pairs: List[Tuple[str, str]], labels: List[int], shuffle, relations):
        dataset = SentencePairDataset(sentence_pairs, self.tokenizer, BERT_MAX_SEQUENCE_LENGTH, labels=labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader
    
    def create_test_data_loader(self, sentence_pairs: List[Tuple[str, str]], shuffle):
        dataset = SentencePairDataset(sentence_pairs, self.tokenizer, max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def predict(self, sentence_pairs, label_dict: Dict[int, str], name_test_set, model_name, batch_size) -> float:
        self.model.eval()

        self.batch_size = batch_size

        data_loader = self.create_test_data_loader(sentence_pairs, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            count = 1
            total_count = len(data_loader)
            for batch in data_loader:
                print("started batch: ", count, "out of", total_count)
                count += 1
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                _, predictions = torch.max(outputs.logits.data, 1)
                all_predictions.extend(predictions.tolist())

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['prediction'] = all_predictions
        df_predictions = df_predictions.replace(label_dict)
        save_processed_df_to_csv(df_predictions, f"{self.task_type}_{model_name}_{name_test_set}_prediction","predictions")

        return all_predictions

    def evaluate(self, sentence_pairs, labels, label_dict: Dict[int, str], name_test_set, model_name, batch_size) -> float:
        self.model.eval()

        label_dict = {0:'0',1:'1'}

        predictions = self.predict(sentence_pairs, label_dict, name_test_set, model_name, batch_size)

        accuracy = accuracy_score(labels, predictions)
        print('Overall Accuracy: %d %%' % (100 * accuracy))

        target_names: List[str] = list(label_dict.values())
        cm: np.ndarray = confusion_matrix(labels, predictions)

        if len(target_names) != len(np.unique(labels)):
            print("Error probably caused by fact that one of the classes is not in the test set")

        #target_names = ['broadMatch','exactMatch','narrowMatch','noMatch']
        target_names: List[str] = [label_dict[i] for i in range(self.num_labels)]

        cm_df: pd.DataFrame = pd.DataFrame(cm, index=[f'Actual {label}' for label in target_names], 
                                        columns=[f'Predicted {label}' for label in target_names])
        print("Confusion Matrix:")
        print(cm_df)

        print("Classification Report:")

        print(classification_report(labels, predictions, target_names=target_names))

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['relation'] = labels
        df_predictions['prediction'] = predictions
        df_predictions = df_predictions.replace(label_dict)

        res_to_save = ['Overall Accuracy: %d %%' % (100 * accuracy), "Confusion Matrix:", cm_df, "Classification Report:", classification_report(labels, predictions, target_names=target_names)]
        save_printed_output_to_file(res_to_save, f"RP_{model_name}_{name_test_set}", "evaluation_results")
        print("saved here:", f"RP_{model_name}_{name_test_set}")

        return accuracy

class BertClassifier_triple_classification(BertClassifier):
    def __init__(self, model_variant, num_labels, parallelize = "F",model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, model_to_be_loaded = model_to_be_loaded, classifier_variant=classifier_variant)
        self.task_type = "TC"

    def create_train_data_loader(self, sentence_triples: List[Tuple[str, str, str]], labels: List[int], shuffle, relations):
        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer, max_length=BERT_MAX_SEQUENCE_LENGTH, labels=labels, relations=relations)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def create_test_data_loader(self, sentence_triples: List[Tuple[str, str, str]], shuffle):
        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer, max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def predict(self, sentence_triples: List[Tuple[str, str, str]], name_test_set, model_name, batch_size):
        self.model.eval()

        self.batch_size = batch_size
        data_loader = self.create_test_data_loader(sentence_triples, shuffle=False)

        all_logits = []
        all_predictions = []

        with torch.no_grad():
            for count, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, predictions = torch.max(logits.data, 1)
                all_logits.extend(logits.tolist())
                all_predictions.extend(predictions.tolist())

        df_predictions = pd.DataFrame(sentence_triples, columns=["head", "relation", "tail"])
        df_predictions['prediction'] = all_predictions
        probits = [softmax(logits) for logits in all_logits]
        df_predictions['probits'] = [[round(num, 3) for num in sublist] for sublist in probits]

        pivot_df = df_predictions.pivot(index=['head', 'tail'], columns='relation', values='probits')

        print(pivot_df)

        probs_df = pivot_df.apply(lambda x: x.map(lambda y: y[1]), axis=1)
        final_df = probs_df.reset_index()

        if "WN18RR" in name_test_set:
            relation_list = WN18RR_RELATIONS
        elif ("CNL" in name_test_set) or ("ESCO" in name_test_set) or ("handcrafted" in name_test_set):
            relation_list = ESCO_CNL_RELATIONS
        elif "stroma" in name_test_set:
            relation_list = STROMA_DATA_RELATIONS
        else:
            print(name_test_set)
            sys.exit("Could not select relation list. Check function 'predict' in BERT TC classifier if still up to date with the used datasets")

        final_df['highest_prob'] = final_df[relation_list].idxmax(axis=1)
        
        save_processed_df_to_csv(final_df, f"TC_{model_name}_{name_test_set}_prediction", "predictions")

        return final_df

    def evaluate(self, test_df, sentence_triples: List[Tuple[str, str, str]], labels: List[int], name_test_set, model_name, batch_size):
        self.model.eval()
        
        final_df = self.predict(sentence_triples, name_test_set, model_name, batch_size)

        test_df_true = test_df[test_df['label'] == 1]
        merged_df = pd.merge(final_df, test_df_true[['head', 'tail', 'relation']], on=['head', 'tail'], how='left')
        merged_df.rename(columns={'relation': 'actual_relation'}, inplace=True)

        accuracy = accuracy_score(merged_df['actual_relation'], merged_df['highest_prob'])
        print('Overall Accuracy: %d %%' % (100 * accuracy))

        if "WN18RR" in name_test_set:
            label_dict = WN18RR_LABEL_DICT
        elif ("CNL" in name_test_set) or ("ESCO" in name_test_set) or ("handcrafted" in name_test_set):
            label_dict = ESCO_CNL_LABEL_DICT
        elif "stroma" in name_test_set:
            label_dict = STROMA_DATA_LABEL_DICT
        else:
            sys.exit("Could not select label dictionary. Check function 'evaluate' in BERT TC classifier if still up to date with the used datasets")

        target_names = list(label_dict.values())
        cm: np.ndarray = confusion_matrix(merged_df['actual_relation'], merged_df['highest_prob'])
        cm_df: pd.DataFrame = pd.DataFrame(cm, index=[f'Actual {label}' for label in target_names],
                                        columns=[f'Predicted {label}' for label in target_names])
        print("Confusion Matrix:")
        print(cm_df)
        print("Classification Report:")
        target_names = [label_dict[i] for i in range(len(label_dict))]
        print(classification_report(merged_df['actual_relation'], merged_df['highest_prob'], target_names=target_names))
        
        res_to_save = ['Overall Accuracy: %d %%' % (100 * accuracy), "Confusion Matrix:", cm_df, "Classification Report:", classification_report(merged_df['actual_relation'], merged_df['highest_prob'], target_names=target_names)]
        save_printed_output_to_file(res_to_save, f"TC_{model_name}_{name_test_set}", "evaluation_results")

class BERTClassifierNewRP(BertClassifier):
    def __init__(self, model_variant, num_labels, parallelize = "F",model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, model_to_be_loaded = model_to_be_loaded, classifier_variant=classifier_variant)
        self.task_type = "NewRP"

    def create_train_data_loader(self, sentence_pairs: List[Tuple[str, str]], labels: List[int], shuffle):
        dataset = SentencePairDataset(sentence_pairs, self.tokenizer, BERT_MAX_SEQUENCE_LENGTH, labels=labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def create_test_data_loader(self, sentence_pairs: List[Tuple[str, str]], shuffle):
        dataset = SentencePairDataset(sentence_pairs, self.tokenizer, max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def compute_loss(self, model, batch, device, class_weights=None):
        outputs = model(**batch)
        logits = outputs.logits
        n = logits.shape[0]

        labels = batch['labels']

        extended_labels = []
        extended_logits = logits.clone()
        weights = []
        for i in range(n):
            for j in range(self.num_labels):
                if j == labels[i]:
                    continue
                extended_labels.append(j)
                extended_logits = torch.cat((extended_logits, logits[i:i+1, :]), 0)
                if class_weights is not None:
                    weights.append(class_weights[labels[i]])
                else:
                    weights.append(1.0)       

        extended_labels = torch.tensor(extended_labels, device=device)
        if class_weights is not None:
            weights = torch.tensor(weights, device=device).unsqueeze(-1)

        positives = logits[torch.arange(n), labels].unsqueeze(-1).repeat(1, self.num_labels - 1).view(-1, 1)
        negatives = extended_logits[torch.arange(n, extended_logits.shape[0]), extended_labels].unsqueeze(-1)
        info_nce_loss = - F.logsigmoid(positives - negatives)
        # if class_weights is not None:
        info_nce_loss *= weights
        info_nce_loss = info_nce_loss.mean()

        if not HP_TUNING_BOOL:
            wandb.log({"step_loss": info_nce_loss.item()})

        return info_nce_loss

    def train(self, data, labels, lr, epochs, batch_size, relations, class_weights=None):
        self.model.train()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        scheduler_steps = len(self.create_train_data_loader(data, labels, shuffle=True)) * self.epochs
        warmup_steps = 0
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps
        )

        if class_weights is not None:
            assert len(class_weights) == self.num_labels, "class_weights length should match num_labels"
        else:
            class_weights = None

        print(f"Model uses class weights {class_weights}")
        
        if self.task_type == "TC":
            data_loader = self.create_train_data_loader(data, labels, shuffle=True, relations=relations)
        elif self.task_type == "RP" or self.task_type == "NewRP":
            data_loader = self.create_train_data_loader(data, labels, shuffle=True)
        else:
            sys.exit("something is wrong in the train function")

        for epoch in range(self.epochs):
            total_loss = 0
            total_steps = 0

            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                

                batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

                self.optimizer.zero_grad()

                loss_value = self.compute_loss(self.model, batch, self.device, class_weights=class_weights)
                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss_value.item()
                total_steps += 1
                
                if not HP_TUNING_BOOL:
                    wandb.log({"step_loss": loss_value.item()})
            
            epoch_loss = total_loss / total_steps

            if not HP_TUNING_BOOL:
                wandb.log({"epoch_loss": epoch_loss})

    def predict(self, sentence_pairs, label_dict: Dict[int, str], name_test_set, model_name, batch_size) -> float:
        self.model.eval()

        self.batch_size = batch_size

        data_loader = self.create_test_data_loader(sentence_pairs, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                _, predictions = torch.max(outputs.logits.data, 1)
                all_predictions.extend(predictions.tolist())

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['prediction'] = all_predictions
        df_predictions = df_predictions.replace(label_dict)
        save_processed_df_to_csv(df_predictions, f"{self.task_type}_{model_name}_{name_test_set}_prediction","predictions")

        return all_predictions

    def evaluate(self, sentence_pairs, labels, label_dict: Dict[int, str], name_test_set, model_name, batch_size) -> float:
        self.model.eval()

        predictions = self.predict(sentence_pairs, label_dict, name_test_set, model_name, batch_size)

        accuracy = accuracy_score(labels, predictions)
        print('Overall Accuracy: %d %%' % (100 * accuracy))

        target_names: List[str] = list(label_dict.values())
        cm: np.ndarray = confusion_matrix(labels, predictions)
        cm_df: pd.DataFrame = pd.DataFrame(cm, index=[f'Actual {label}' for label in target_names], 
                                        columns=[f'Predicted {label}' for label in target_names])
        print("Confusion Matrix:")
        print(cm_df)

        print("Classification Report:")
        target_names: List[str] = [label_dict[i] for i in range(self.num_labels)]
        print(classification_report(labels, predictions, target_names=target_names))

        if len(sentence_pairs[0]) == 3:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "relation", "tail"])
        elif len(sentence_pairs[0]) ==2:
            df_predictions = pd.DataFrame(sentence_pairs, columns=["head", "tail"])
        else:
            sys.exit('something went wrong in RP predict')

        df_predictions['relation'] = labels
        df_predictions['prediction'] = predictions
        df_predictions = df_predictions.replace(label_dict)

        res_to_save = ['Overall Accuracy: %d %%' % (100 * accuracy), "Confusion Matrix:", cm_df, "Classification Report:", classification_report(labels, predictions, target_names=target_names)]
        save_printed_output_to_file(res_to_save, f"RP_{model_name}_{name_test_set}", "evaluation_results")

        return accuracy


class BERTClassifierNewTC(BertClassifier):
    def __init__(self, model_variant, num_labels, parallelize = "F",model_to_be_loaded = None, classifier_variant='default'):
        super().__init__(model_variant, num_labels, parallelize=parallelize, model_to_be_loaded = model_to_be_loaded, classifier_variant=classifier_variant)
        self.task_type = "NewTC"

    def create_train_data_loader(self, sentence_triples: List[Tuple[str, str, str]], labels: List[int], shuffle):
        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer, BERT_MAX_SEQUENCE_LENGTH, labels)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    def create_test_data_loader(self, sentence_triples: List[Tuple[str, str, str]], shuffle):
        dataset = SentenceTripleDataset(sentence_triples, self.tokenizer, max_length=BERT_MAX_SEQUENCE_LENGTH)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return data_loader

    # def compute_loss(self, pos_batch, neg_batch, class_weights=None): # TODO: aanpassen zodat relatie vervangen wordt
    #     # Forward pass for positive samples
    #     pos_logits, pos_labels = self.model(
    #         input_ids=pos_batch['input_ids'].to(self.device),
    #         attention_mask=pos_batch['attention_mask'].to(self.device),
    #         labels=pos_batch['labels'].to(self.device)
    #     )

    #     neg_logits, neg_labels = self.model(
    #         input_ids=neg_batch['input_ids'].to(self.device),
    #         attention_mask=pos_batch['attention_mask'].to(self.device),
    #         labels=neg_batch['labels'].to_device
    #     )

    #     # Concatenate the logits and prepare labels
    #     logits = torch.cat([pos_logits, neg_logits], dim=0)
    #     labels = torch.cat([pos_labels, torch.zeros(len(neg_logits), dtype=torch.long)], dim=0).to(self.device)

    #     # Optionally use class weights
    #     if class_weights is not None:
    #         loss = F.cross_entropy(logits, labels, weight=class_weights.to(self.device))
    #     else:
    #         loss = F.cross_entropy(logits, labels)

    #     return loss
            
    def infonce_loss(self, anchor, positive, negatives):
        # Calculate the numerator (e^dot(anchor, positive))
        numerator = torch.exp(torch.sum(anchor * positive, dim=-1, keepdim=True))
        
        # Reshape negatives to have the same number of groups as the anchor and positive
        reshaped_negatives = negatives.view(anchor.shape[0], -1, anchor.shape[1])
        
        # Calculate the denominator term
        denominator_terms = torch.exp(torch.sum(anchor.unsqueeze(1) * reshaped_negatives, dim=-1))
        
        # Sum the denominator terms along dim=1 and add the numerator
        denominator = numerator + torch.sum(denominator_terms, dim=1, keepdim=True)
        
        # Calculate the InfoNCE loss
        loss = -torch.log(numerator / denominator)
        
        return torch.mean(loss)

    def train(self, data, labels, lr, epochs, batch_size, class_weights=None):
        self.model.train()
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        scheduler_steps = len(self.create_train_data_loader(data, labels, shuffle=True)) * self.epochs
        warmup_steps = 0
        self.scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_steps
        )

        if class_weights is not None:
            assert len(class_weights) == self.num_labels, "class_weights length should match num_labels"
        else:
            class_weights_tensor = None

        print(f"Model uses class weights {class_weights}")
        
        data_loader = self.create_train_data_loader(data, labels, shuffle=True)

        for epoch in range(self.epochs):
            total_loss = 0
            total_steps = 0

            for batch in data_loader:
                pos_input_ids = batch['input_ids'].to(self.device)
                pos_attention_mask = batch['attention_mask'].to(self.device)
                pos_labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()

                pos_batch = {
                    'input_ids': pos_input_ids,
                    'attention_mask': pos_attention_mask,
                    'labels': pos_labels
                }

                # pos_scores = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask)

                decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in pos_input_ids]

                def extract_relations(tokenized_texts):
                    results = []
                    for text in tokenized_texts:
                        parts = [part.strip() for part in text.split('[SEP]')]
                        head = parts[0].replace('[CLS]', '').strip()
                        relation = parts[1].strip()
                        tail = parts[2].strip()
                        results.append((head, relation, tail))
                    return results
                
                def alter_relation(triples):
                    altered_triples = []
                    options = ['exactMatch', 'broadMatch', 'narrowMatch']
                    
                    for triple in triples:
                        current_relation = triple[1]
                        other_relations = [opt for opt in options if opt != current_relation]
                        
                        for new_relation in other_relations:
                            new_triple = (triple[0], new_relation, triple[2])
                            altered_triples.append(new_triple)
                    
                    return altered_triples
                
                e = extract_relations(decoded_texts)
                negatives = alter_relation(e)
                            
                # Tokenize negatives using SentenceTripleDataset
                neg_dataset = SentenceTripleDataset(negatives, self.tokenizer, max_length=128)
                neg_input_ids, neg_attention_mask = [], []
                for idx in range(len(neg_dataset)):
                    item = neg_dataset[idx]
                    neg_input_ids.append(item['input_ids'])
                    neg_attention_mask.append(item['attention_mask'])

                neg_input_ids = torch.stack(neg_input_ids).to(self.device)
                neg_attention_mask = torch.stack(neg_attention_mask).to(self.device)

                # Forward pass for negatives
                pos_scores = self.model(input_ids=pos_input_ids, attention_mask=pos_attention_mask).logits
                neg_scores = self.model(input_ids=neg_input_ids, attention_mask=neg_attention_mask).logits

                # Compute InfoNCE Loss
                loss_value = self.infonce_loss(pos_scores, pos_scores, neg_scores)

                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss_value.item()
                total_steps += 1
                
                if not HP_TUNING_BOOL:
                    wandb.log({"step_loss": loss_value.item()})
            
            epoch_loss = total_loss / total_steps

            if not HP_TUNING_BOOL:
                wandb.log({"epoch_loss": epoch_loss})