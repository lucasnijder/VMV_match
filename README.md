# Advancing Ontology Alignment in the Labor Market: Combining Large Language Models with Domain Knowledge
Last update: 17-2-2024

NOTE: this branch contains the ONTOLOGY MATCHER. For the mapping refiner I refer to the branch `cleaned_up`

## Overview
This repository contains the code for ontology matching. It requires a logged-in Weights & Biases account to run. 

The current implementation takes two ontologies and returns a set of matches. 

NOTE: the ESCO-CompetentNL use case is not included as CompetentNL is not publicly available yet. 

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data_preparation)
3. [New use cases](#new-use-cases-quirine-maaike)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Running SpaCy](#running-spacy)
7. [Running GPT-4 on top of Match-BERT or SpaCy](#running-gpt-4-on-top-of-match-bert-or-spacy)
---

## Installation
```python
pip install -r requirements.txt
```

---

## Preparing Test Cases
To prepare data sets for specific test cases, the main.py script can be used. With one command all steps for data preprocessing are executed. 

For preparing the ESCO-O*NET use case:
```
python3 main.py --exp data_ESCO-ONET
```


## Individual Data Set Preparation
To prepare individual data sets, the data_preparation.py script can be used directly. The raw datasets are located in the `./data/raw` folder. To prepare raw data run the command. The preprocessing is performed in two steps. The first step proccesses the raw data files  into triples in csv format and saves them in the `./data/processed/triples` folder. The second step processes the triple csvs into pkl files which can be loaded in the PyTorch DataSet objects. These are saved in the `./data/processed/train/matcher` folder for train sets and `./data/processed/test/matcher` for test sets. 

```
python3 data_preparation.py --action XXX
```
For example:
```
python3 data_preparation.py --action ESCO_ontology_to_triples
```

The XXX can be replaced by either raw to triple commands:

`ONET_ontology_to_triples`, convert raw O*NET file to triples scv
`ESCO_ontology_to_triples`, convert raw ESCO file to triples csv

`ESCO-ONET_mapping_to_triples`, convert raw ESCO-O*NET mapping to triples csv

Or by triples to train and test format commands:
- `train_set_converter_matcher`, convert train triples into the format required for training (.pkl)
- `test_set_converter_matcher`, convert test triples into the format reqruired for testing (.pkl)
For both commands you also need to specify `data_sets` to indicate the use case. 

The train or test set is saved as two pickle (.pkl) files and a csv. The first pickle file contains the pairs/triples, the second pickle file contains the labels and the csv file contains pairs/triples and the labels for an inspectable overview. They are saved in either `./data/processed/train/matcher` or `./data/processed/test/matcher` and are named after the task and dataset. So for example: 
- `TC_ESCO-ONET_test_labels.pkl`
- `TC_ESCO-ONET_test_triples.pkl`
- `TC_ESCO-ONET_test.csv`
---


## New use cases (@quirine, @maaike)
If you want to train the refiner on a new use case do the following:
1. convert the two ontologies into a csv in triple format with columns `head`,`relation`,`tail`. Where the relations are exclusively `broadMatch`, `narrowMatch` and `exactMatch`. 
2. place this csv in `./data/processed/triples` and name it ONTOLOGY1_triples.csv and ONTOLOGY2_triples.csv. Replace ONTOLOGY1 & 2 with the names of your ontologies. 
3. run `data_preparation.py --action train_set_converter_matcher --data_set ONTOLOGY1,ONTOLOGY2 --task_type RP --balanced F`
4. now you can train the model using the training instructions

To evaluate an unseen test set, do the following:
1. convert your existing mapping with pairs you want to know the relation between into a csv with columns `head` and `tail`
2. add a new column called `relation` with a random string, e.g. 'placeholder'
3. place this csv in `./data/processed/triples` and name it ONTOLOGY1-ONTOLOGY2_triples.csv. Replace the ONTOLOGY1 & 2 with the names of your ontologies. 
3. run `data_preparation.py --action test_set_converter_matcher --data_set ONTOLOGY1,ONTOLOGY2 --task_type RP`
4. now you can use the previously trained model to predict the relations of the pairs in the test set using the evaluation instructions. Note that the evalation report will be printed, but you can ignore this. 

## Training Match-BERT
To train the model, run the following in the terminal. This command can also be used in the main.py file in the run_training.py function. There you can adjust all variables and run it using `python3 main.py --exp train`. 

```python
python3 train.py --model_variant XXX --classifier_variant XXX --task_type XXX --train_set_names XXX,YYY,ZZZ --train_set_balanced XXX --anchors XXX --num_labels XXX --lr XXX --epochs XXX --batch_size XXX --parallelize XXX --balance_class_weights XXX
```
For example:
```
python train.py ---model_variant bert-base-uncased --classifier_variant default --task_type RP --train_set_names ESCO,ONET_matcher --train_set_balanced F --anchors matcher --num_labels 2 --lr 0.00001 --epochs 4 --batch_size 64 --parallelize F --balance_class_weights T
```

The variables are:
- `model_variant`: which model to use, use bert-base-uncased
- `classifier_variant`: which classifier to use, use default
- `task_type`: RP or TC. Use RP.
- `train_set_names`: name of the train set to be used. The name exists of the included datasets in alphabetical order divided by a comma and with "_matcher" behind it. E.g. if you prepared a train set with ESCO and ONET triples, you enter ESCO,ONET_matcher. 
- `train_set_balanced`: whether you want to use a train set of which the classes have been balanced: T/F. Default is F. 
- `anchors`: Set as: matcher
- `num_labels`: represents the number of classes the model has to predict. For ontology matching this is 2 (True and False)
- `lr`: the learning rate (we use 0.00001)
- `epochs`: the number of epochs (we use 3)
- `batch_size`: the batch size (we use 64)
- `parallelize`: whether the training calculations should be paralellized over multiple GPUs (T/F). Default is F.
- `balance_class_weights`: whether you want to add class weights to the training in order to take class imbalances into account (T/F). Default is T

Each model is given a name plus three integers, e.g. `holly136`. The training progress of the model can be found on Wandb.ai. The model is saved LOCALLY in the `./models` folder, e.g. for `holly136`: `./models/distilbert-base-uncased/holly136_weights.pth`

---

## Evaluating Match-BERT

To evaluate the model, run:

```python
python3 evaluate.py --model_variant XXX --classifier_variant XXX --eval_type XXX --dataset_name XXX --anchors XXX --task_type XXX --model_name XXX --num_labels XXX --taser_default_model XXX
```

For example:
```
python3 evaluate.py --model_variant bert-base-uncased --classifier_variant default --eval_type test --dataset_name ESCO-ONET_matcher --anchors matcher --task_type RP --model_name philip475 --num_labels 2 --taser_default_model F
```

IMPORTANT: `model_variant`, `classifier_variant`, `task_type` and `num_labels` need to match the settings used during training.


The variables are:
- `model_variant`: the base model of the trained model, e.g. bert-base-uncased
- `classifier_variant`: set to: default 
- `eval_type`: which dataset type to use for evaluation, always use: test
- `dataset_name`: which dataset to use for evaluation. Data needs to be converted into test set using `test_set_converter_matcher` in `data_preparation.py` first. 
- `anchors`: set to: matcher
- `task_type`: which task type to use (RP or TC). Always use RP.
- `model_name`: the name given to the model, e.g. philip475.
- `num_labels`: the number of classes the model can predict. Use 2. 
- `taser_default_model`: whether the TaSeR base model was used, if unknown use F. Always use F. 

## Running SpaCy
Use the `calculate_spacy_sim.py` to run the SpaCy method.

Example:
```
python3 calculate_spacy_sim.py --test_set ESCO-ONET
```

## Running GPT-4 on top of Match-BERT or SpaCy
To run this method you first need to run the underlying candidate generation model, Match-BERT or SpaCy. 

Then you can run the GPT-4 method by first entering the OpenAI API key in the `GPT-4_method.py` file and calling the script using for instance:
```
python3 GPT-4_method.py --model spacy_top5 --test_set ESCO-ONET
```