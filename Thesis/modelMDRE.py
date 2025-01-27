from typing import Optional
import sys
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import sys
import mmsdk
import csv
import pandas as pd
from sklearn.model_selection import KFold

from collections import defaultdict
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from mmsdk import mmdatasdk as md
from sklearn.metrics import accuracy_score
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
from SingleEncoderModelText import SingleEncoderModelText
from SingleEncoderModelAudio import SingleEncoderModelAudio
from SEncoderMDRE import EncoderMDRE
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds import standard_train_fold, standard_valid_fold, standard_test_fold
from sklearn.metrics import precision_score, recall_score, f1_score


def initialize_sdk():
    if SDK_PATH is None:
        print("SDK path is not specified! Please specify first.")
        exit(0)
    sys.path.append(SDK_PATH)
    print(f"SDK path is set to {SDK_PATH}")

def setup_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    DATASETMD = md.cmu_mosi

    try:
        md.mmdataset(DATASETMD.highlevel, DATA_PATH)
    except RuntimeError:
        print("High-level features already downloaded.")

    try:
        md.mmdataset(DATASETMD.raw, DATA_PATH)
    except RuntimeError:
        print("Raw data already downloaded.")

    try:
        md.mmdataset(DATASETMD.labels, DATA_PATH)
    except RuntimeError:
        print("Labels already downloaded.")

    return DATASETMD

def load_features_and_save(DATASET):
    visual_field = 'CMU_MOSI_Visual_Facet_41'
    acoustic_field = 'CMU_MOSI_COVAREP'
    text_field = 'CMU_MOSI_TimestampedWords'
    wordvectors_field = 'CMU_MOSI_TimestampedWordVectors'

    recipe = {
        text_field: os.path.join(DATA_PATH, text_field) + '.csd',
        wordvectors_field: os.path.join(DATA_PATH, wordvectors_field) + '.csd',
        visual_field: os.path.join(DATA_PATH, visual_field) + '.csd',
        acoustic_field: os.path.join(DATA_PATH, acoustic_field) + '.csd'
    }

    print("Loading dataset features...")
    dataset = md.mmdataset(recipe)

    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features

    dataset.align(text_field, collapse_functions=[avg])
    print("text field aligned, len textfield: ", len(dataset[text_field].keys()))
    words = []
    for seq_key in dataset[text_field].keys():
        computational_sequence = dataset[text_field][seq_key]  # Access the sequence
        data = computational_sequence['features']  # Assuming the words are in 'features'
        words.extend(data.flatten())  # Flatten and extend into the words list

    # Print the total number of words
    total_words = len(words)
    print(f"Total number of words: {total_words}")

    # Save the words to a CSV file
    words_df = pd.DataFrame(words, columns=['Words'])
    words_df.to_csv('text_field.csv', index=False)
    print("Words saved to 'text_field.csv'")

    return dataset, visual_field, acoustic_field, text_field, wordvectors_field

def save_fields_to_csv_with_row_count():
    # Paths to the CSD files
    fields = {
        'CMU_MOSI_Visual_Facet_41': '/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_Visual_Facet_41.csd',
        'CMU_MOSI_COVAREP': '/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_COVAREP.csd',
        'CMU_MOSI_TimestampedWords': '/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWords.csd',
        'CMU_MOSI_TimestampedWordVectors': '/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd'
    }

    # Dictionary to store row counts
    row_counts = {}

    # Loop through each field and save its contents to a CSV file
    for field_name, path_to_csd in fields.items():
        output_csv = f"{field_name}.csv"  # Output CSV file name

        # Load the CSD file
        print(f"Loading {field_name} from {path_to_csd}")
        dataset = md.mmdataset({field_name: path_to_csd})

        row_count = 0  # Initialize row count
        # Open a CSV file to write the field's data
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Loop through all segments and write features
            for segment in dataset[field_name].keys():
                features = dataset[field_name][segment]['features']  # Extract features
                for feature in features:
                    writer.writerow(feature)  # Write each feature as a row in the CSV file
                    row_count += 1  # Increment row count

        # Save the row count for the field
        row_counts[field_name] = row_count
        print(f"Data for {field_name} has been saved to {output_csv} with {row_count} rows.")


    # Save the summary of row counts to a CSV file
    summary_csv = "row_counts_summary.csv"
    with open(summary_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Field Name", "Row Count"])  # Write header
        for field, count in row_counts.items():
            writer.writerow([field, count])  # Write each field and its row count

    print(f"\nSummary of row counts has been saved to {summary_csv}")


def align_labels(dataset):
    label_field = 'CMU_MOSI_Opinion_Labels'
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field) + '.csd'}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)
    
    return dataset, label_field


def split_data(DATASETMD):
    train_split = DATASETMD.standard_folds.standard_train_fold
    dev_split = DATASETMD.standard_folds.standard_valid_fold
    test_split = DATASETMD.standard_folds.standard_test_fold

    print(f"lengths: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}\n")
    print(train_split)
    print(dev_split)
    print(test_split)
    return train_split, dev_split, test_split


def preprocess_data_modified(dataset, 
                             visual_field, acoustic_field, text_field, wordvectors_field, label_field, 
                             train_split, dev_split, test_split):
    print("Preprocessing data modified...")

    word_to_ids = defaultdict(lambda: len(word_to_ids))
    print(f"len word2id: {len(word_to_ids)}")

    EPS = 1e-8

    pattern = re.compile('(.*)\[.*\]')

    train, dev, test = [], [], []
    
    num_drop = 0


    for segment in dataset[label_field].keys():
        #i assume this groups the video segments to represent one video
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        
        _words = dataset[text_field][segment]['features']  # Extract the words
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        _wordvectors = dataset[wordvectors_field][segment]['features']
        
        if not (_words.shape[0] == _visual.shape[0] == _acoustic.shape[0] == _wordvectors.shape[0]):
            num_drop += 1
            continue

        words, visual, acoustic, wordvectors = [], [], [], []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word_to_ids[word[0].decode('utf-8')])
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
                wordvectors.append(_wordvectors[i, :])            
        
        
        words = np.array(words)
        visual = np.array(visual)
        acoustic = np.array(acoustic)
        wordvectors = np.array(wordvectors)


        std_dev_visual = np.std(visual, axis=0, keepdims=True)
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + std_dev_visual))
        visual[:, std_dev_visual.flatten() == 0] = EPS  # Safeguard for zero standard deviation

        # Z-normalization for acoustic modality (across all acoustic features)
        acoustic_mean = np.nanmean(acoustic, axis=0, keepdims=True)
        std_dev_acoustic = np.nanstd(acoustic, axis=0, keepdims=True)
        std_dev_acoustic = np.nan_to_num(std_dev_acoustic)
        std_dev_acoustic[std_dev_acoustic == 0] = EPS  # Safeguard for zero standard deviation

        acoustic = np.nan_to_num((acoustic - acoustic_mean) / (EPS + std_dev_acoustic))

        # Z-normalization for word vectors
        wordvectors_mean = np.nanmean(wordvectors, axis=0, keepdims=True)
        std_dev_wordvectors = np.nanstd(wordvectors, axis=0, keepdims=True)
        std_dev_wordvectors = np.nan_to_num(std_dev_wordvectors)
        std_dev_wordvectors[std_dev_wordvectors == 0] = EPS  # Safeguard for zero standard deviation
        wordvectors = np.nan_to_num((wordvectors - wordvectors_mean) / (EPS + std_dev_wordvectors))

        # plot_hist2(wordvectors, acoustic)

        # Ensure no NaN or Inf values in the data
        if np.any(np.isnan(acoustic)) or np.any(np.isinf(acoustic)):
            print(f"Error in acoustic data for segment {vid}")
        if np.any(np.isnan(visual)) or np.any(np.isinf(visual)):
            print(f"Error in visual data for segment {vid}")
        if np.any(np.isnan(words)) or np.any(np.isinf(words)):
            print(f"Error in wordvectors data for segment {vid}")

        if vid in train_split:
            train.append(((words, visual, acoustic, wordvectors), label, segment))
        elif vid in dev_split:
            dev.append(((words, visual, acoustic, wordvectors), label, segment))
        elif vid in test_split:
            test.append(((words, visual, acoustic, wordvectors), label, segment))

    UNK = word_to_ids['<unk>']
    PAD = word_to_ids['<pad>']

    print(f"END len word_to_vector: {len(word_to_ids)}")
    print(f"Dropped {num_drop} inconsistent datapoints.")
    print(f"words: {words}")

    def return_unk():
        return UNK
    word_to_ids.default_factory = return_unk
    
    return train, dev, test, word_to_ids

from matplotlib import pyplot as plt


def collate_batch(batch, pad_value):
    """Collate function to handle variable-length sequences in a batch."""
    batch = sorted(batch, key=lambda x: len(x[0][0]), reverse=True)
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()

    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=pad_value, batch_first=True)
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first=True)

    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    # print("lengths:",lengths)
    return sentences, acoustic, labels, lengths


def train_model(model, train_loader, dev_loader, MAX_EPOCH=1000, patience=8, num_trials=3, grad_clip_value=1.0):
    """
    Trains the model using the given training and development data loaders.
    """
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    CUDA = torch.cuda.is_available()

    curr_patience = patience

    optimizer = model.create_optimizer(lr=0.001)
    print("Optimizer created")

    if CUDA:
        model.cuda()

    criterion = nn.MSELoss(reduction='sum')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step()

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []
    for e in range(MAX_EPOCH):
        model.train()
        train_iter = tqdm(train_loader)
        train_loss = 0.0

        for batch in train_iter:
            model.zero_grad()
            t, a, y, l = batch

            print(f"Batch t shape: {t.shape}")
            print(f"Batch a shape: {a.shape}")
            print(f"Batch y shape: {y.shape}")
            print(f"Batch l shape: {l.shape}")

            batch_size = t.size(0)
            print("batch_size:", batch_size)
            # print("l:", l)
            if CUDA:
                t = t.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()

            y_tilde = model(t, a, l)
            print(f"Model output y_tilde shape: {y_tilde.shape}")

            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()

            train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Training loss: {round(train_loss, 4)}")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch in dev_loader:
                model.zero_grad()
                t, a, y, l = batch

                print(f"Validation batch t shape: {t.shape}")
                print(f"Validation batch a shape: {a.shape}")
                print(f"Validation batch y shape: {y.shape}")
                print(f"Validation batch l shape: {l.shape}")

                if CUDA:
                    t = t.cuda()
                    a = a.cuda()
                    y = y.cuda()
                    l = l.cuda()

                y_tilde = model(t, a, l)
                print(f"Validation model output y_tilde shape: {y_tilde.shape}")

                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_loader)
        valid_losses.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 4)}")
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            torch.save(model.state_dict(), 'modelmdre.std')
            torch.save(optimizer.state_dict(), 'optimmdre.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load('modelmdre.std'))
                optimizer.load_state_dict(torch.load('optimmdre.std'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break

    return model

def evaluate(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted')
    }
    return metrics

def convert_to_sentiment_category(score):
    score = float(score)
    if score >= 2. and score <= 3.:
        return 'strongly positive'
    elif score >= 1. and score < 2.:
        return 'positive'
    elif score < 1. and score > 0.:
        return 'weakly positive'
    elif score == 0.:
        return 'neutral'
    elif score <= 0. and score > -1.:
        return 'weakly negative'
    elif score <= -1. and score > -2.:
        return 'negative'
    elif score <= -2. and score >= -3.:
        return 'neutral'
    else:
        return 'unknown'
    
def plot_sentiment_histogram(csv_file):
    data = pd.read_csv(csv_file)
    
    # Extract the labels (assumes the label is the last column)
    labels = data.iloc[:, -1].tolist()

    labels = [re.sub(r"\[(.*?)\]", r'\1', label) for label in labels]

    print("labels:", labels)
    
    labels = [re.sub(r"\'(.*?)\'", r'\1', label) for label in labels]


    labels = [float(label) if isinstance(label, str) and label.replace('.', '', 1).isdigit() else np.nan for label in labels]

    print("labels:", labels)


    # Apply the sentiment category conversion function to the labels
    sentiment_categories = [convert_to_sentiment_category(score) for score in labels]
    
    # Count the occurrences of each sentiment category
    category_counts = pd.Series(sentiment_categories).value_counts()

    # Create a directory called 'plots' if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Sentiment Category Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot as 'histogram.png' in the 'plots' directory
    plt.tight_layout()
    plt.savefig('plots/histogram.png')


def test_model_classification(model, test_loader):
    CUDA = torch.cuda.is_available()
    model.load_state_dict(torch.load('modelmdre.std'))
    print("Model loaded successfully!")

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            t, a, y, l = batch

            if CUDA:
                t = t.cuda()
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()

            y_tilde = model(t, a, l)
            loss = nn.MSELoss(reduction='sum')(y_tilde, y)
            print(f"Batch Loss: {loss.item()}")

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(y_tilde.detach().cpu().numpy())

            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test set performance (Average Loss): {avg_test_loss}")

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    print("First 10 True Values and Predictions:")
    for true, pred in zip(y_true[:10], y_pred[:10]):
        print(f"True Value: {true}, Predicted Value: {pred}")

    # Convert to sentiment categories
    y_true_categories = [convert_to_sentiment_category(score) for score in y_true]
    y_pred_categories = [convert_to_sentiment_category(score) for score in y_pred]


    
    # Evaluate metrics
    metrics = evaluate(y_true_categories, y_pred_categories)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")

    return metrics


def build():
    """Build, train, and evaluate the model."""

    initialize_sdk()
    datasetMD = setup_data()
    dataset, visual_field, acoustic_field, text_field, wordvectors_field = load_features_and_save(datasetMD)
    dataset, label_field = align_labels(dataset)
    train_split, dev_split, test_split = split_data(datasetMD)
    train, dev, test, word2id = preprocess_data_modified(dataset, visual_field, acoustic_field, text_field, wordvectors_field, label_field, train_split, dev_split, test_split)
    plot_sentiment_histogram("labels.csv")
    # save_fields_to_csv_with_row_count()        
    # save_word_to_vector_mapping()
 

    print("words: ", word2id)


    audio_input_size = 74
    audio_hidden_dim = 128
    audio_num_layers = 2
    audio_dropout = 0.5
    dic_size=len(word2id)
    
    use_glove=True
    text_encoder_size=300
    text_num_layers=2
    text_hidden_dim=128
    text_dropout=0.2
    output_size= int(1)

    batch_size = 56
    model = EncoderMDRE(
        word2id = word2id,
        encoder_size_audio = audio_input_size,
        num_layer_audio = audio_num_layers,
        hidden_dim_audio = audio_hidden_dim,
        dr_audio = audio_dropout,
        dic_size=dic_size,
        use_glove=use_glove,
        encoder_size_text=text_encoder_size,
        num_layer_text=text_num_layers,
        hidden_dim_text=text_hidden_dim,
        dr_text=text_dropout,
        output_size=output_size,
        word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
    )
    word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))

    print("Starting training...")
    trained_model = train_model(model, train_loader, dev_loader, MAX_EPOCH=1000, patience=8, grad_clip_value=1.0)

    print("Starting testing...")
    metrics = test_model_classification(trained_model, test_loader)

    return trained_model, metrics


if __name__ == "__main__":
    build()
    model, metrics = build()
    print(f"Model training completed with test metrics: {metrics}")
