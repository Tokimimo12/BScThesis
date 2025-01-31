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
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds import standard_train_fold, standard_valid_fold, standard_test_fold

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

        # words, visual, acoustic, wordvectors = [], [], [], []
        # for i, word in enumerate(_words):
        #     words.append(word_to_ids[word[0].decode('utf-8')])
        #     if word[0] != b'sp':
        #         visual.append(_visual[i, :])
        #         acoustic.append(_acoustic[i, :])
        #         wordvectors.append(_wordvectors[i, :])

        # for i, word in enumerate(words):
        #     word_text = word[0].decode('utf-8')  # Decode the word from bytes
        #     if word_text != 'sp':  # Ignore silence padding ('sp')
        #         word_to_ids[word_text].append(wordvectors[i])

        words, visual, acoustic, wordvectors = [], [], [], []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word_to_ids[word[0].decode('utf-8')])
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
                wordvectors.append(_wordvectors[i, :])            
        # for i, word in enumerate(_words):

        #     # Use _visual[i] when working with 1D or 2D arrays; it’s simpler and achieves the same result.
        #     # Use _visual[i, :] if you want to emphasize that you’re taking a full row from the second dimension explicitly. This can improve code readability when dealing with multi-dimensional arrays.

        #     if word[0] != b'sp':  # Ignore silence markers
        #         # Process words: decode and map to ID
        #         decoded_word = word[0].decode('utf-8')  # Decode byte string
        #         word_id = word_to_ids[decoded_word]  # Convert word to ID using mapping
        #         words.append(word_id)  # Append ID to words list
                
        #         # Append corresponding visual, acoustic, and wordvectors features
        #         visual.append(_visual[i, :])
        #         acoustic.append(_acoustic[i, :])
        #         wordvectors.append(_wordvectors[i, :])

        
        # Stack features row-wise (word ID with corresponding visual, acoustic, wordvectors)
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




def save_unique_words(word_to_ids, output_csv="unique_words.csv"):
    """
    Saves the unique words and their corresponding IDs to a CSV file.

    Args:
        word_to_ids (defaultdict): A mapping of words to their IDs.
        output_csv (str): Path to the output CSV file.
    """
    directory = os.path.dirname(output_csv)
    if directory:  # Only try to create the directory if it exists in the path
        os.makedirs(directory, exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for word, word_id in word_to_ids.items():
            writer.writerow([word, word_id])
    
    print(f"Unique words have been saved to {output_csv}")
    

# Example usage after your preprocessing:

# def preprocess_data_initially(dataset, visual_field, acoustic_field, text_field, wordvectors_field, label_field, train_split, dev_split, test_split):
#     print("Preprocessing data initially...")

#     word2id = defaultdict(lambda: len(word2id))
#     print(f"len word2id: {len(word2id)}")
    
#     UNK = word2id['<unk>']
#     PAD = word2id['<pad>']
#     pattern = re.compile('(.*)\[.*\]')

#     train, dev, test = [], [], []
#     EPS = 1e-8
#     num_drop = 0

#     for segment in dataset[label_field].keys():
#         vid = re.search(pattern, segment).group(1)
#         label = dataset[label_field][segment]['features']
#         _words = dataset[text_field][segment]['features']  # Extract the words
#         _visual = dataset[visual_field][segment]['features']
#         _acoustic = dataset[acoustic_field][segment]['features']
#         _wordvectors = dataset[wordvectors_field][segment]['features']

#         if not (_words.shape[0] == _visual.shape[0] == _acoustic.shape[0] == _wordvectors.shape[0]):
#             num_drop += 1
#             continue

#         words, visual, acoustic, wordvectors = [], [], [], []
#         for i, word in enumerate(_words):
#             if word[0] != b'sp':
#                 words.append(word2id[word[0].decode('utf-8')])
#                 visual.append(_visual[i, :])
#                 acoustic.append(_acoustic[i, :])
#                 wordvectors.append(_wordvectors[i, :])

#         words = np.asarray(words)
#         visual = np.asarray(visual)
#         acoustic = np.asarray(acoustic)
#         wordvectors = np.asarray(wordvectors)


#         std_dev_visual = np.std(visual, axis=0, keepdims=True)
#         visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + std_dev_visual))
#         visual[:, std_dev_visual.flatten() == 0] = EPS  # Safeguard for zero standard deviation

#         # Z-normalization for acoustic modality (across all acoustic features)
#         acoustic_mean = np.nanmean(acoustic, axis=0, keepdims=True)
#         std_dev_acoustic = np.nanstd(acoustic, axis=0, keepdims=True)
#         std_dev_acoustic = np.nan_to_num(std_dev_acoustic)
#         std_dev_acoustic[std_dev_acoustic == 0] = EPS  # Safeguard for zero standard deviation

#         acoustic = np.nan_to_num((acoustic - acoustic_mean) / (EPS + std_dev_acoustic))

#         # Z-normalization for word vectors
#         wordvectors_mean = np.nanmean(wordvectors, axis=0, keepdims=True)
#         std_dev_wordvectors = np.nanstd(wordvectors, axis=0, keepdims=True)
#         std_dev_wordvectors = np.nan_to_num(std_dev_wordvectors)
#         std_dev_wordvectors[std_dev_wordvectors == 0] = EPS  # Safeguard for zero standard deviation
#         wordvectors = np.nan_to_num((wordvectors - wordvectors_mean) / (EPS + std_dev_wordvectors))

#         # plot_hist2(wordvectors, acoustic)

#         # Ensure no NaN or Inf values in the data
#         if np.any(np.isnan(acoustic)) or np.any(np.isinf(acoustic)):
#             print(f"Error in acoustic data for segment {vid}")
#         if np.any(np.isnan(visual)) or np.any(np.isinf(visual)):
#             print(f"Error in visual data for segment {vid}")
#         if np.any(np.isnan(words)) or np.any(np.isinf(words)):
#             print(f"Error in wordvectors data for segment {vid}")

#         if vid in train_split:
#             train.append(((words, visual, acoustic, wordvectors), label, segment))
#         elif vid in dev_split:
#             dev.append(((words, visual, acoustic, wordvectors), label, segment))
#         elif vid in test_split:
#             test.append(((words, visual, acoustic, wordvectors), label, segment))

#     print(f"Dropped {num_drop} inconsistent datapoints.")
#     vocab_size = len(word2id)
#     print(f"Vocabulary size: {vocab_size}")
    
#     def return_unk():
#         return UNK
#     word2id.default_factory = return_unk

#     return train, dev, test, word2id

def save_wordvectors_to_csv():
    # Path to the CSD file
    path_to_csd = "/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd"
    output_csv = "word_vectors.csv"

    # Load the CSD file
    dataset = md.mmdataset({'wordvectors': path_to_csd})

    # Open a CSV file to write word vectors
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Loop through all segments and write word vectors
        for segment in dataset['wordvectors'].keys():
            word_vectors = dataset['wordvectors'][segment]['features']  # Extract word vectors
            for vector in word_vectors:
                writer.writerow(vector)  # Write each word vector as a row in the CSV file

    print(f"Word vectors have been saved to {output_csv}")

# def seeGloVeVectors():
#     # Path to the CSD file
#     path_to_csd = "/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd"

#     # Load the CSD file
#     dataset = md.mmdataset({'wordvectors': path_to_csd})

#     # Print dataset keys (segment names)
#     print("Segments in the dataset:", list(dataset['wordvectors'].keys())[:5])  # Print first 5 keys

#     # Explore a sample segment
#     sample_segment = list(dataset['wordvectors'].keys())[0]
#     print(f"\nSample segment: {sample_segment}")
#     print("Word vectors for the segment (shape):", dataset['wordvectors'][sample_segment]['features'].shape)
#     print("First 5 word vectors:")
#     print(dataset['wordvectors'][sample_segment]['features'][:10])  # Print first 5 word vectors

def save_word_to_vector_mapping():
    # Paths to the CSD files
    path_to_wordvectors_csd = "/home1/s4680340/BScThesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd"
    path_to_words_csd = "/home1/s4680340/BScThesis/Thesis/data/CMU_MOSI_TimestampedWords.csd"
    output_csv = "word_to_vector_mapping.csv"

    # Load the word vectors and timestamped words
    dataset = md.mmdataset({'wordvectors': path_to_wordvectors_csd, 'words': path_to_words_csd})

    # Dictionary to store word-to-vector mapping
    word_to_vector = defaultdict(list)

    # Loop through all segments
    for segment in dataset['wordvectors'].keys():
        word_vectors = dataset['wordvectors'][segment]['features']
        words = dataset['words'][segment]['features']  # Extract the words
        
        if word_vectors.shape[0] != words.shape[0]:
            print(f"Skipping segment {segment} due to mismatched shapes")
            continue
        
        # Map each word to its corresponding vector
        for i, word in enumerate(words):
            word_text = word[0].decode('utf-8')  # Decode the word from bytes
            if word_text != 'sp':  # Ignore silence padding ('sp')
                word_to_vector[word_text].append(word_vectors[i])

    # Average word vectors for words that appear multiple times
    averaged_word_to_vector = {word: np.mean(vectors, axis=0) for word, vectors in word_to_vector.items()}

    # Write the words and their vectors to a CSV file
    total_rows = 0  # Initialize row counter
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # header = ['word'] + [f"dim_{i+1}" for i in range(len(next(iter(averaged_word_to_vector.values()))))]
        # writer.writerow(header)
        
        # Write the word-to-vector mapping
        for word, vector in averaged_word_to_vector.items():
            row = [word] + vector.tolist()
            writer.writerow(row)
            total_rows += 1  # Increment row counterayaya

    print(f"Word-to-vector mapping has been saved to {output_csv}")
    print(f"Total number of rows (unique words): {total_rows}")

def load_embeddings(w2i, path_to_embedding, embedding_size=300):
    """Load GloVe-like word embeddings and create embedding matrix."""
    emb_mat = np.random.randn(len(w2i), embedding_size)
    with open(path_to_embedding, 'r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, total=2196017):
            content = line.strip().split()
            vector = np.asarray(content[-embedding_size:], dtype=float)
            word = ' '.join(content[:-embedding_size])
            if word in w2i:
                emb_mat[w2i[word]] = vector

    emb_mat_tensor = torch.tensor(emb_mat).float()
    torch.save(emb_mat_tensor, 'embedding_matrix.pt')
    print("Embedding matrix saved as 'embedding_matrix.pt'.")
    return emb_mat_tensor


def collate_batch(batch, pad_value):
    """Collate function to handle variable-length sequences in a batch."""
    batch = sorted(batch, key=lambda x: len(x[0][0]), reverse=True)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=pad_value, batch_first=True)
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()
    lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
    return sentences, labels, lengths


# def train_model(model, train_loader, dev_loader, max_epoch=1000, patience=8, grad_clip_value=1.0):
#     """Train the model with the given data loaders."""
#     CUDA = torch.cuda.is_available()
#     optimizer = model.create_optimizer(lr=0.001)
#     criterion = nn.MSELoss(reduction='sum')
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
#     lr_scheduler.step()

#     if CUDA:
#         model.cuda()

#     best_valid_loss = float('inf')
#     curr_patience = patience
#     num_trials = 3

#     for e in range(max_epoch):
#         model.train()
#         train_loss = 0.0
#         for batch in tqdm(train_loader, desc=f"Epoch {e}/{max_epoch}"):
#             model.zero_grad()
#             t, y, l = batch
#             if CUDA:
#                 t, y, l = t.cuda(), y.cuda(), l.cuda()
#             y_tilde = model(t, l)
#             loss = criterion(y_tilde, y)
#             loss.backward()
#             torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
#             optimizer.step()
#             train_loss += loss.item()

#         avg_train_loss = train_loss / len(train_loader)
#         print(f"Epoch {e} - Training loss: {avg_train_loss:.4f}")

#         model.eval()
#         valid_loss = 0.0
#         with torch.no_grad():
#             for batch in dev_loader:
#                 t, y, l = batch
#                 if CUDA:
#                     t, y, l = t.cuda(), y.cuda(), l.cuda()
#                 y_tilde = model(t, l)
#                 loss = criterion(y_tilde, y)
#                 valid_loss += loss.item()

#         avg_valid_loss = valid_loss / len(dev_loader)
#         print(f"Epoch {e} - Validation loss: {avg_valid_loss:.4f}")

#         if avg_valid_loss <= best_valid_loss:
#             best_valid_loss = avg_valid_loss
#             print("New best model found! Saving...")
#             torch.save(model.state_dict(), 'modeltext.std')
#             torch.save(optimizer.state_dict(), 'modeltext.std')
#             curr_patience = patience
#         else:
#             curr_patience -= 1
#             if curr_patience <= 0:
#                 print("Early stopping due to lack of improvement.")
#                 break

#         if num_trials <= 0:
#             print("Running out of patience, early stopping.")
#             break

#     return model



def convert_to_sentiment_category(score):
    if score == 3:
        return 'strongly positive'
    elif score >= 2 and score < 3:
        return 'positive'
    elif score >= 1 and score < 2:
        return 'weakly positive'
    elif score < 1 and score > -1:
        return 'neutral'
    elif score <= -1 and score > -2:
        return 'weakly negative'
    elif score <= -2 and score > -3:
        return 'negative'
    elif score == -3:
        return 'strongly negative'
    else:
        return 'unknown'


def train_model(model, train_loader, dev_loader, MAX_EPOCH=1000, patience=8, num_trials = 3, grad_clip_value=1.0):
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
            t, y, l = batch
            batch_size = t.size(0)
            if CUDA:
                t = t.cuda()
                y = y.cuda()
                l = l.cuda()

            _, y_tilde, _ = model(t, l)
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
                t, y, l = batch
                if CUDA:
                    t = t.cuda()
                    y = y.cuda()
                    l = l.cuda()
                _, y_tilde, _ = model(t, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_loader)
        valid_losses.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 4)}")
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            torch.save(model.state_dict(), 'modeltext.std')
            torch.save(optimizer.state_dict(), 'optimtext.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load('modeltext.std'))
                optimizer.load_state_dict(torch.load('optimtext.std'))
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

def test_model(model, test_loader):
    """
    Tests the model using the given test data loader.
    """
    CUDA = torch.cuda.is_available()
    model.load_state_dict(torch.load('modeltext.std'))
    print("Model loaded successfully!")

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            t, y, l = batch

            if CUDA:
                t = t.cuda()
                y = y.cuda()
                l = l.cuda()
            print(f"l: {l}")

            _, y_tilde, _ = model(t, l)
            loss = nn.MSELoss(reduction='sum')(y_tilde, y)
            print(f"Batch Loss: {loss.item()}")

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(y_tilde.detach().cpu().numpy())

            test_loss += loss.item()
            
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test set performance (Average Loss): {avg_test_loss}")

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    y_true_categories = [convert_to_sentiment_category(score) for score in y_true]
    y_pred_categories = [convert_to_sentiment_category(score) for score in y_pred]

    metrics = evaluate(y_true_categories, y_pred_categories)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")

    # roc_auc_multiclass(y_true, y_pred_prob)


    print("Attention!!!!")
    return metrics

def extract_first_column(csv_path, output_csv="extracted_words.csv"):
    """
    Extract the first column from a CSV file and save it to a text file.

    Args:
    csv_path (str): Path to the input CSV file.
    output_csv (str): Path to the output file where the first column will be saved.
    """
    first_column = []
    directory = os.path.dirname(output_csv)
    if directory:  # Only try to create the directory if it exists in the path
        os.makedirs(directory, exist_ok=True)

    try:
        with open(csv_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:  # Ensure the row is not empty
                    first_column.append(row[0])  # Extract the first column value

        # Save the first column to a text file
        with open(output_csv, mode='w', encoding='utf-8') as txtfile:
            txtfile.write("\n".join(first_column))

        print(f"First column has been extracted and saved to {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")

def compare(file1, file2):
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    # Get the words from the first column of each file
    words1 = set(df1[0])
    words2 = set(df2[0])

    # Find common words
    common_words = words1.intersection(words2)

    # Find words that are in only one of the files
    only_in_file1 = words1 - words2
    only_in_file2 = words2 - words1

    # Prepare data for 'common.csv'
    common_data = list(common_words)

    # Prepare data for 'different.csv'
    different_data = []
    for word in only_in_file1:
        different_data.append([word, file1])
    for word in only_in_file2:
        different_data.append([word, file2])

    # Create DataFrames for the new CSV files
    common_df = pd.DataFrame(common_data, columns=["Common Words"])
    different_df = pd.DataFrame(different_data, columns=["Different Words", "Source File"])

    # Save the results to new CSV files
    common_df.to_csv('common.csv', index=False)
    different_df.to_csv('different.csv', index=False)

    print("CSV files created: common.csv and different.csv")



def build():
    """Build, train, and evaluate the model."""

    initialize_sdk()
    datasetMD = setup_data()
    dataset, visual_field, acoustic_field, text_field, wordvectors_field = load_features_and_save(datasetMD)
    dataset, label_field = align_labels(dataset)
    train_split, dev_split, test_split = split_data(datasetMD)
    # save_fields_to_csv_with_row_count()    
    train, dev, test, word2id = preprocess_data_modified(dataset, visual_field, acoustic_field, text_field, wordvectors_field, label_field, train_split, dev_split, test_split)
    
    print("words: ", word2id)
    save_unique_words(word2id, output_csv="unique_words_modified.csv")
    
    extract_first_column("/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv", output_csv="extracted_words_unprocessed.csv")
    
    compare('/home1/s4680340/BScThesis/Thesis/unique_words_modified.csv', 
                  '/home1/s4680340/BScThesis/Thesis/extracted_words_unprocessed.csv')
    
    # save_word_to_vector_mapping()

    # pretrained_emb = None
    # if os.path.exists(CACHE_PATH):
    #     pretrained_emb, word2id = torch.load(CACHE_PATH)
    #     print(f"Size of vocabulary (word2id) after pretrained 1: {len(word2id)}")
    # elif WORD_EMB_PATH:
    #     pretrained_emb = load_embeddings(word2id, WORD_EMB_PATH)
    #     torch.save((pretrained_emb, word2id), CACHE_PATH)
    
    model = SingleEncoderModelText(
        word2id = word2id,
        dic_size=len(word2id),
        use_glove=True,
        encoder_size=300,
        num_layers=2,
        hidden_dim=128,
        dr=0.2,
        output_size=1,
        word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
    )
    word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'

    # word_to_check = "quality"
    # is_correct = model.verify_embedding(word_to_check, '/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv')
    # print(f"Embedding verification for '{word_to_check}': {'Passed' if is_correct else 'Failed'}")

    # model.verify_all_embeddings(word2id=word2id, word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv', output_file = 'embedding_verification.csv')
    # model.save_embeddings_to_csv(word2id, word_to_vector_path, output_file='embedding_verification.csv')
    model.check_embedding_layer(word2id, word_to_vector_path)
    
    embedding_weights = model.embedding.weight.data
    print(f"Embedding weights shape: {embedding_weights.shape}")

    dummy_input = torch.tensor([[1, 2, 3], [4, 5, 0]])  # Example input
    lengths = torch.tensor([3, 2])  # Sequence lengths
    output = model(dummy_input, lengths)
    print(f"Model output: {output}")


    '''
    

    print("Initializing model...")
    model = SingleEncoderModelText(
        dic_size=len(word2id),
        use_glove=True,
        encoder_size=300,
        num_layers=2,
        hidden_dim=128,
        dr=0.2,
        output_size=1,
        word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
    )

    
    '''
    batch_size = 56
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))

    
    print("Starting training...")
    trained_model = train_model(model, train_loader, dev_loader, MAX_EPOCH=1000, patience=8, grad_clip_value=1.0)

    print("Starting testing...")
    accuracy = test_model(trained_model, test_loader)

    return trained_model, accuracy
if __name__ == "__main__":
    # build()
    model, test_accuracy = build()
    print(f"Model training completed with test accuracy: {test_accuracy:.4f}")
