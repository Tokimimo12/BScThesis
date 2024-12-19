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

def load_features(DATASET):
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
    return dataset, visual_field, acoustic_field, text_field, wordvectors_field

def preprocess_data(DATASETMD, dataset, visual_field, acoustic_field, text_field, wordvectors_field):
    label_field = 'CMU_MOSI_Opinion_Labels'
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field) + '.csd'}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)

    
    train_split = DATASETMD.standard_folds.standard_train_fold
    dev_split = DATASETMD.standard_folds.standard_valid_fold
    test_split = DATASETMD.standard_folds.standard_test_fold
    # train_split = standard_train_fold
    # dev_split = standard_valid_fold
    # test_split = standard_test_fold
    

    print(f"lengths: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}\n")
    print(train_split)
    print(dev_split)
    print(test_split)


    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']
    pattern = re.compile('(.*)\[.*\]')

    train, dev, test = [], [], []
    EPS = 1e-8
    num_drop = 0

    for segment in dataset[label_field].keys():
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        _words = dataset[text_field][segment]['features']
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        _wordvectors = dataset[wordvectors_field][segment]['features']

        if not (_words.shape[0] == _visual.shape[0] == _acoustic.shape[0] == _wordvectors.shape[0]):
            num_drop += 1
            continue

        words, visual, acoustic, wordvectors = [], [], [], []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word2id[word[0].decode('utf-8')])
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
                wordvectors.append(_wordvectors[i, :])

        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)
        wordvectors = np.asarray(wordvectors)


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

    print(f"Dropped {num_drop} inconsistent datapoints.")
    vocab_size = len(word2id)
    print(f"Vocabulary size: {vocab_size}")
    
    def return_unk():
        return UNK
    word2id.default_factory = return_unk

    return train, dev, test, word2id


def seeGloVeVectors():
    # Path to the CSD file
    path_to_csd = "/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd"

    # Load the CSD file
    dataset = md.mmdataset({'wordvectors': path_to_csd})

    # Print dataset keys (segment names)
    print("Segments in the dataset:", list(dataset['wordvectors'].keys())[:5])  # Print first 5 keys

    # Explore a sample segment
    sample_segment = list(dataset['wordvectors'].keys())[0]
    print(f"\nSample segment: {sample_segment}")
    print("Word vectors for the segment (shape):", dataset['wordvectors'][sample_segment]['features'].shape)
    print("First 5 word vectors:")
    print(dataset['wordvectors'][sample_segment]['features'][:10])  # Print first 5 word vectors

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



def save_word_to_vector_mapping():
    import csv
import numpy as np
from collections import defaultdict
from mmsdk import mmdatasdk as md

def save_word_to_vector_mapping():
    # Paths to the CSD files
    path_to_wordvectors_csd = "/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWordVectors.csd"
    path_to_words_csd = "/home1/s4680340/BSc-Thesis/Thesis/data/CMU_MOSI_TimestampedWords.csd"
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
        
        # Write the header
        header = ['word'] + [f"dim_{i+1}" for i in range(len(next(iter(averaged_word_to_vector.values()))))]
        writer.writerow(header)
        
        # Write the word-to-vector mapping
        for word, vector in averaged_word_to_vector.items():
            row = [word] + vector.tolist()
            writer.writerow(row)
            total_rows += 1  # Increment row counter

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


def train_model(model, train_loader, dev_loader, max_epoch=1000, patience=8, grad_clip_value=1.0):
    """Train the model with the given data loaders."""
    CUDA = torch.cuda.is_available()
    optimizer = model.create_optimizer(lr=0.001)
    criterion = nn.MSELoss(reduction='sum')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step()

    if CUDA:
        model.cuda()

    best_valid_loss = float('inf')
    curr_patience = patience
    num_trials = 3

    for e in range(max_epoch):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {e}/{max_epoch}"):
            model.zero_grad()
            t, y, l = batch
            if CUDA:
                t, y, l = t.cuda(), y.cuda(), l.cuda()
            y_tilde = model(t, l)
            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {e} - Training loss: {avg_train_loss:.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                t, y, l = batch
                if CUDA:
                    t, y, l = t.cuda(), y.cuda(), l.cuda()
                y_tilde = model(t, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(dev_loader)
        print(f"Epoch {e} - Validation loss: {avg_valid_loss:.4f}")

        if avg_valid_loss <= best_valid_loss:
            best_valid_loss = avg_valid_loss
            print("New best model found! Saving...")
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(optimizer.state_dict(), 'best_optimizer.pth')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                print("Early stopping due to lack of improvement.")
                break

    return model


def test_model(model, test_loader):
    """Test the trained model on the test dataset."""
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    y_true = []
    y_pred = []
    criterion = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            t, y, l = batch
            y_tilde = model(t, l)
            loss = criterion(y_tilde, y)
            test_loss += loss.item()

            y_true.append(y.cpu().numpy())
            y_pred.append(y_tilde.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test set performance (Average Loss): {avg_test_loss:.4f}")

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def build():
    """Build, train, and evaluate the model."""

    initialize_sdk()
    datasetMD = setup_data()
    dataset, visual_field, acoustic_field, text_field, wordvectors_field = load_features(datasetMD)

    train, dev, test, word2id = preprocess_data(datasetMD, dataset, visual_field, acoustic_field, text_field, wordvectors_field)
    seeGloVeVectors()
    save_wordvectors_to_csv()
    save_word_to_vector_mapping()

    # pretrained_emb = None
    # if os.path.exists(CACHE_PATH):
    #     pretrained_emb, word2id = torch.load(CACHE_PATH)
    #     print(f"Size of vocabulary (word2id) after pretrained 1: {len(word2id)}")
    # elif WORD_EMB_PATH:
    #     pretrained_emb = load_embeddings(word2id, WORD_EMB_PATH)
    #     torch.save((pretrained_emb, word2id), CACHE_PATH)


    # input_size = 74
    # hidden_sizes = 128
    # num_layers = 2
    # output_size = 1
    # dropout = 0.5

    # print("Initializing model...")
    # model = SingleEncoderModelText(
    #     dic_size=len(word2id),
    #     use_glove=True,
    #     encoder_size=300,
    #     num_layers=2,
    #     hidden_dim=128,
    #     dr=0.2,
    #     output_size=1
    # )

    # batch_size = 56
    # train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    # dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    # test_loader = DataLoader(test, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))

    
    print("Starting training...")
    # trained_model = train_model(model, train_loader, dev_loader)

    # print("Starting testing...")
    # accuracy = test_model(trained_model, test_loader)

    # return trained_model, accuracy


if __name__ == "__main__":
    build()
    # model, test_accuracy = build()
    # print(f"Model training completed with test accuracy: {test_accuracy:.4f}")
