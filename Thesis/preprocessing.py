from typing import Optional, Tuple, List, Dict
import sys
import os
import re
import random
import csv
import numpy as np
from collections import defaultdict
import torch
from mmsdk import mmdatasdk as md
import pandas as pd
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH



class Preprocessing:
    def __init__(self):
        self.sdk_path = SDK_PATH
        self.data_path = DATA_PATH
        self.word2id = defaultdict(lambda: len(self.word2id))
        self.word2id['<unk>']  # Initialize UNK token
        self.word2id['<pad>']  # Initialize PAD token
        self.UNK = self.word2id['<unk>']
        self.PAD = self.word2id['<pad>']
        self.EPS = 1e-8
        
    def initialize_sdk(self):
        if not self.sdk_path:
            raise ValueError("SDK path is not specified! Please specify it first.")
        sys.path.append(self.sdk_path)
        print(f"SDK path is set to {self.sdk_path}")

    def setup_data(self):
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

    def load_features_and_save(self, DATASET):
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

    def preprocess_data_modified(self, dataset, 
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

            print ("label:", label)
            num_classes = 7
            class_index = Preprocessing.convert_to_sentiment_category(label)
            print ("class_index:", class_index)
            one_hot_label = np.zeros(num_classes)
            one_hot_label[class_index] = 1
            print ("one_hot_label:", one_hot_label)

            dataset[label_field][segment]['features'] = one_hot_label  # Replace with the class index
            label = one_hot_label

            print ("labeltest:", label)
            
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

        
        
        return train, dev, test, word_to_ids


    def _z_normalize(self, data: np.ndarray) -> np.ndarray:
        mean = np.nanmean(data, axis=0, keepdims=True)
        std_dev = np.nanstd(data, axis=0, keepdims=True)
        std_dev = np.nan_to_num(std_dev)
        std_dev[std_dev == 0] = self.EPS
        return np.nan_to_num((data - mean) / (self.EPS + std_dev))


    def save_fields_to_csv_with_row_count(self):
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


    def align_labels(self, dataset):
        label_field = 'CMU_MOSI_Opinion_Labels'
        label_recipe = {label_field: os.path.join(DATA_PATH, label_field) + '.csd'}
        dataset.add_computational_sequences(label_recipe, destination=None)
        dataset.align(label_field)
        
        return dataset, label_field


    def split_data(self, DATASETMD):
        train_split = DATASETMD.standard_folds.standard_train_fold
        dev_split = DATASETMD.standard_folds.standard_valid_fold
        test_split = DATASETMD.standard_folds.standard_test_fold

        print(f"lengths: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}\n")
        print(train_split)
        print(dev_split)
        print(test_split)
        return train_split, dev_split, test_split
    

    def save_unique_words(self, word_to_ids, output_csv="unique_words.csv"):
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
        

    def save_wordvectors_to_csv(self):
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

    def seeGloVeVectors(self):
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

    def save_word_to_vector_mapping(self):
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



    def extract_first_column(self, csv_path, output_csv="extracted_words.csv"):
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

    @staticmethod
    def convert_to_sentiment_category(score):
        score = float(score)
        if 2. <= score <= 3.:
            return 6 #'strongly positive'
        elif 1. <= score < 2.:
            return 5 #'positive'
        elif 0. < score < 1.:
            return 4 #'weakly positive'
        elif score == 0.:
            return 3 #'neutral'
        elif -1. < score < 0.:
            return 2 #'weakly negative'
        elif -2. < score <= -1.:
            return 1 #'negative'
        elif -3. <= score <= -2.:
            return 0 #'strongly negative'
        else:
            print(f"Warning: Sentiment score out of expected range: {score}")
            return None  

