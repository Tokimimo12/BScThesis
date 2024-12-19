from typing import Optional, Tuple, List, Dict
import sys
import os
import re
import random
import numpy as np
from collections import defaultdict
import torch
from mmsdk import mmdatasdk as md

class MOSIDataProcessor:
    def __init__(self, sdk_path: str, data_path: str):
        self.sdk_path = sdk_path
        self.data_path = data_path
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

    def setup_data(self) -> md.mmdataset:
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        dataset_md = md.cmu_mosi

        try:
            md.mmdataset(dataset_md.highlevel, self.data_path)
        except RuntimeError:
            print("High-level features already downloaded.")

        try:
            md.mmdataset(dataset_md.raw, self.data_path)
        except RuntimeError:
            print("Raw data already downloaded.")

        try:
            md.mmdataset(dataset_md.labels, self.data_path)
        except RuntimeError:
            print("Labels already downloaded.")

        return dataset_md

    def load_features(self, dataset_md: md.mmdataset) -> Tuple[md.mmdataset, str, str, str, str]:
        visual_field = 'CMU_MOSI_Visual_Facet_41'
        acoustic_field = 'CMU_MOSI_COVAREP'
        text_field = 'CMU_MOSI_TimestampedWords'
        wordvectors_field = 'CMU_MOSI_TimestampedWordVectors'

        recipe = {
            text_field: os.path.join(self.data_path, text_field) + '.csd',
            wordvectors_field: os.path.join(self.data_path, wordvectors_field) + '.csd',
            visual_field: os.path.join(self.data_path, visual_field) + '.csd',
            acoustic_field: os.path.join(self.data_path, acoustic_field) + '.csd'
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

    def preprocess_data(self, 
                        dataset_md: md.mmdataset, 
                        dataset: md.mmdataset, 
                        visual_field: str, 
                        acoustic_field: str, 
                        text_field: str, 
                        wordvectors_field: str) -> Tuple[List, List, List, Dict]:
        label_field = 'CMU_MOSI_Opinion_Labels'
        label_recipe = {label_field: os.path.join(self.data_path, label_field) + '.csd'}
        dataset.add_computational_sequences(label_recipe, destination=None)
        dataset.align(label_field)

        train_split = dataset_md.standard_folds.standard_train_fold
        dev_split = dataset_md.standard_folds.standard_valid_fold
        test_split = dataset_md.standard_folds.standard_test_fold

        print(f"Lengths: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}")

        train, dev, test = [], [], []
        num_drop = 0
        pattern = re.compile('(.*)\[.*\]')

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
                    words.append(self.word2id[word[0].decode('utf-8')])
                    visual.append(_visual[i, :])
                    acoustic.append(_acoustic[i, :])
                    wordvectors.append(_wordvectors[i, :])

            words = np.asarray(words)
            visual = self._z_normalize(np.asarray(visual))
            acoustic = self._z_normalize(np.asarray(acoustic))
            wordvectors = self._z_normalize(np.asarray(wordvectors))

            if vid in train_split:
                train.append(((words, visual, acoustic, wordvectors), label, segment))
            elif vid in dev_split:
                dev.append(((words, visual, acoustic, wordvectors), label, segment))
            elif vid in test_split:
                test.append(((words, visual, acoustic, wordvectors), label, segment))

        print(f"Dropped {num_drop} inconsistent datapoints.")
        vocab_size = len(self.word2id)
        print(f"Vocabulary size: {vocab_size}")

        self.word2id.default_factory = lambda: self.UNK
        return train, dev, test, self.word2id

    def _z_normalize(self, data: np.ndarray) -> np.ndarray:
        mean = np.nanmean(data, axis=0, keepdims=True)
        std_dev = np.nanstd(data, axis=0, keepdims=True)
        std_dev = np.nan_to_num(std_dev)
        std_dev[std_dev == 0] = self.EPS
        return np.nan_to_num((data - mean) / (self.EPS + std_dev))

# Example Usage
# processor = MOSIDataProcessor(SDK_PATH, DATA_PATH)
# processor.initialize_sdk()
# dataset_md = processor.setup_data()
# dataset, visual_field, acoustic_field, text_field, wordvectors_field = processor.load_features(dataset_md)
# train, dev, test, word2id = processor.preprocess_data(dataset_md, dataset, visual_field, acoustic_field, text_field, wordvectors_field)
