
import jieba
import os
import json
import random
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import Counter


class text_cnn_data_process:
    def __init__(self, text_path, stop_word_path, output_path, low_freq = 10, is_training= True):
        self.text_path = text_path
        self.stop_word_path = stop_word_path
        self.__low_freq = low_freq # lowest word freq
        self.__output_path = output_path
        self.__is_training = is_training
        self.__text_length = 200   ## 文章最长200

        self.vocab_size = None

    def load_data(self, input_file) -> List[List[str]] and List[int] :
        """

        :param input_file: input text dir
        :return: token_list and labels_list
        """
        print("loading data...")
        lfilenames = []
        labelnames = []
        output_token = []

        for (dirpath, dirnames, filenames) in os.walk(input_file):
            for filename in filenames:
                filename_path = os.path.join(dirpath, filename)
                lfilenames.append(filename_path)
                labelnames.append(dirpath.split("-")[-1])
        label_set = list(sorted(set(labelnames)))
        labels_dic = {label: i  for i, label in enumerate(label_set)}

        labels = [labels_dic[label] for label in labelnames]  ## label to index

        for filename in tqdm(lfilenames):
            with open(filename, 'rt', encoding='gb18030', errors='ignore') as reader:
                file = reader.read()
                output_token.append(jieba.lcut(file))

        return output_token, labels

    def remove_stop_word(self, articles):

        #所有词的list
        all_word =[word for article in articles for word in article]
        #统计词频
        word_count = Counter(all_word)
        #词频排序
        sort_word_count =sorted(word_count.items(), key=lambda x: x[1],reverse=True)
        words = [ item[1] for item in sort_word_count if  item[1] > self.__low_freq]

        if self.stop_word_path:
            with open(self.stop_word_path, 'r', encoding='utf-8') as f:
                stop_word = [ line.strip() for line in f.readlines()]
            words = [word for word in words if word not in stop_word]

        return words


    def gen_vocab(self, input_token) -> Dict[str:int]:
        """

        :param input_token:
        :return:
        """
        if self.__is_training:
            vocab = ["<PAD>", "UNK"] + input_token

            self.vocab_size = len(vocab)

            word_to_idx = dict(zip(vocab, list(range(len(vocab)))))

            with open(os.path.join(self.__output_path,"word_to_idx.json"), "w") as f:
                json.dump(word_to_idx, f, ensure_ascii=False)

        else:
            with open(os.path.join(self.__output_path,"word_to_idx.json"), 'r') as f:
                word_to_idx = json.load(f)
        return word_to_idx

    def word_to_idx(self, input_articles_list, vocab):
        """
        trans input_articles to index
        :param input_articles_list:
        :param vocab:
        :return:
        """

        output_list = [[ vocab.get(word, "<UNK>") for word in article] for article in input_articles_list]

        return output_list

    def gen_data(self):

        # 1. load_data
        texts, labels_ids = self.load_data(self.text_path)

        # 2. remove_stop_word
        remove_stop_word = self.remove_stop_word(texts)

        # 3. generate vocab
        word_vocab = self.gen_vocab(remove_stop_word)

        # 4. word to index
        articles_ids = self.word_to_idx(texts, word_vocab)

        return articles_ids, labels_ids

    def padding(self, x, y):
        """
        限定每篇文章总长为200
        :param x:
        :param y:
        :param text_length:
        :return:
        """
        file_padding = [file[:self.__text_length] if len(file) > self.__text_length
                        else file +[0] *(self.__text_length - len(file))
                        for file in x]
        return dict(file= file_padding, labels = y)

    def next_batch(self, x, y, batch_size):
        z = list(zip(x, y))
        random.shuffle(z)
        x, y = zip(*z)
        nums_batch = len(x) // batch_size
        for i in range(nums_batch):
            start = i * batch_size
            batch_x = x[start:start+batch_size]
            batch_y = y[start:start+batch_size]
            yield self.padding(batch_x, batch_y)







