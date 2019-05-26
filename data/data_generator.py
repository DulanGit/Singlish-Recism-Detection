'''
Created on Mar 31, 2019

@author: dulan
'''
import csv
import random
import logging
import numpy as np
import os

from main.preprocess.singlish_preprocess import singlish_preprocess
from main.preprocess.sinhala_preprocess import sinhala_preprocess
from params import DICTIONARY, FILES, DIR
from data.data_loader import data_loader
from params import MISC
from main.logger import Logger


class data_generator(object):
    def __init__(self):
        self.output_basename = FILES.SEP_CSV_FILE_PATHS
        self.data_loader_obj = data_loader()
        self.dictionary = self.data_loader_obj.load_dict(FILES.DICTIONARY_FILE_PATH)
        self.sinhala_pre_process_o = sinhala_preprocess()
        self.singlish_pre_process_o = singlish_preprocess()
        self.type_count = None
        self.logger = Logger.get_logger(DIR.LOGS_DIR, 'data-gen.log')

    def delete_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def write_to_csv(self, write_this_list, type=''):
        with open(self.output_basename.format(type), 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(write_this_list)
        writeFile.close()

    def get_random_word(self, list_of_words):
        length_of_list = len(list_of_words)
        if length_of_list == 1:
            return list_of_words[0]
        else:
            return list_of_words[random.randint(0, length_of_list - 1)]

    def set_count(self, split_dict, tags, balanced=None):
        if balanced is None:
            balanced = []
        totals = [np.count_nonzero(tags == MISC.CLASSES[0]), np.count_nonzero(tags == MISC.CLASSES[1])]
        for i in [0,1]:
            msg = "{:8} : {:4} - {}%".format(MISC.CLASSES[i], totals[i], (totals[i] / sum(totals))//0.01)
            self.log_n_print(msg)
        self.type_count = {}
        # self.type_limit = {}
        for type in split_dict.keys():
            self.type_count[type] = {}
            for i in range(2):
                self.type_count[type][MISC.CLASSES[i]] = {}
                if type in balanced:
                    self.type_count[type][MISC.CLASSES[i]]['limit'] = int(float(split_dict[type]) * int(min(totals)))
                else:
                    self.type_count[type][MISC.CLASSES[i]]['limit'] = int(float(split_dict[type]) * int(totals[i]))
                self.type_count[type][MISC.CLASSES[i]]['count'] = 0

        # self.log_n_print(self.type_limit)

    def get_type(self, clas):
        for key in self.type_count.keys():
            if self.type_count[key][clas]['count'] < self.type_count[key][clas]['limit']:
                self.type_count[key][clas]['count'] += 1
                return key
        return 'other'

    def load_sinhala_data(self):
        return self.data_loader_obj.load_data_from_excel(FILES.EXCEL_DATA_FILE_PATH)

    def convert_to_singlish(self,content, tags, augment=1):
        self.delete_file(self.output_basename.format('all'))
        self.delete_file(self.output_basename.format('all_prepro'))
        for i, line in enumerate(content):
            words = self.sinhala_pre_process_o.pre_process(line)
            word_len = np.array([len(word) for word in words])
            sorted_word_len = np.argsort(word_len)[::-1]
            for j in range(augment):
                for word in words[sorted_word_len]:
                    if word in self.dictionary.keys():
                        if not self.dictionary[word][DICTIONARY.SINGLISH_WORD] == []:
                            line = line.replace(word, self.get_random_word(self.dictionary[word][DICTIONARY.SINGLISH_WORD]))

                if len(line) < 2:
                    print("{} : {}".format(i, line))
                    continue
                self.write_to_csv([line, tags[i]], type='all')
                # line = self.singlish_pre_process_o.pre_process(line)
                # line = " ".join(line)
                #
                # self.write_to_csv([str(line), tags[i]], type='all_prepro')

    def log_n_print(self, msg):
        self.logger.info(msg)
        print(msg)

    def split_data(self, split_dict = None):
        for key in split_dict.keys():
            self.delete_file(self.output_basename.format(key))
        content, tags = self.data_loader_obj.load_data_csv(self.output_basename.format('all_prepro'))
        mapIndexPosition = list(zip(content, tags))
        random.shuffle(mapIndexPosition)
        content, tags = zip(*mapIndexPosition)

        content = np.array(content)
        tags = np.array(tags)
        self.set_count(split_dict, tags, balanced=['test'])
        for i, line in enumerate(content):
            _type=self.get_type(tags[i])
            self.write_to_csv([line, tags[i]], type=_type)
        self.log_n_print(self.type_count)

    def preprocess_csv(self):
        self.delete_file(self.output_basename.format('all_prepro'))
        content, tags = self.data_loader_obj.load_data_csv(self.output_basename.format('all'))
        for i, text in enumerate(content):
            line = self.singlish_pre_process_o.pre_process(text)
            line = " ".join(line)
            self.write_to_csv([str(line), tags[i]], type='all_prepro')

    def data_dump(self):
        self.delete_file(self.output_basename.format('all'))
        lines, tags = self.data_loader_obj.load_data_from_excel(FILES.EXCEL_DATA_FILE_PATH,0,1)
        for line, tag in zip(lines, tags):
            self.write_to_csv([str(line), tag], type='all')



if __name__ == '__main__':

    dg_obj = data_generator()
    # content, tags = dg_obj.load_sinhala_data()
    # content = ['යුදෙව්වන් යනු පරම්පරාවෙන් පැවත එන පරපෝෂිතයන් නිසා ඔවුන් විනාශ විය යුතුය. පෘථිවිය පිරිසිදු කිරීම සඳහා වූ ජාතික සමාජවාදය සාතන්ගේ නියෝගයෙන් නැවත නැඟිටින්න! 666blacksun.com']
    # tags = ['Racist']
    # dg_obj.convert_to_singlish(content, tags, augment=1)
    dg_obj.data_dump()
    dg_obj.preprocess_csv()
    dg_obj.split_data(split_dict = {'train':0.8, 'test':0.2})


