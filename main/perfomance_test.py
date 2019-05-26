'''
Created on May 04, 2019

@author: dulan
'''
import time
from main.graph import Graph
from random_word import RandomWords
from params import DIR

class PerformanceTest(object):
    def __init__(self, name):
        self.name = name
        self.graph_obj_1 = Graph(DIR.GRAPH_SAV_LOC, 'lstm-graph')

    def get_random_sen(self, length):
        r = RandomWords()
        return ' '.join(r.get_random_words(limit=length))

    def perform_test(self, func):
        average = 100
        average_time = []
        word_length = []
        max_len = 50
        sentence = self.get_random_sen(max_len)
        for len in range(3, max_len, 1):
            part_sen = ' '.join(sentence.split(' ')[0:len])
            print("Length {} Sentence : {}".format(len, part_sen))
            start = time.time()
            for i in range(average):
                func(part_sen)
            end = time.time()
            avg_time = (end - start) * 1.0 / average
            print(end - start)
            average_time.append(avg_time)
            word_length.append(len)
        print(word_length)
        print(average_time)

        self.graph_obj_1.set_lables('Performance Test - {}'.format(self.name),
                                    'No of Words in Sentence', 'Average time taken (Seconds)')
        self.graph_obj_1.plot_xy(average_time, word_length, '{}-perf-test'.format(self.name))