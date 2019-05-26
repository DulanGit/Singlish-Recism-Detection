'''
Created on Mar 31, 2019

@author: dulan
'''
from enum import Enum, unique
import os
import os.path as osp

ABS_PATH_TO_FILE = os.path.dirname(__file__)

@unique
class PRED_TYPE(Enum):
    SIMPLE_NN   = 1
    SVM         = 2
    MNB         = 3

class DICTIONARY:
    SINGLISH_WORD   = 'singlish'
    WORD_COUNT      = 'count'

class DIR:
    DEF_SAV_LOC     = osp.join(ABS_PATH_TO_FILE, 'main/save')
    GRAPH_SAV_LOC   = osp.join(ABS_PATH_TO_FILE, 'main/save/graphs')
    LOGS_DIR        = osp.join(ABS_PATH_TO_FILE, 'main/logs')

class FILES:
    TAG_DATA_FILENAME_EXCEL = osp.join(ABS_PATH_TO_FILE, 'data/final-data-set.xlsx')
    TAG_DICT_FILENAME       = osp.join(ABS_PATH_TO_FILE, 'data/dict.json')
    EXCEL_DATA_FILE_PATH    = osp.join(ABS_PATH_TO_FILE, 'data/xlsx/Simple-Data-Set.xlsx')
    DICTIONARY_FILE_PATH    = osp.join(ABS_PATH_TO_FILE, 'data/dict.json')
    CSV_FILE_PATH           = osp.join(ABS_PATH_TO_FILE, 'data/csv/output.csv')
    SEP_CSV_FILE_PATHS      = osp.join(ABS_PATH_TO_FILE, 'data/csv/output_{}.csv')
    TEST_CSV_FILE_PATH      = osp.join(ABS_PATH_TO_FILE, 'tests/test_output.csv')

class PREPRO:
    LEMMAS_FILENAME = osp.join(ABS_PATH_TO_FILE, 'main/preprocess/lemmas')

class MISC:
    TAG_SAVE_COUNT = 5
    CLASSES = ['Racist', 'Neutral']

class GRUP:
    LSTM_MAX_WORD_COUNT     = 60
    FOLDS_COUNT             = 4
    VALIDATION_TEST_SIZE    = 0.2
    MAX_EPOCHS              = 15
    BATCH_SIZE              = 4
    DICT_SIZE               = -1
    OUTPUT_DIR              = osp.join(ABS_PATH_TO_FILE, 'main/save')
    N_GRAM_LEN              = 1
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }
    BREAK_AFTER_FOLD        = -1

    DICTIONARY_FILENAME = 'gru_dict'
    MODEL_FILENAME = 'gru_model.h5'
    MODEL_FILENAME_ACC = 'gru_model_{}.h5'
    LOG_FILE_NAME = 'gru-model'
    GRAPH_NAME              = 'gru-graph'
    PERF_TEST_NAME          = 'GRU'

class LSTMP:
    LSTM_MAX_WORD_COUNT     = 20
    FOLDS_COUNT             = 4
    VALIDATION_TEST_SIZE    = 0.2
    MAX_EPOCHS              = 15
    BATCH_SIZE              = 4
    DICT_SIZE               = -1
    OUTPUT_DIR              = osp.join(ABS_PATH_TO_FILE, 'main', 'save')
    N_GRAM_LEN              = 1
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }
    BREAK_AFTER_FOLD        = -1

    DICTIONARY_FILENAME = 'lstm_dict'
    MODEL_FILENAME = 'lstm_model.h5'
    MODEL_FILENAME_ACC = 'lstm_model_{}.h5'
    LOG_FILE_NAME = 'lstm-model'
    GRAPH_NAME = 'lstm-graph'
    PERF_TEST_NAME = 'LSTM'


class SNN:
    TRANS_FILENAME = 'snn_trans'
    MODEL_FILENAME = 'snn_model.h5'
    INPUT_FILENAME = 'snn_input'
    LOG_FILE_NAME   = 'snn'

class NN:
    FOLDS_COUNT             = 5
    VALIDATION_TEST_SIZE    = 0.2
    MAX_EPOCHS              = 10
    BATCH_SIZE              = 10
    OUTPUT_DIR              = osp.join(ABS_PATH_TO_FILE,'save')
    LOG_FILE_NAME           = 'neural-net'
    DATA_SET_CLASSES = {
        'Neutral': [0, 1],
        'Racist': [1, 0]
    }
    BREAK_AFTER_FOLD        = 1
    TRANS_FILENAME = 'nn_trans'
    MODEL_FILENAME = 'nn_model.h5'
    INPUT_FILENAME = 'nn_input'

class SVMF():
    BOW_FILENAME    = 'svm_bow'
    TFIDF_FILENAME  = 'svm_tfidf'
    MODEL_FILENAME  = 'svm_model'
    INPUT_FILENAME  = 'svm_input'
    LOG_FILE_NAME   = osp.join(ABS_PATH_TO_FILE, DIR.LOGS_DIR, 'svm.log')

class MNB():
    BOW_FILENAME    = 'mnb_bow'
    TFIDF_FILENAME  = 'mnb_tfidf'
    MODEL_FILENAME  = 'mnb_model'
    INPUT_FILENAME  = 'mnb_input'
    LOG_FILE_NAME   = osp.join(ABS_PATH_TO_FILE, DIR.LOGS_DIR, 'mnb.log')
