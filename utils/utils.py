import pandas as pd
from datasets import Dataset
import json

def create_dataset(config):
    train_data_list = []
    test_data_list = []
    data_test = pd.read_csv(config['test_file_path']).dropna().reset_index(drop=True)
    sentence_ch = data_test['sent_chinois'].reset_index(drop=True)
    sentence_fr = data_test['sent_fr'].reset_index(drop=True)
    for index in range(len(sentence_ch)):
        test_data_list.append({'ch': sentence_ch[index],
                               'fr': sentence_fr[index]})

    test_dataset = Dataset.from_dict({'translation': test_data_list})
    data_train = pd.read_csv(config['train_file_path']).dropna().reset_index(drop=True)
    data_all = pd.concat([data_train, data_test])
    word_ch = data_all['mot_chinois'].reset_index(drop=True)
    word_fr = data_all['mot_fr'].reset_index(drop=True)
    sentence_ch = data_train['sent_chinois'].reset_index(drop=True)
    sentence_fr = data_train['sent_fr'].reset_index(drop=True)
    for index in range(len(word_ch)):
        train_data_list.append({'ch': word_ch[index],
                                'fr': word_fr[index]})

    for index in range(len(sentence_ch)):
        train_data_list.append({'ch': sentence_ch[index],
                                'fr': sentence_fr[index]})
    train_dataset = Dataset.from_dict({'translation': train_data_list})

    return (train_dataset, test_dataset)

def renew_tokenizer(config, tokenizer):
    try:
        orignal_voca = list(tokenizer.vocab.keys())
    except:
        with open(tokenizer.vocab_file) as file:
            orignal_voca = list(json.load(file).keys())

    data_source = preprocess_dataframe(config)
    ch_key = 'mot_chinois'
    ch_token_set = list(set([x for x in data_source[ch_key]]))
    new_ch_voca = []
    for ch_token in ch_token_set:
        if ch_token not in orignal_voca:
            new_ch_voca.append(ch_token)

    print('add ch tokens: ', len(new_ch_voca))
    tokenizer.add_tokens(new_ch_voca)
    return tokenizer

def preprocess_dataframe(config):
    data_frame_train = pd.read_csv(config['train_file_path']).dropna().reset_index(drop=True)
    data_frame_test = pd.read_csv(config['test_file_path']).dropna().reset_index(drop=True)
    data_source = pd.concat([data_frame_train, data_frame_test]).reset_index(drop=True)
    for index in range(len(data_source['mot_fr'])):
        if ' ' == data_source['mot_fr'][index][0]:
            data_source['mot_fr'][index] = data_source['mot_fr'][index][1:]

    return data_source

# def create_dataset(config, goal):
#     if goal == 'train':
#         word_data = pd.read_csv(config['word_file_path'])
#         train_sentence_data = pd.read_csv(config['train_file_path'])
#         data = pd.concat([word_data, train_sentence_data])
#     else:
#         data = pd.read_csv(config['train_file_path'])
#
#     data = data.reset_index(drop=True)
#     data_list = []
#     for index in range(len(data)):
#         data_list.append({'ch': data['ch'][index],
#                           'fr': data['fr'][index]})
#
#     dataset = Dataset.from_dict({'translation': data_list})
#     return dataset

# def create_voca():
#     save_path = '../data/voca/pu.csv'
#     item_list = ['风尘碌碌',
#                  '一事无成',
#                  '十全十美']
#     explain_list = ['形容在旅途上辛苦忙碌的样子',
#                     '是连一样事情也没有做成，指什么事情都做不成，形容毫无成就，更多的用于贬义的语境',
#                     '形容十分美好']
#     voca_dict = {'item': item_list,
#                  'explanation': explain_list}
#     dataframe = pd.DataFrame(voca_dict)
#     dataframe.to_csv(save_path)
#
# def token_file_process(file_path=r'../data/token Fr_ch.txt'):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#
#     content = content.replace('\n', '')
#     content = content.split(' ')
#     content = 0

def split():
    test_index = [322, 234, 277, 462, 102, 217, 44, 198, 343, 231, 490, 38, 156, 313, 1, 344, 402, 187, 422, 475, 369, 6, 108, 91, 456, 95, 206, 302, 492, 138, 85, 195, 165, 449, 243, 463, 2, 61, 239, 77, 365, 285, 265, 209, 467, 315, 443, 410, 169, 120, 70, 461, 39, 340, 4, 103, 377, 179, 225, 260, 24, 465, 122, 291, 479, 284, 131, 427, 273, 310, 112, 317, 415, 440, 210, 266, 257, 56, 106, 162, 353, 420, 452, 57, 8, 194, 419, 337, 164, 331, 389, 185, 293, 494, 76, 430, 36, 173, 71, 59, 82]
    data = pd.read_csv(r'../data/data_temp_2.csv').dropna().reset_index(drop=True)
    length_file = [x for x in range(len(data))]
    test = data.loc[test_index].reset_index(drop=True)
    train_index = list(set(length_file) - set(test_index))
    train = data.loc[train_index].reset_index(drop=True)
    test.to_csv('../data/data_temp_test.csv')
    train.to_csv('../data/data_temp_train.csv')


if __name__ == "__main__":
   split()
    # dataset_create(config, goal)