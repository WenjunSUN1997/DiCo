import pandas as pd
from datasets import Dataset

def create_voca():
    save_path = '../data/voca/pu.csv'
    item_list = ['风尘碌碌',
                 '一事无成',
                 '十全十美']
    explain_list = ['形容在旅途上辛苦忙碌的样子',
                    '是连一样事情也没有做成，指什么事情都做不成，形容毫无成就，更多的用于贬义的语境',
                    '形容十分美好']
    voca_dict = {'item': item_list,
                 'explanation': explain_list}
    dataframe = pd.DataFrame(voca_dict)
    dataframe.to_csv(save_path)

def token_file_process(file_path=r'../data/token Fr_ch.txt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.replace('\n', '')
    content = content.split(' ')
    content = 0


def create_dataset(config, goal):
    if goal == 'train':
        word_data = pd.read_csv(config['word_file_path'])
        train_sentence_data = pd.read_csv(config['train_file_path'])
        data = pd.concat([word_data, train_sentence_data])
    else:
        data = pd.read_csv(config['train_file_path'])

    data = data.reset_index(drop=True)
    data_list = []
    for index in range(len(data)):
        data_list.append({'ch': data['ch'][index],
                          'fr': data['fr'][index]})

    dataset = Dataset.from_dict({'translation': data_list})
    return dataset


if __name__ == "__main__":
    config = {'device': 'cpu',
              'max_token_num': 512,
              'word_file_path': r'../data/word.csv',
              'train_file_path': r'../data/word.csv',
              'model_name': 'bert-base-multilingual-cased'}
    goal = 'train'
    dataset_create(config, goal)