from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class Datasetor(Dataset):
    def __init__(self, config, goal, tokenizer):
        super(Datasetor, self).__init__()
        self.tokenizer = tokenizer
        self.device = config['device']
        self.max_token_num = config['max_token_num']
        if goal == 'train':
            word_data = pd.read_csv(config['word_file_path'])
            train_sentence_data = pd.read_csv(config['train_file_path'])
            self.data = pd.concat([word_data, train_sentence_data])
        else:
            self.data = pd.read_csv(config['train_file_path'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        ch_text = self.data['ch'][item]
        fr_text = self.data['fr'][item]
        ch_output_tokenizer = self.tokenizer(ch_text,
                                             return_tensors='pt',
                                             max_length=self.max_token_num,
                                             truncation=True,
                                             padding='max_length')
        fr_output_tokenizer = self.tokenizer(fr_text,
                                             return_tensors='pt',
                                             max_length=self.max_token_num,
                                             truncation=True,
                                             padding='max_length')
        for key, value in ch_output_tokenizer.items():
            ch_output_tokenizer[key] = value.to(self.device)

        for key, value in fr_output_tokenizer.items():
            fr_output_tokenizer[key] = value.to(self.device)

        return {'ch': ch_output_tokenizer,
                'fr': fr_output_tokenizer}

if __name__ == "__main__":
    model_name = ''
    goal = 'train'
    config = {'device': 'cpu',
              'max_token_num': 512,
              'word_file_path': r'../data/word.csv',
              'train_file_path': r'../data/word.csv'}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasetor = Datasetor(config, goal, tokenizer)