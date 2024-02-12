import pandas as pd
from utils.voca_creator import create_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from utils.evaluate import get_evaluator

def train(config):
    def preprocess_function(dataset):
        inputs = [unit[config['source_lang']] for unit in dataset['translation']]
        targets = [unit[config['target_lang']] for unit in dataset['translation']]
        model_inputs = tokenizer(inputs, max_length=config['max_token_num'], truncation=True)
        # Setup the tokenizer for targets
        labels = tokenizer(targets, max_length=config['max_token_num'], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
    #model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = renew_tokenizer(config, tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    train_dataset = create_dataset(config, 'train')
    test_dataset = create_dataset(config, 'test')
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    args = Seq2SeqTrainingArguments(
        f"{config['model_name']}-finetuned-{config['source_lang']}-to-{config['target_lang']}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10000,
        predict_with_generate=True,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=get_evaluator(tokenizer)
    )
    trainer.train()

def renew_tokenizer(config, tokenizer):
    orignal_voca = list(tokenizer.vocab.keys())
    ch_token_set = pd.read_csv(config['word_file_path'])['ch']
    fr_token_set = []
    for word_unit in pd.read_csv(config['word_file_path'])['fr']:
        fr_token_set += word_unit.split(' ')

    new_ch_voca = []
    new_fr_voca = []
    for ch_token in ch_token_set:
        if ch_token not in orignal_voca:
            new_ch_voca.append(ch_token)

    for fr_token in fr_token_set:
        if fr_token not in orignal_voca:
            new_fr_voca.append(fr_token)

    print('add ch tokens: ', len(new_ch_voca))
    print('add fr tokens: ', len(new_fr_voca))
    tokenizer.add_tokens(new_ch_voca)
    tokenizer.add_tokens(new_fr_voca)
    return tokenizer

if __name__ == "__main__":
    config = {'device': 'cpu',
              'max_token_num': 512,
              'word_file_path': r'data/word.csv',
              'train_file_path': r'data/word.csv',
              'model_name': 'Babelscape/mrebel-large',
              'target_lang': 'ch',
              'source_lang': 'fr',
              'batch_size': 8,
              }
    train(config)

    '''
    google/mt5-base,
    facebook/mbart-large-50-many-to-many-mmt,
    facebook/nllb-200-distilled-600M
    Babelscape/mrebel-large
    '''

    '''dslim/bert-base-NER'''