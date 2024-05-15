import argparse
import submitit
from utils.utils import create_dataset, renew_tokenizer, preprocess_dataframe
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
     DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
from utils.evaluate import get_evaluator
from datetime import datetime
import evaluate

def train(config):
    def preprocess_function(dataset):
        inputs = [unit[config['source_lang']] for unit in dataset['translation']]
        targets = [unit[config['target_lang']] for unit in dataset['translation']]
        model_inputs = tokenizer(inputs, max_length=config['max_token_num'], truncation=True)
        # Setup the tokenizer for targets
        labels = tokenizer(targets, max_length=config['max_token_num'], truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print('********'+config['model_name']+'*****************')
    output_dir = config['output_dir'] + '/' + config['model_name'].split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], use_fast=True)
    if config['goal'] == 'train':
        tokenizer = renew_tokenizer(config, tokenizer)
        model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])
        model.resize_token_embeddings(len(tokenizer))
    elif config['goal'] == 'eval':
        tokenizer = AutoTokenizer.from_pretrained(config['weight_file_path'])
        model = AutoModelForSeq2SeqLM.from_pretrained(config['weight_file_path'])
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config['model_name'])

    train_dataset, test_dataset = create_dataset(config)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    args = Seq2SeqTrainingArguments(
        # f"{config['model_name']}-finetuned-{config['source_lang']}-to-{config['target_lang']}_{config['train_part']}",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        save_strategy="epoch",
        save_total_limit=1,
        metric_for_best_model='sacrebleu',
        greater_is_better=True,
        load_best_model_at_end=True,
        use_cpu=False if config['device'] != 'cpu' else True,
        # warmup_ratio=0.06,
        learning_rate=config['lr'],
        lr_scheduler_type="cosine",
        seed=3407,
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        # weight_decay=0.001,
        num_train_epochs=config['num_train_epochs'],
        output_dir=output_dir,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    if 'nllb' in config['model_name']:
        source_lang = 'zho_Hans'
        target_lang = 'fra_Latn'
    elif 'mbart' in config['model_name']:
        source_lang = 'zh_CN'
        target_lang = 'fr_XX'
    elif 'mrebel' in config['model_name']:
        source_lang = 'zh_CN'
        target_lang = 'fr_XX'
    elif 'm2m100' in config['model_name']:
        source_lang = 'zh'
        target_lang = 'fr'

    if config['goal'] == 'benchmark':
        translator = pipeline('translation',
                              model=model,
                              tokenizer=tokenizer,
                              src_lang=source_lang,
                              tgt_lang=target_lang,
                              max_length=config['max_token_num'],
                              device=config['device'])
        trans_dataset_ch = [x[config['source_lang']] for x in test_dataset['translation']]
        trans_dataset_fr = [x[config['target_lang']] for x in test_dataset['translation']]
        output_translator = [x['translation_text'] for x in translator(trans_dataset_ch)]
        metric = evaluate.load("sacrebleu")
        result = metric.compute(predictions=output_translator, references=trans_dataset_fr)
        print(config['model_name'])
        print(result)
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=get_evaluator(tokenizer)
        )
        print(trainer.model.device)
        if config['goal'] == 'train':
            trainer.train()
        elif config['goal'] == 'eval':
            print(trainer.evaluate())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_lang", default='ch')
    parser.add_argument("--target_lang", default='fr')
    parser.add_argument("--model_name", default='facebook/mbart-large-50-many-to-many-mmt')
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--goal", default='train', choices=['train', 'eval', 'benchmark'])
    parser.add_argument("--output_dir", default='data/')
    parser.add_argument("--max_token_num", default=256, type=int)
    parser.add_argument("--num_train_epochs", default=1000, type=int)
    parser.add_argument("--train_file_path", default='data/data_temp_train.csv')
    parser.add_argument("--test_file_path", default='data/data_temp_test.csv')
    parser.add_argument("--weight_file_path", default=None)
    parser.add_argument("--job_name", default='m2m100_sent')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=4e-5, type=float)
    config = vars(parser.parse_args())
    model_name_list = [
                       'facebook/mbart-large-50-many-to-many-mmt',
                       'facebook/nllb-200-distilled-600M',
                       'Babelscape/mrebel-large',
                       'facebook/m2m100_418M'
                       ]
    # train(config)
    time_now = datetime.now().strftime("%m%d-%H:%M")
    if config['goal'] == 'train':
        store_path = '/Utilisateurs/wsun01/DiCo_log/train_log/' + time_now
    else:
        store_path = '/Utilisateurs/wsun01/DiCo_log/eva_log/' + time_now

    config['output_dir'] = store_path
    for model_name in model_name_list:
        config['model_name'] = model_name
        config['job_name'] = model_name.split('/')[-1]
        executor = submitit.AutoExecutor(folder=store_path)  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
        executor.update_parameters(
            job_name=config['job_name'],
            timeout_min=2160 * 4,
            gpus_per_node=1,
            cpus_per_task=5,
            mem_gb=30,
            slurm_partition='gpu-a6000',
            slurm_additional_parameters={
                'nodelist': 'l3icalcul07'
            }
        )
        executor.submit(train, config)

    '''
    facebook/mbart-large-50-many-to-many-mmt,
    facebook/nllb-200-distilled-600M
    Babelscape/mrebel-large
    facebook/m2m100_418M
    '''
