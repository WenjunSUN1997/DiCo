### Requirements
requirements.txt

### Run the fine-tuned model 
```shell
python train.py --goal eval --device [your_device] --test_file_path [path_to_test_file] --weight_file_path [path_to_model]
```

### Fine-tune the translation model
```shell
python train.py 

--source_lang ch #the input language

--target_lang fr #the output language

--goal train

--model_name google/mt5-base #the name of language model

--device cuda:0 #the device to ruin your code, cpu or cuda

--output_dir data/ #the folder to store the log and fine-tuned model

--max_token_num 216 #the number of tokens onput to the modle 

--num_train_epochs 1000 #the number of training epoches

--train_file_path data/train.csv #the path to training sentence file

--test_file_path data/test.csv #the path to test sentence file

--batch_size 16 #the batch size of training and test

--lr 4e-5 #the learning rate
```

### Other setings
If you wanna use slurm cluster --job_name is to set the job name.
If not, ignore it. 
