from train import train
import submitit
import torch

# def task():
#     config = {'task': 'doc_ner',
#               'type': 'unmasked',
#               'device': 'cuda:0',
#               'batch_size': 2,
#               'max_token_num': 1024,
#               'half': True,
#               'lr': 0.0001,
#               'weight': True,
#               'sim_dim': 4096,
#               'model_name': 'MAGAer13/mplug-owl-llama-7b',
#               'dataset_name': 'nielsr/funsd-layoutlmv3'}
#     train(config)

def task():
    # print(torch.cuda.device_count())
    config = {'device': 'cpu',
              'max_token_num': 512,
              'word_file_path': r'data/word.csv',
              'train_file_path': r'data/word.csv',
              'model_name': 'google/mt5-base',
              'target_lang': 'fr',
              'source_lang': 'ch',
              'batch_size': 8,
              }
    train(config)

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder='/Utilisateurs/wsun01/logs/')  # Can specify cluster='debug' or 'local' to run on the current node instead of on the cluster
    executor.update_parameters(
        # timeout_min=12000,
        nodes=1,
        gpus_per_node=0,
        tasks_per_node=1,
        cpus_per_task=10,
        # mem_gb=80,
        slurm_partition='general',
        slurm_additional_parameters={
            # 'gres': 'gpu:A6000:1',
            'nodelist': 'l3icalcul07'
        })
    executor.submit(task)