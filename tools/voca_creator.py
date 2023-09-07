import pandas as pd

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

if __name__ == "__main__":
    create_voca()