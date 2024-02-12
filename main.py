import pandas
from model.analyzor import Analyzor
import argparse

def dico(text_path,
         voca_path,
         device,
         check_point,
         pu_dict_path):

    with open(text_path, 'r', encoding='utf-8') as file:
        suorce_text = file.read()
    model = Analyzor(text_path,
                     voca_path,
                     device,
                     check_point,
                     pu_dict_path)
    model.pu_detect_static(source_text=suorce_text)
    model.translate(traget_lang='fra_Latn',
                    source_lang='zho_Hans',
                    source_text='今风尘碌碌，一事无成')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", default='data/text/dream_of_the_red_chambre.txt')
    parser.add_argument("--voca_path", default=None)
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--check_point", default='facebook/nllb-200-distilled-600M')
    parser.add_argument("--pu_dict_path", default='data/voca/pu.csv')
    args = parser.parse_args()
    text_path = args.text_path
    voca_path = args.voca_path
    device = args.device
    check_point = args.check_point
    pu_dict_path = args.pu_dict_path
    dico(text_path=text_path,
         voca_path=voca_path,
         device=device,
         check_point=check_point,
         pu_dict_path=pu_dict_path)


