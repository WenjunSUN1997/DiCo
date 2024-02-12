import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead

class Analyzor():
    def __init__(self,
                 text_path,
                 voca_path,
                 device,
                 check_point,
                 pu_dict_path):
        self.text_path = text_path
        self.voca_path = voca_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(check_point)
        self.model = AutoModelWithLMHead.from_pretrained(check_point).to(device)
        self.pu_dict = pd.read_csv(pu_dict_path)

    def translate(self, traget_lang, source_lang, source_text):
        output_tokeniz = self.tokenizer.prepare_seq2seq_batch([source_text],
                                                              return_tensors="pt")
        for key, value in output_tokeniz.items():
            output_tokeniz[key] = value.to(self.device)

        result_trans = self.model.generate(**output_tokeniz)
        text_trans = self.tokenizer.batch_decode(result_trans,
                                                 skip_special_tokens=True)
        return text_trans

    def pu_detect_static(self, source_text):
        pu_list = self.pu_dict['item'].tolist()
        element_counts = {}
        for item in pu_list:
            # 使用字符串的count方法来计算元素在字符串中出现的次数
            count = source_text.count(item)
            # 将元素和对应的出现次数存储到字典中
            element_counts[item] = count

        result = {}

        for index in range(len(element_counts)):
            if element_counts[pu_list[index]] != 0:
                result[pu_list[index]] = {'fre': element_counts[pu_list[index]],
                                          'explanation': self.pu_dict['explanation'][index]}

        return result






