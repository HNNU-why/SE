from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset

my_dataset = load_from_disk('./ChnSentiCorp')['train']  # 获取 train 集
my_dataset = my_dataset.remove_columns('label')
print(my_dataset)
# print("原始的数据:", my_dataset[1:3], end='\n')  # 打印 2 个例子看看
# my_dataset.to_csv(path_or_buf='./ChnSentiCorp/train/save_csv_data.csv')  # 导出为 csv 格式
# csv_dataset = load_dataset(path='csv', data_files='./ChnSentiCorp/train/save_csv_data.csv', split='train')  # 加载 csv 格式数据
# print("csv 格式的数据:", csv_dataset[1:3], end='\n')  # 打印 2 个例子看看

my_dataset.to_json(path_or_buf='./ChnSentiCorp/train/save_json_data.json')  # 导出为 json 格式
json_dataset = load_dataset(path='json', data_files='./ChnSentiCorp/train/save_json_data.json', split='train')  # 加载 json 格式数据
print("json 格式的数据:", json_dataset[1], json_dataset[30],json_dataset[60],json_dataset[90],end='\n')  # 打印 2 个例子看看
print(json_dataset)
# arrow_data = json_dataset.to

