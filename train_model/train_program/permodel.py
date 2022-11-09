from transformers import BertModel
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, str):
        self.dataset = load_from_disk('../data/ChnSentiCorp')[str]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label
dataset = Dataset('train')
print(dataset[0])

# 加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载预分词器
print(token)


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents, 
                                   truncation=True,  
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels

# 数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=32, 
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    break
print(len(loader))

# 加载预训练模型
pretrained = BertModel.from_pretrained(
    'bert-base-chinese')  # 通过name加载网络上的与训练模型，并且缓存到cache中

for param in pretrained.parameters():
    param.requires_grad_(False)

# 模型试算
out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids)  # 将tokenizer分词器得到的结果参数直接喂给预训练模型

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        self.fc1 = torch.nn.Linear(768, 60)  # 三层全连接层
        self.relu1 = torch.nn.ReLU()
        # 第二层
        self.fc2 = torch.nn.Linear(60, 30)
        self.relu2 = torch.nn.ReLU()
        # 第三层
        self.fc3 = torch.nn.Linear(30, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():  # 该方法表示当前计算不需要反向传播
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        # 将预训练模型抽取数据中的特征（情感分类只取第0个词的特征）交给全连接层计算
        out = self.fc1(out.last_hidden_state[:, 0])
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = out.softmax(dim=1)  # 指定第1个维度的和为1
        return out  # torch(32,1)


modela = Model()
modela.load_state_dict(torch.load('./model/3-DNN_model.pth'))

def predo():
    modela.eval()  # 把dropout和BN层置为验证模式。
    correct = 0
    total = 0
    output_test = 0  # 模型判断出来的情感度
    actual_test = 0  # 真实标签的情感度
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('test'),
                                              batch_size=32,  # 每一次测试的批量大小是32
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        if i == 5:
            break
        print(i)
        with torch.no_grad():
            out = modela(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)  # 测试也是通过与训练模型提取特征
        print(out.shape, out)  # torch.Size([32, 2]),即每个句子被分为0和1的概率
        out = out.argmax(dim=1)  # 返回指定维度最大值的序号  0 or 1 ，即最终分析的句子是正面的还是反面的
        print(out.shape, out)  # torch.Size([32])

        correct += (out == labels).sum().item()
        output_test += (out).sum().item()
        actual_test += (labels).sum().item()

        print(out == labels)
        print(labels.shape, labels)  # torch.Size([32])
        total += len(labels)
        print("第" + str(i) + "批次模型测试的情感度为：", output_test / total)
        print("第" + str(i) + "批次真实的情感度为：", actual_test / total)
        print("第" + str(i) + "轮次准确率" + str(correct / total), correct, total)
    print(correct / total)
predo()
