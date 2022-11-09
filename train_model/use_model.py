import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self, str):
        self.dataset = load_from_disk('../data/ChnSentiCorp')[str].remove_columns('label')
        print(self.dataset, self.dataset.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        return text
dataset = Dataset('train')

print(dataset[0])

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')      #加载预分词器
print(token)
def collate_fn(data):
    sents = [i[0] for i in data]


    #编码---使用自带的批量成对编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,      #传递数据的格式有所变化
                                   truncation=True,         #达到最大长度后截断
                                   #达到最大长度后截断
                                   padding='max_length',
                                   max_length=500,
                                   #返回pytorch类型，默认是返回list
                                   return_tensors='pt',
                                   #返回length，标识长度
                                   return_length=True)
    #将分词后的结果取出来
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']


    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids


#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,     #决定与训练模型的单次训练句子
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader):
    break

print(len(loader))
from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')     #通过name加载网络上的与训练模型，并且缓存到cache中

#不训练,不需要计算梯度：不会修改预训练模型的参数，只训练下游任务模型
for param in pretrained.parameters():
    param.requires_grad_(False)

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)   #将tokenizer分词器得到的结果参数直接喂给预训练模型
print(out.last_hidden_state.shape)          #输出最后一个隐藏层的参数类型 --torch.Size(16,500,768)   16:batchSize,每一批训练16个句子，500是句子长度（多则截断，短则补长），768是词编码的维度，即每个词编码成768维度的向量

#使用bert抽取特征，然后再做下游任务的迁移学习
#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)       #只包括一个全连接的神经网络，输入是与训练模型的768维度向量，输出是二
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():       #该方法表示当前计算不需要反向传播
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])      #将预训练模型抽取数据中的特征（情感分类只取第0个词的特征）交给全连接层计算
        out = out.softmax(dim=1)        #指定第1个维度的和为1
        return out


model = Model()

print(model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)


modela = Model()
modela.load_state_dict(torch.load('./model/modelpara.pth'))


dataset = Dataset('validation')       #得到没有标签的数据集

def predo():
    modela.eval()    #把dropout和BN层置为验证模式。
    correct = 0
    total = 0


    loader_test = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=32,        #每一次测试的批量大小是32
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader_test):

        if i == 5:
            break

        print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)      #测试也是通过与训练模型提取特征
        print(out.shape,out)        #torch.Size([32, 2]),即每个句子被分为0和1的概率
        out = out.argmax(dim=1)     #返回指定维度最大值的序号  0 or 1 ，即最终分析的句子是正面的还是反面的
        print(out.shape,out)        #torch.Size([32])

        correct += (out).sum().item()
        total += 32

    print(correct / total)          #correct / total决定最后的情绪类型


predo()