import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset
#定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self,str):
        self.dataset = load_from_disk('./data/ChnSentiCorp')[str]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label
dataset = Dataset('train')

print(dataset[0])

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')     
print(token)
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    #编码---使用自带的批量成对编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,      
                                   truncation=True,         
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    #将分词后的结果取出来
    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels


#数据加载器
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=16,     #决定与训练模型的单次训练句子
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,labels) in enumerate(loader):
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
           token_type_ids=token_type_ids)   
print(out.last_hidden_state.shape)          

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)       
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():       
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])      #将预训练模型抽取数据中的特征（情感分类只取第0个词的特征）交给全连接层计算
        out = out.softmax(dim=1)       
        return out






modela = Model()
modela.load_state_dict(torch.load('./model/modelpara.pth'))


def predo():
    modela.eval()    #把dropout和BN层置为验证模式。
    correct = 0
    total = 0
    print("训练集：",end=" ")
    print(type(Dataset('validation')))
    loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
                                              batch_size=32,        #每一次测试的批量大小是32
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
                        token_type_ids=token_type_ids)     
        print(out.shape,out)        
        out = out.argmax(dim=1)   
        print(out.shape,out)        #torch.Size([32])
        correct += (out == labels).sum().item()
        print(out == labels)
        print(labels.shape,labels)    
        total += len(labels)
        print("第"+str(i)+"轮次准确率"+str(correct / total))

    print(correct / total)

predo()
