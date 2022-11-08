import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset
#定义本地数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self,str):
        self.dataset = load_from_disk('./data/ChnSentiCorp')[str]  #加载本地数据

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']
        return text, label
      
dataset = Dataset('train')
print(dataset[0])

#加载字典和分词工具
token = BertTokenizer.from_pretrained('bert-base-chinese')      #加载预分词器
print(token)
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,      #传递数据的格式有所变化
                                   truncation=True,         #达到最大长度后截断                            
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   #返回length，标识长度
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
                                     batch_size=16,     
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)

for i, (input_ids, attention_mask, token_type_ids,labels) in enumerate(loader):
    break

print(len(loader))
from transformers import BertModel

#加载预训练模型
pretrained = BertModel.from_pretrained('bert-base-chinese')     

for param in pretrained.parameters():
    param.requires_grad_(False)

#模型试算
out = pretrained(input_ids=input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids)   
print(out.last_hidden_state.shape)         

#使用bert抽取特征，然后再做下游任务的迁移学习
#定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)      
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():       #该方法表示当前计算不需要反向传播
            out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])     
        out = out.softmax(dim=1)        
        return out


model = Model()

print(model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)

from transformers import AdamW

#训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()     #计算以下损失，反向更新

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)

    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)

        print("第"+str(i)+"轮  ", "损失率是"+str(loss.item()), "准确率是"+str(accuracy))

    if i == 300:
        break

torch.save(model.state_dict(),'./model/modelpara.pth')
print("保存成功！")

