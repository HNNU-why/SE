from transformers import AdamW
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
token = BertTokenizer.from_pretrained('bert-base-chinese') 
print(token)


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 编码---使用自带的批量成对编码
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

print(len(loader))  # 批处理大小batch_size*loder的大小等于训练集train的长度

# 加载预训练模型
pretrained = BertModel.from_pretrained(
    'bert-base-chinese')  # 通过name加载网络上的与训练模型，并且缓存到cache中

for param in pretrained.parameters():
    param.requires_grad_(False)

# 模型试算
out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids)  
# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 只包括一个全连接的神经网络，输入是与训练模型的768维度向量，输出是二
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():  
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        # 将预训练模型抽取数据中的特征（情感分类只取第0个词的特征）交给全连接层计算
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)  # 指定第1个维度的和为1
        return out  # torch(32,1)


model = Model()
model.load_state_dict(torch.load('./model/model_new.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(model(input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids).shape)


# 训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()

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

        print("第" + str(i) + "轮  ",
              "损失率是" + str(loss.item()),
              "准确率是" + str(accuracy))
    if i % 100 == 0:
        torch.save(model.state_dict(), './model/model_new.pth')
        print("保存成功！")
    if i == 300:
        break

torch.save(model.state_dict(), './model/model_new.pth')
print("保存成功！")
