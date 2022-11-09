from transformers import AdamW
from transformers import BertModel
import torch
from datasets import load_from_disk
from transformers import BertTokenizer
from datasets import load_dataset
# 定义数据集


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
token = BertTokenizer.from_pretrained('bert-base-chinese') 
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

# 加载预训练模型
pretrained = BertModel.from_pretrained(
    'bert-base-chinese')  # 通过name加载网络上的与训练模型，并且缓存到cache中

for param in pretrained.parameters():
    param.requires_grad_(False)

# 模型试算
out = pretrained(input_ids=input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids)  # 将tokenizer分词器得到的结果参数直接喂给预训练模型

# 使用bert抽取特征，然后再做下游任务的迁移学习
# 定义下游任务模型

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
        out = self.fc1(out.last_hidden_state[:, 0])
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = out.softmax(dim=1)  # 指定第1个维度的和为1
        return out  # torch(32,1)

model = Model()
model.load_state_dict(torch.load('./model/3-DNN_model.pth'))

# 训练
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):  
    out = model(input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids) 

    loss = criterion(out, labels) 
    loss.backward() 
    optimizer.step()  # 优化
    optimizer.zero_grad()  # 归0梯度

    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print("第" + str(i) + "轮  ",
              "loss: " + str(loss.item()),
              "accuracy: " + str(accuracy))

    if i % 50 == 0 and i != 0:
        torch.save(model.state_dict(), './model/3-DNN_model.pth')
        print("保存成功！")
    if i == 600:
        break

torch.save(model.state_dict(), './model/3-DNN_model.pth')
print("保存成功！")

# modeltest = Model()
# modeltest.load_state_dict(torch.load('./model/modelpara.pth'))
# #测试
# def test():
#     modeltest.eval()
#     correct = 0
#     total = 0
#
#     loader_test = torch.utils.data.DataLoader(dataset=Dataset('validation'),
#                                               batch_size=32,
#                                               collate_fn=collate_fn,
#                                               shuffle=True,
#                                               drop_last=True)
#
#     for i, (input_ids, attention_mask, token_type_ids,
#             labels) in enumerate(loader_test):
#
#         if i == 5:
#             break
#
#         print(i)
#
#         with torch.no_grad():
#             out = model(input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         token_type_ids=token_type_ids)
#
#         out = out.argmax(dim=1)
#         correct += (out == labels).sum().item()
#         total += len(labels)
#
#     print(correct / total)
# test()
