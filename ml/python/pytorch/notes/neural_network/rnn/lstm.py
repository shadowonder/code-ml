# 导入所需库
import torch
import numpy as np
from torch import nn, optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext import data, datasets
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)
device = torch.device("mps")


# 获取spacy分词器
tokenizer = get_tokenizer("spacy", "en_core_web_sm")

# 创建词汇表
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

train_iter, test_iter = datasets.IMDB()
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<pad>'])
vocab.set_default_index(vocab['<pad>'])

# 自定义数据处理函数
def data_process(raw_text_iter):
    data = []
    for label, text in raw_text_iter:
        tokens = tokenizer(text)
        data.append((torch.tensor([vocab[token] for token in tokens], dtype=torch.long), int(label)))

    return data

# 获取训练和测试数据
train_data = data_process(train_iter)
test_data = data_process(test_iter)

# 创建DataLoader
def collate_batch(batch):
    text_list, label_list = zip(*batch)
    text_lengths = [len(text) for text in text_list]
    padded_text_list = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True).transpose(0, 1)
    return torch.tensor(label_list, dtype=torch.float).to("mps"), padded_text_list.to("mps"), torch.tensor(text_lengths, dtype=torch.long).to("mps")

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

# 定义LSTM模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

rnn = RNN(len(vocab), 100, 256)

# 优化器和损失函数
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss().to(device)
rnn.to(device)

# 准确率计算函数
def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

# 训练函数
def train(rnn, iterator, optimizer, criteon):
    avg_acc = []
    rnn.train()

    for i, batch in enumerate(iterator):
        labels, padded_texts, text_lengths = batch
        pred = rnn(padded_texts).squeeze(1)
        loss = criteon(pred, labels)
        acc = binary_acc(pred, labels).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    print('avg acc:', avg_acc)

# 评估函数
def eval(rnn, iterator, criteon):
    avg_acc = []
    rnn.eval()

    with torch.no_grad():
        for batch in iterator:
            labels, padded_texts, text_lengths = batch
            pred = rnn(padded_texts).squeeze(1)
            acc = binary_acc(pred, labels).item()
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()
    print('>>test:', avg_acc)

# 开始训练和评估
for epoch in range(10):
    print(f'Epoch {epoch+1}:')
    train(rnn, train_loader, optimizer, criteon)
    eval(rnn, test_loader, criteon)