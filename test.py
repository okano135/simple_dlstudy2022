# python        3.9.7
# pytorch       1.10.2
# torchtext     0.11.2
# mkl           2018.0.2 (In my environment(M1 Mac), the latest version(2022,0,0) caused an INTEL MKL ERROR)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split='train')

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

train_iter, test_iter = AG_NEWS()
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
criterion = torch.nn.CrossEntropyLoss()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

model = TextClassificationModel(vocab_size, emsize, num_class)
# print(model)

BATCH_SIZE = 64

Model_path = '/raid/okano/model/txt_cls_t/2022_02_25_08_54_50'
Model_name = '/cls_t_prms_7.pth'
model.load_state_dict(torch.load(Model_path+Model_name, map_location=device))
print(f'loaded {Model_path}{Model_name}')
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

test_dataset = to_map_style_dataset(test_iter)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))

# test news articles
#
# 1. CNN "Fame and fury: China's wildly different reactions to US-born Olympians"
# https://edition.cnn.com/2022/02/11/china/eileen-gu-zhu-yi-nathan-chen-comparison-intl-hnk-dst/index.html
#
# 2. CNN "At least 7 British citizens and 1 American are being held by the Taliban in Afghanistan"
# https://edition.cnn.com/2022/02/11/politics/westerners-held-taliban/index.html
#
# 3. CNN "Don't panic if you got a scary IRS notice"
# https://edition.cnn.com/2022/02/11/success/irs-tax-letters-suspended/index.html
#
# 4. CNN "The end of a wild pandemic ride: What it was like for Peloton employees who lost their jobs this week"
# https://edition.cnn.com/2022/02/11/tech/peloton-layoffs/index.html
#

ex_text_str = [
"In the span of a week, three American-born athletes of Chinese descent have been thrust into the spotlight at the Beijing Winter Olympics -- to very different reactions in China.\
All three were trained in the United States and are only a few years apart in age, but their paths diverged on the way to the Games: freestyle skier Eileen Gu and figure skater Zhu Yi chose to compete for China, while Nathan Chen, another figure skater, opted for Team USA.\
Gu and Chen both won gold, while Zhu faltered on the ice during two consecutive showings. \
The public responses they've received in the Olympic host nation also took different turns."
,
"At least eight Westerners have been arrested by the Taliban in Afghanistan during different incidents in the last two months, CNN has learned, marking a sharp escalation of Taliban actions against Westerners living in the country.\
No formal charges appear to have been lodged against the detained men. \
They include seven British citizens including one who is an American legal resident and one US citizen, according to the sources with direct knowledge of the matter in Afghanistan, the United States, and the UK.\
The former vice president of Afghanistan, Amrullah Saleh tweeted that \"nine\" Westerners had been \"kidnapped\" by the Taliban, naming journalists Andrew North, formerly of the BBC who was in the country working for the United Nations and Peter Jouvenal, who has worked with the BBC and CNN, both are British citizens.\
The reason for each specific detention is unclear, and they are not thought all to be related.\
Jouvenal's detention was confirmed by his family and friends to CNN."
,
"Imagine having filed and paid your taxes last year, then months later you get a letter in the mail from the IRS saying you didn't.\
That's what's happening to many taxpayers this year thanks to automated notices being sent by the IRS.\
But if you got one, don't panic. \
There's a fair chance the IRS simply hasn't seen what you already sent in. \
That's because it's dealing with a mountain of returns and correspondence that has built up over the past two years. \
During that time, the agency was called on to deliver several rounds of economic impact payments and other financial Covid-19 relief, while trying to protect its own workforce from Covid."
,
"Late Monday night, some Peloton (PTON) staffers noticed they were unable to access work productivity apps like Slack and Okta, which they used regularly on the job. \
Peloton's employees had been told about a scheduled maintenance window that might cause service outages, according to one employee, but that didn't stop others from bracing for the worst.\
\"I'm freaking out,\" another former Peloton employee who worked in the company's product department recalled to CNN Business. \
He said coworkers frantically texted each other as the\
y speculated about what the morning might bring. \
Peloton was reporting its earnings Tuesday, and weeks earlier the CEO said the company was reviewing its costs and that layoffs were on the table."
]

for i in range(len(ex_text_str)):
    print('\n' + ex_text_str[i] + '\n')
    print("This is a %s news\n" %ag_news_label[predict(ex_text_str[i], text_pipeline)])