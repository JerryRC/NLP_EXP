
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

from dataloader import get_loader


MNAME = "bert-base-uncased"

TRAINFILE = "semeval/semeval_train.txt"
TESTFILE = "semeval/semeval_test.txt"
VALFILE = "semeval/semeval_val.txt"

LR = 1e-5

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class Classifier(torch.nn.Module):
    def __init__(self, classes=19):
        super(Classifier, self).__init__()
        self.linear = torch.nn.Linear(768, classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, data):
        logit = self.linear(data['sent'])
        loss = self.loss(logit, data['label'])
        res = torch.argmax(logit, dim=-1)
        return loss, res


def main():

    # 读取数据
    tokenizer = BertTokenizer.from_pretrained(MNAME)
    train_loader, _ = get_loader(tokenizer, TRAINFILE)
    val_loader, classes = get_loader(tokenizer, VALFILE)

    # 加载模型
    model = BertModel.from_pretrained(MNAME).to(device)
    mlp = Classifier(classes=classes).to(device)
    
    # model.eval()
    # model.load_state_dict(torch.load("ckpt/bert_baseline_e{}_it{}.pth".format(9,11)))
    parameters_to_optimize = []
    parameters_to_optimize = list(model.named_parameters()) 
    parameters_to_optimize += list(mlp.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters_to_optimize, lr=LR)

    acc = 0.0
    reslist = []
    labellist = []
    best_acc = 0.0
    for e in range(10):
        model.train()
        mlp.train()
        print(f"Epoch {e+1}")
        tbar = tqdm(train_loader, total=len(train_loader), ncols=100, disable=False)
        for it, batch_data in enumerate(tbar):
            optimizer.zero_grad()
            
            input_sent = batch_data["sent"].to(device)
            bz, _ = input_sent.size()
            attention_mask = (input_sent != tokenizer.pad_token_id).bool().reshape(bz,-1).to(device) 
            
            # with torch.no_grad():
            embeddings = model(input_sent.reshape(bz,-1), attention_mask=attention_mask).last_hidden_state[:,0,:]
            
            # 将嵌入向量送入mlp
            loss, res = mlp({'sent': embeddings, 'label': batch_data['label'].to(device)})
            
            loss.backward()
            optimizer.step()

            reslist.append(res)
            labellist.append(batch_data['label'].to(device))

            tbar.set_postfix(loss=loss.item(), acc=acc)

            if it % 4 == 0 or it == len(tbar) - 1:
                acc = torch.sum(torch.cat(reslist) == torch.cat(labellist)).item() / len(torch.cat(reslist))

                model.eval()
                mlp.eval()
                s = False
                with torch.no_grad():
                    test_acc = 0.0
                    test_reslist = []
                    test_labellist = []
                    for _, batch_test in enumerate(val_loader):
                        input_sent = batch_test["sent"].to(device)
                        bz, _ = input_sent.size()
                        attention_mask = (input_sent != tokenizer.pad_token_id).bool().reshape(bz,-1).to(device) 
                        embeddings = model(input_sent.reshape(bz,-1), attention_mask=attention_mask).last_hidden_state[:,0,:]
                        loss, res = mlp({'sent': embeddings, 'label': batch_test['label'].to(device)})
                        test_reslist.append(res)
                        test_labellist.append(batch_test['label'].to(device))
                    
                    test_acc = torch.sum(torch.cat(test_reslist) == torch.cat(test_labellist)).item() / len(torch.cat(test_reslist))

                    if test_acc > best_acc and e > 3:
                        best_acc = test_acc
                        torch.save(model.state_dict(), "ckpt/bert_baseline_e{}_it{}.pth".format(e+1,it+1))
                        torch.save(mlp.state_dict(), "ckpt/mlp_baseline_e{}_it{}.pth".format(e+1,it+1))
                        s = True

                    with open("bert_baseline_log.txt", "a") as f:
                        f.write("Epoch {} Iter {} Train Acc {:.4f} Test Acc {:.4f}\n".format(e+1, it+1, acc, test_acc))
                        if s:
                            f.write("======Best Model Saved======\n")
                    s = False
                        
                    

        reslist = []
        labellist = []


def test():
    # 读取数据
    tokenizer = BertTokenizer.from_pretrained(MNAME)
    test_loader, classes = get_loader(tokenizer, TESTFILE)

    # 加载模型
    model = BertModel.from_pretrained(MNAME).to(device)
    mlp = Classifier(classes=classes).to(device)

    model.load_state_dict(torch.load("ckpt/bert_baseline_e{}_it{}.pth".format(6,37)))
    mlp.load_state_dict(torch.load("ckpt/mlp_baseline_e{}_it{}.pth".format(6,37)))
    model.eval()
    mlp.eval()

    acc = 0.0
    reslist = []
    labellist = []
    with torch.no_grad():
        for it, batch_data in enumerate(test_loader):
            input_sent = batch_data["sent"].to(device)
            bz, _ = input_sent.size()
            attention_mask = (input_sent != tokenizer.pad_token_id).bool().reshape(bz,-1).to(device) 
            embeddings = model(input_sent.reshape(bz,-1), attention_mask=attention_mask).last_hidden_state[:,0,:]
            loss, res = mlp({'sent': embeddings, 'label': batch_data['label'].to(device)})
            reslist.append(res)
            labellist.append(batch_data['label'].to(device))

    acc = torch.sum(torch.cat(reslist) == torch.cat(labellist)).item() / len(torch.cat(reslist))
    print("Iter {} Acc {:.4f}".format(it+1, acc))


def train_mlp():
    # 读取数据
    tokenizer = BertTokenizer.from_pretrained(MNAME)
    train_loader, _ = get_loader(tokenizer, TRAINFILE)
    val_loader, classes = get_loader(tokenizer, VALFILE)

    # 加载模型
    model = BertModel.from_pretrained(MNAME).to(device)
    mlp = Classifier(classes=classes).to(device)
    
    model.eval()
    model.load_state_dict(torch.load("ckpt/bert_baseline_e{}_it{}.pth".format(6,37)))

    train_loader = retrieval(model, tokenizer, train_loader)
    val_loader = retrieval(model, tokenizer, val_loader)

    parameters_to_optimize = []
    parameters_to_optimize += list(mlp.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    parameters_to_optimize = [
        {'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(parameters_to_optimize, lr=1e-3)

    acc = 0.0
    reslist = []
    labellist = []
    best_acc = 0.0
    for e in range(1000):
        mlp.train()
        print(f"Epoch {e+1}")
        tbar = tqdm(train_loader, total=len(train_loader), ncols=100, disable=False)
        for it, batch_data in enumerate(tbar):
            optimizer.zero_grad()
            
            embeddings = batch_data["sent"].to(device)
            # 将嵌入向量送入mlp
            loss, res = mlp({'sent': embeddings, 'label': batch_data['label'].to(device)})
            
            loss.backward()
            optimizer.step()

            reslist.append(res)
            labellist.append(batch_data['label'].to(device))

            tbar.set_postfix(loss=loss.item(), acc=acc)

            if it % 4 == 0 or it == len(tbar) - 1:
                acc = torch.sum(torch.cat(reslist) == torch.cat(labellist)).item() / len(torch.cat(reslist))

                model.eval()
                mlp.eval()
                s = False
                with torch.no_grad():
                    test_acc = 0.0
                    test_reslist = []
                    test_labellist = []
                    for _, batch_test in enumerate(val_loader):
                        embeddings = batch_test["sent"].to(device)
                        loss, res = mlp({'sent': embeddings, 'label': batch_test['label'].to(device)})
                        test_reslist.append(res)
                        test_labellist.append(batch_test['label'].to(device))
                    
                    test_acc = torch.sum(torch.cat(test_reslist) == torch.cat(test_labellist)).item() / len(torch.cat(test_reslist))

                    # if test_acc > best_acc:
                    #     best_acc = test_acc
                    #     torch.save(model.state_dict(), "bert_baseline_e{}_it{}.pth".format(e+1,it+1))
                    #     torch.save(mlp.state_dict(), "mlp_baseline_e{}_it{}.pth".format(e+1,it+1))
                    #     s = True

                    # with open("bert_baseline_log.txt", "a") as f:
                    #     f.write("Epoch {} Iter {} Train Acc {:.4f} Test Acc {:.4f}\n".format(e+1, it+1, acc, test_acc))
                    #     if s:
                    #         f.write("======Best Model Saved======\n")
                    # s = False
        print("Epoch {} Iter {} Train Acc {:.4f} Test Acc {:.4f}".format(e+1, it+1, acc, test_acc))         
                    

        reslist = []
        labellist = []


def retrieval(sentence_encoder, tokenizer, data_loader):
    sentence_encoder.eval()

    with torch.no_grad():
        tbar = tqdm(data_loader, total=len(data_loader), disable=False, desc="get_train_emb", ncols=100)
        data_logreg = []
        for data_item in tbar:
            if torch.cuda.is_available():
                data_item['sent'] = data_item['sent'].to(device)

            input_sent = data_item["sent"]
            bz, _ = input_sent.size()
            attention_mask = (input_sent != tokenizer.pad_token_id).bool().reshape(bz, -1).to(device) 
            data_emb_batch = sentence_encoder(input_sent.reshape(bz, -1), attention_mask=attention_mask).last_hidden_state[:, 0, :]

            batch_data = {'sent': data_emb_batch.cpu(), 'label': data_item['label']}
            data_logreg.append(batch_data)

    return data_logreg


if __name__ == "__main__":
    # main()
    # test()
    train_mlp()
