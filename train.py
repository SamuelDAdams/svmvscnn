from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets import load_dataset
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
#from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from torch.nn import functional as func
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from os import path, makedirs
import nets
import matplotlib.pyplot as plt
#import hiddenlayer as hl
#import graphviz

# def build_cnn(config):
#     if config == 1:

def gpu_to_cpu(loc, cls):
    model = cls()
    model = torch.load(loc, map_location=torch.device('cpu'))
    torch.save(model, loc)



def graph(loss_vals, model_name, dataset_name):
    epochs = [x[0] for x in loss_vals]
    loss = [x[1] for x in loss_vals]
    val_loss = [x[2] for x in loss_vals]
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Training info for ' + model_name +' on ' + dataset_name)
    plt.legend()
    plt.savefig('graphs/' + model_name + 'loss')
    plt.clf()

def train_svm(X_tr, y_tr, X_te, y_te):
    svc = LinearSVC(verbose=True)
    svc.fit(X_tr, y_tr)
    svm_predictions = svc.predict(X_te)
    print('SVM_tfidf_acc = {}'.format(accuracy_score(y_te, svm_predictions)))
    print('SVM_tfidf confusion matrix:')
    print(confusion_matrix(y_te, svm_predictions))

def train_model(model, opt, train_load, val_load, test_load, epochs, model_name=""):
    print('Starting training')
    epoch_losses = []
    best_val_loss = 1000
    for epoch in range(epochs):
        model.train()
        losses = []
        lossf = nn.CrossEntropyLoss()
        for _, batch in enumerate(train_load):
            embed_gpu, lab_gpu = tuple(i.to(device) for i in batch)
            opt.zero_grad()
            logits = model(embed_gpu)
            loss = lossf(logits, lab_gpu)
            losses.append(loss.item() / len(batch))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        epoch_loss = np.mean(losses)
        print('Training Loss: {}'.format(epoch_loss))
        val_loss = validate_model(model, val_load, 'Validation')
        epoch_losses.append((epoch, epoch_loss, val_loss))
        graph(epoch_losses, model_name, 'amazon_us_reviews')
        if val_loss < best_val_loss:
            torch.save(model, 'checkpoint.pth')
    validate_model(model, test_load, 'Test')
    torch.save(model, 'models/' + model_name + '.pth')
    gpu_to_cpu('checkpoint.pth', nets.Net6)
    gpu_to_cpu('models/' + model_name + '.pth', nets.Net6)
    return epoch_losses


def validate_model(model, loader, set_name):
    model.eval()
    acc = []
    losses = []
    for batch in loader:
        gpu_embed, gpu_lab = tuple(i.to(device) for i in batch)

        with torch.no_grad():
            logits = model(gpu_embed)
        lossf = nn.CrossEntropyLoss()
        loss = lossf(logits, gpu_lab)
        losses.append(loss.item()/len(batch))
        _, predictions = torch.max(logits, dim=1)
        accuracy = (predictions == gpu_lab).cpu().numpy().mean()
        acc.append(accuracy)
    print('{} loss: {}'.format(set_name, np.mean(losses)))
    print('{} acc: {}'.format(set_name, np.mean(acc)))
    return np.mean(losses)

device = 'cuda'
gpu_to_cpu('checkpoint.pth', nets.Net6)
# device = 'cpu'
#torch.autograd.set_detect_anomaly(True)
DATASET_NAME = 'amazon_us_reviews'
SUBSET = 'Digital_Software_v1_00'
if not path.exists('train/embeddings.npy'):
    distilbert = SentenceTransformer('stsb-distilbert-base', device='cuda')
    dataset = load_dataset(DATASET_NAME, SUBSET)
    data = dataset['train']['review_body']
    labels_sav = dataset['train']['star_rating']
    amounts = []
    for cl in np.unique(labels_sav):
        amounts.append((cl, labels_sav.count(cl)))
    print(amounts)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels_sav, stratify=labels_sav, test_size=0.2)

    train_bin_labels = list(map(lambda x: 1 if x > 3 else 0, train_labels))
    test_bin_labels = list(map(lambda x: 1 if x > 3 else 0, test_labels))

    train_distilbert_embed = distilbert.encode(train_data, show_progress_bar=False)  #progress bar crashes on sentence-transformers=0.4.1, fixed in 0.4.1.2, but not yet available on conda
    test_distilbert_embed = distilbert.encode(test_data, show_progress_bar=False)
    makedirs('train', exist_ok=True)
    makedirs('test', exist_ok=True)
    np.save('train/text.npy', train_data, allow_pickle=True)
    np.save('train/embeddings.npy', train_distilbert_embed, allow_pickle=True)
    np.save('train/bin_labels.npy', train_bin_labels, allow_pickle=True)
    np.save('train/labels.npy', train_labels, allow_pickle=True)

    np.save('test/text.npy', test_data, allow_pickle=True)
    np.save('test/embeddings.npy', test_distilbert_embed, allow_pickle=True)
    np.save('test/bin_labels.npy', test_bin_labels, allow_pickle=True)
    np.save('test/labels.npy', test_labels, allow_pickle=True)

train_embeddings = np.load('train/embeddings.npy', allow_pickle=True)
train_labels = np.load('train/bin_labels.npy', allow_pickle=True)

test_embeddings = np.load('test/embeddings.npy', allow_pickle=True)
test_labels = np.load('test/bin_labels.npy', allow_pickle=True)
# vectorizer = TfidfVectorizer()
# tfidf_data = vectorizer.fit_transform(lines)
# dataset = None

# X_tfidf_train, X_tfidf_test, X_embed_train, X_embed_test, y_train, y_test = \
#     train_test_split(tfidf_data, distilbert_embeddings, labels, test_size=0.2)
# X_embed_train, X_embed_val, y_embed_train, y_embed_val = \
#     train_test_split(X_embed_train, y_train, test_size=0.2)
X_embed_train, X_embed_test, = \
    tuple(torch.tensor(dat) for dat in [train_embeddings, test_embeddings])
y_train_tensor, y_test_tensor = \
    tuple(torch.tensor(dat, dtype=torch.long) \
    for dat in [train_labels, test_labels])

X_train = TensorDataset(X_embed_train, y_train_tensor)
train_sampler = RandomSampler(X_train)
train_loader = DataLoader(X_train, sampler=train_sampler, batch_size=50)

test_data = TensorDataset(X_embed_test, y_test_tensor)
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=50)



#net = nets.Net1(filter_sizes=[2,3,4], filter_amount=10, dropout=.1,  classes=5)
# train_svm(X_tfidf_train, y_train, X_tfidf_test, y_test)
# svc = LinearSVC(verbose=True, class_weight='balanced')
net = nets.Net6()
#arch = hl.build_graph(net, torch.zeros(1,1,768))
#arch.save("architecture", format="jpg")
net.to(device)
optimizer = AdamW(net.parameters(), lr=0.002)
train_model(net, optimizer, train_loader, test_loader, test_loader, 100, model_name='CNN')



# Turns out svm is kinda bad at this, acc 0.6174
# scaler = StandardScaler()
# norm_embeddings = scaler.fit(distilbert_embeddings)
# X_embed_norm_train = scaler.transform(X_embed_train)
# X_embed_norm_test = scaler.transform(X_embed_test)
# svc = LinearSVC(verbose=True, class_weight='balanced')
# svc.fit(X_embed_norm_train, y_train)
# svm_embed_predictions = svc.predict(X_embed_norm_test)
# print('SVM_embed_acc = {}'.format(accuracy_score(y_test, svm_embed_predictions)))
# print('SVM_embed confusion matrix:')

print('ok')
