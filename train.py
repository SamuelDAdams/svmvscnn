from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets import load_dataset
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from torch.nn import functional as func
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from os import path
import nets
import matplotlib.pyplot as plt

# def build_cnn(config):
#     if config == 1:

def graph(loss_vals, model_name, dataset_name):
    epochs = [x[0] for x in loss_vals]
    loss = [x[1] for x in loss_vals]
    plt.plot(epochs, loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss for '+model_name +' on ' + dataset_name)

def checkpaths(dataset_name, subset_name):
    base_name = ''
    if subset_name != '':
        base_name = dataset_name + '_'+ subset_name
    return path.exists(base_name + '_embeddings.npy') and path.exists(base_name + '_labels.npy') 

def train_svm(X_tr, y_tr, X_te, y_te):
    svc = LinearSVC(verbose=True)
    svc.fit(X_tr, y_tr)
    svm_predictions = svc.predict(X_te)
    print('SVM_tfidf_acc = {}'.format(accuracy_score(y_te, svm_predictions)))
    print('SVM_tfidf confusion matrix:')
    print(confusion_matrix(y_te, svm_predictions))

def train_model(model, opt, train_load, val_load, epochs):
    print('Starting training')
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        losses = []
        lossf = nn.CrossEntropyLoss()
        for _, batch in enumerate(train_load):
            embed_gpu, lab_gpu = tuple(i.to('cuda') for i in batch)
            model.zero_grad()
            logits = model(embed_gpu)
            loss = lossf(logits, lab_gpu)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        epoch_loss = sum(losses)/len(train_load)
        epoch_losses.append(epoch, epoch_loss)
        print('Training Loss: {}'.format(epoch_loss))

        validate_model(model, val_load)
    return epoch_losses


def validate_model(model, val_load):
    model.eval()
    acc = []
    losses = []
    for batch in val_load:
        gpu_embed, gpu_lab = tuple(i.to('cuda') for i in batch)

        with torch.no_grad():
            logits = model(gpu_embed)
        lossf = nn.CrossEntropyLoss()
        loss = lossf(logits, gpu_lab)
        losses.append(loss.item())
        _, predictions = torch.max(logits, dim=1)
        accuracy = (predictions == gpu_lab).cpu().numpy().mean() * 100
        acc.append(accuracy)
    print('Val loss: {}'.format(np.mean(losses)))
    print('Val acc: {}'.format(np.mean(acc)))

#torch.autograd.set_detect_anomaly(True)
datasets = {'amazon_us_reviews' : ['Digital_Software_v1_00']}
distilbert = SentenceTransformer('stsb-distilbert-base', device='cuda')
for name in datasets.keys():
    for subset in datasets.get(name):
        dataset = None
        if not checkpaths(name, subset):
            if subset == []:
                dataset = load_dataset(name)
            else:
                dataset = load_dataset(name, subset)
            data = dataset['train']['review_body']
            labels_sav = dataset['train']['star_rating']
            labels_sav = list(map(lambda x: 1 if x > 2 else 0, labels_sav))

            distilbert_embed = distilbert.encode(data, show_progress_bar=False)  #progress bar crashes on sentence-transformers=0.4.1, fixed in 0.4.1.2, but not yet available on conda
            base_name = name + '_'+ subset
            np.save(base_name + '_embeddings.npy', distilbert_embed, allow_pickle=True)
            np.save(base_name + '_labels.npy', labels_sav, allow_pickle=True)
        
        base_name = name + '_' + subset
        dataset = load_dataset(name, subset)
        lines = dataset['train']['review_body']
        distilbert_embeddings = np.load(base_name + '_embeddings.npy', allow_pickle=True)
        labels = np.load(base_name + '_labels.npy', allow_pickle=True)
        vectorizer = TfidfVectorizer()
        tfidf_data = vectorizer.fit_transform(lines)
        dataset = None

        X_tfidf_train, X_tfidf_test, X_embed_train, X_embed_test, y_train, y_test = train_test_split(tfidf_data, distilbert_embeddings, labels, test_size=0.2)
        X_embed_train, X_embed_test, = tuple(torch.tensor(dat) for dat in [X_embed_train, X_embed_test])
        y_train_tensor, y_test_tensor =  tuple(torch.tensor(dat, dtype=torch.long) for dat in [y_train, y_test])

        X_train = TensorDataset(X_embed_train, y_train_tensor)
        train_sampler = RandomSampler(X_train)
        train_loader = DataLoader(X_train, sampler=train_sampler, batch_size=50)

        val_data = TensorDataset(X_embed_test, y_test_tensor)
        val_sampler = SequentialSampler(val_data)
        val_loader = DataLoader(val_data, sampler=val_sampler, batch_size=50)

        
        
        #net = nets.Net1(filter_sizes=[2,3,4], filter_amount=10, dropout=.1,  classes=5) 
        net = nets.Net2(dropout=.3, channels=16, kernels=[2, 4, 8, 16, 32, 64], classes=2)
        net.to('cuda')
        optimizer = AdamW(net.parameters(), lr=0.001)
        train_model(net, optimizer, train_loader, val_loader, 15)
        train_svm(X_tfidf_train, y_train, X_tfidf_test, y_test)
        #svc = LinearSVC(verbose=True, class_weight='balanced')
        

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
