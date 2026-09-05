import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

class Phase4_1A_Production:
    def __init__(self):
        self.algorithms = ['BiCNN', 'BiLSTM', 'CNN', 'DNN', 'GRU', 'LSTM']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC20Functional20Non-Functional20Classification.csv"
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 50
        self.batch_size = 32

    def load_data(self, path):
        df = pd.read_csv(path, encoding='latin1')
        label_col = None
        for col in df.columns:
            if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']:
                label_col = col
                break
        if not label_col:
            label_col = df.columns[-1] 
            
        text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            
        df[label_col] = df[label_col].astype(str).str.strip()
        df['text'] = df[text_col].astype(str).fillna("")
        
        fr_labels = ['F', 'FR', 'Functional', 'functional', 'f']
        df['Stage1_Label'] = df[label_col].apply(lambda x: 0 if x in fr_labels else 1)
        df['Global_Class'] = pd.factorize(df[label_col])[0]
        return df, label_col

    def get_real_embeddings(self, texts, embedding_name):
        print(f"      -> Generating REAL {embedding_name} Vectors...")
        if embedding_name == 'TF-IDF':
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(texts).toarray()
            return torch.tensor(X, dtype=torch.float32), X.shape[1]
        elif embedding_name == 'SBERT':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            X = model.encode(texts)
            return torch.tensor(X, dtype=torch.float32), X.shape[1]
        elif embedding_name == 'BERT':
            model = SentenceTransformer('bert-base-uncased')
            X = model.encode(texts)
            return torch.tensor(X, dtype=torch.float32), X.shape[1]
        elif embedding_name == 'MPNet':
            model = SentenceTransformer('all-mpnet-base-v2')
            X = model.encode(texts)
            return torch.tensor(X, dtype=torch.float32), X.shape[1]
        elif embedding_name == 'Word2Vec':
            import gensim.downloader as api
            print("         (Loading real Word2Vec Google News 300... cached)")
            wv = api.load('word2vec-google-news-300')
            X = []
            for text in texts:
                words = text.split()
                vecs = [wv[w] for w in words if w in wv]
                X.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
            return torch.tensor(np.array(X), dtype=torch.float32), 300
        elif embedding_name == 'GloVe':
            import gensim.downloader as api
            print("         (Loading real GloVe Wiki 300... cached)")
            gl = api.load('glove-wiki-gigaword-300')
            X = []
            for text in texts:
                words = text.split()
                vecs = [gl[w] for w in words if w in gl]
                X.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
            return torch.tensor(np.array(X), dtype=torch.float32), 300

    def get_dl_model(self, algo_name, input_dim, is_stage1, num_classes):
        out_dim = 1 if is_stage1 else num_classes
        activation_fn = nn.Sigmoid() if is_stage1 else nn.Softmax(dim=1)
        
        class MasterNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.algo = algo_name
                
                # 1. DNN Architecture
                if self.algo == 'DNN':
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU()
                    )
                    self.fc = nn.Linear(64, out_dim)
                    
                # 2. RNN Architectures (LSTM, GRU, BiLSTM)
                elif self.algo in ['LSTM', 'GRU', 'BiLSTM']:
                    bi = (self.algo == 'BiLSTM')
                    if self.algo in ['LSTM', 'BiLSTM']:
                        self.rnn = nn.LSTM(input_dim, 64, batch_first=True, bidirectional=bi)
                    else:
                        self.rnn = nn.GRU(input_dim, 64, batch_first=True, bidirectional=bi)
                    self.fc = nn.Linear(128 if bi else 64, out_dim)
                    
                # 3. CNN Architectures (CNN, BiCNN)
                elif self.algo in ['CNN', 'BiCNN']:
                    self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
                    if self.algo == 'BiCNN':
                        # Simulate BiCNN with multiple kernel sizes parallel extraction
                        self.conv2 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
                    self.fc = nn.Linear(128 if self.algo == 'BiCNN' else 64, out_dim)
                    
            def forward(self, x):
                if self.algo == 'DNN':
                    feat = self.net(x)
                elif self.algo in ['LSTM', 'GRU', 'BiLSTM']:
                    x = x.unsqueeze(1) # shape: (batch, 1, input_dim)
                    out, _ = self.rnn(x)
                    feat = out.squeeze(1) # shape: (batch, hidden_dim)
                elif self.algo in ['CNN', 'BiCNN']:
                    x = x.unsqueeze(1) # shape: (batch, 1, input_dim)
                    c1 = torch.relu(self.conv1(x))
                    c1 = torch.max(c1, dim=2)[0] # Global max pool
                    if self.algo == 'BiCNN':
                        c2 = torch.relu(self.conv2(x))
                        c2 = torch.max(c2, dim=2)[0]
                        feat = torch.cat((c1, c2), dim=1)
                    else:
                        feat = c1
                return activation_fn(self.fc(feat))
                
        return MasterNet().to(self.device)

    def train_model(self, model, dataloader, loss_fn):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = model(batch_x)
                if isinstance(loss_fn, nn.BCELoss):
                    preds = preds.squeeze()
                    batch_y = batch_y.float()
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_pipeline(self, s1_model, s2_model, X_test, y_stage1_test, y_stage2_test):
        s1_model.eval()
        s2_model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(self.device)
            s1_preds = s1_model(X_test_device).squeeze()
            s1_pred_classes = (s1_preds > 0.5).long().cpu()
            s1_acc = accuracy_score(y_stage1_test.numpy(), s1_pred_classes.numpy())
            
            nfr_mask = (y_stage1_test == 1)
            X_test_nfr = X_test[nfr_mask].to(self.device)
            y_stage2_true_nfr = y_stage2_test[nfr_mask]
            s2_acc = 0.0
            s2_pred_classes_full = torch.zeros(len(X_test), dtype=torch.long) - 1 
            
            if len(X_test_nfr) > 0:
                s2_preds = s2_model(X_test_nfr)
                s2_pred_nfr_classes = torch.argmax(s2_preds, dim=1).cpu()
                s2_acc = accuracy_score(y_stage2_true_nfr.numpy(), s2_pred_nfr_classes.numpy())
                
                predicted_nfr_mask = (s1_pred_classes == 1)
                X_test_predicted_nfr = X_test[predicted_nfr_mask].to(self.device)
                if len(X_test_predicted_nfr) > 0:
                    pipeline_s2_preds = s2_model(X_test_predicted_nfr)
                    pipeline_s2_pred_classes = torch.argmax(pipeline_s2_preds, dim=1).cpu()
                    s2_pred_classes_full[predicted_nfr_mask] = pipeline_s2_pred_classes
            
            correct_fr = ((y_stage1_test == 0) & (s1_pred_classes == 0)).sum().item()
            correct_nfr = ((y_stage1_test == 1) & (s1_pred_classes == 1) & (y_stage2_test == s2_pred_classes_full)).sum().item()
            overall_acc = (correct_fr + correct_nfr) / len(X_test)
        return s1_acc, s2_acc, overall_acc

    def run_grid(self):
        print("==================================================")
        print("PHASE 4-1A: CORRECTED ARCHITECTURES (TRUE LSTM/CNN/GRU)")
        print("==================================================\n")
        
        results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for dataset_name, path in self.dataset_paths.items():
            print(f"--- Processing Dataset: {dataset_name} ---")
            df, label_col = self.load_data(path)
            
            nfr_df = df[df['Stage1_Label'] == 1].copy()
            nfr_df['Stage2_Label'] = pd.factorize(nfr_df[label_col])[0]
            num_nfr_classes = nfr_df['Stage2_Label'].nunique()
            
            df['Stage2_Label'] = -1
            df.loc[nfr_df.index, 'Stage2_Label'] = nfr_df['Stage2_Label']
            
            for embedding in self.embeddings:
                try:
                    X_tensor, input_dim = self.get_real_embeddings(df['text'].tolist(), embedding)
                except Exception as e:
                    print(f"      [!] Failed to load {embedding}: {e}")
                    continue
                
                for algo in self.algorithms:
                    print(f"    [*] Running 5-Fold CV: {algo} + {embedding}")
                    s1_fold_accs, s2_fold_accs, overall_fold_accs = [], [], []
                    
                    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, df['Global_Class'])):
                        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
                        y_s1_train, y_s1_test = torch.tensor(df.iloc[train_idx]['Stage1_Label'].values, dtype=torch.long), torch.tensor(df.iloc[test_idx]['Stage1_Label'].values, dtype=torch.long)
                        y_s2_train, y_s2_test = torch.tensor(df.iloc[train_idx]['Stage2_Label'].values, dtype=torch.long), torch.tensor(df.iloc[test_idx]['Stage2_Label'].values, dtype=torch.long)
                        
                        s1_loader = DataLoader(TensorDataset(X_train, y_s1_train), batch_size=self.batch_size, shuffle=True)
                        s1_model = self.get_dl_model(algo, input_dim, True, 2)
                        s1_model = self.train_model(s1_model, s1_loader, nn.BCELoss())
                        
                        train_nfr_mask = (y_s1_train == 1)
                        s2_loader = DataLoader(TensorDataset(X_train[train_nfr_mask], y_s2_train[train_nfr_mask]), batch_size=self.batch_size, shuffle=True)
                        s2_model = self.get_dl_model(algo, input_dim, False, num_nfr_classes)
                        s2_model = self.train_model(s2_model, s2_loader, nn.CrossEntropyLoss())
                        
                        s1_acc, s2_acc, overall_acc = self.evaluate_pipeline(s1_model, s2_model, X_test, y_s1_test, y_s2_test)
                        s1_fold_accs.append(s1_acc)
                        s2_fold_accs.append(s2_acc)
                        overall_fold_accs.append(overall_acc)

                    results.append({'Phase': 'Stage 1', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Fold 0': s1_fold_accs[0], 'Fold 1': s1_fold_accs[1], 'Fold 2': s1_fold_accs[2], 'Fold 3': s1_fold_accs[3], 'Fold 4': s1_fold_accs[4], 'Average': np.mean(s1_fold_accs)})
                    results.append({'Phase': 'Stage 2', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Fold 0': s2_fold_accs[0], 'Fold 1': s2_fold_accs[1], 'Fold 2': s2_fold_accs[2], 'Fold 3': s2_fold_accs[3], 'Fold 4': s2_fold_accs[4], 'Average': np.mean(s2_fold_accs)})
                    results.append({'Phase': 'Overall', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Fold 0': overall_fold_accs[0], 'Fold 1': overall_fold_accs[1], 'Fold 2': overall_fold_accs[2], 'Fold 3': overall_fold_accs[3], 'Fold 4': overall_fold_accs[4], 'Average': np.mean(overall_fold_accs)})

        pd.DataFrame(results).to_csv('FINAL_Phase4_1A_Real_Results.csv', index=False)
        print("\n[SUCCESS] Phase 4-1A CORRECTED ARCHITECTURES complete.")

if __name__ == "__main__":
    Phase4_1A_Production().run_grid()
