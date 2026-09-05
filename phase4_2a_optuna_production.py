import os
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_sample_weight
import optuna
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

class Phase4_2A_Classical_Optuna:
    def __init__(self):
        self.algorithms = ['LinearSVC', 'RF', 'LR', 'NB', 'XGBoost', 'KNN']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC20Functional20Non-Functional20Classification.csv"
        }
        self.n_optuna_trials = 10 # Rapid tuning to prevent massive runtimes

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
        print(f"      -> Generating {embedding_name} Vectors (Cached across algorithms)...")
        if embedding_name == 'TF-IDF':
            vectorizer = TfidfVectorizer(max_features=5000)
            X = vectorizer.fit_transform(texts).toarray()
            return X
        elif embedding_name == 'SBERT':
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model.encode(texts)
        elif embedding_name == 'BERT':
            model = SentenceTransformer('bert-base-uncased')
            return model.encode(texts)
        elif embedding_name == 'MPNet':
            model = SentenceTransformer('all-mpnet-base-v2')
            return model.encode(texts)
        elif embedding_name == 'Word2Vec':
            import gensim.downloader as api
            wv = api.load('word2vec-google-news-300')
            X = []
            for text in texts:
                words = text.split()
                vecs = [wv[w] for w in words if w in wv]
                X.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
            return np.array(X)
        elif embedding_name == 'GloVe':
            import gensim.downloader as api
            gl = api.load('glove-wiki-gigaword-300')
            X = []
            for text in texts:
                words = text.split()
                vecs = [gl[w] for w in words if w in gl]
                X.append(np.mean(vecs, axis=0) if vecs else np.zeros(300))
            return np.array(X)

    def get_optuna_model(self, algo_name, trial):
        if algo_name == 'LinearSVC':
            c = trial.suggest_loguniform('C', 1e-3, 1e2)
            return LinearSVC(C=c, class_weight='balanced', random_state=42)
        elif algo_name == 'RF':
            n_est = trial.suggest_int('n_estimators', 50, 200)
            depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
            return RandomForestClassifier(n_estimators=n_est, max_depth=depth, class_weight='balanced', random_state=42)
        elif algo_name == 'LR':
            c = trial.suggest_loguniform('C', 1e-3, 1e2)
            return LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42)
        elif algo_name == 'NB':
            var_smooth = trial.suggest_loguniform('var_smoothing', 1e-9, 1e-2)
            return GaussianNB(var_smoothing=var_smooth)
        elif algo_name == 'XGBoost':
            n_est = trial.suggest_int('n_estimators', 50, 200)
            depth = trial.suggest_int('max_depth', 3, 10)
            lr = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
            return XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        elif algo_name == 'KNN':
            n_neigh = trial.suggest_int('n_neighbors', 3, 15)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            return KNeighborsClassifier(n_neighbors=n_neigh, weights=weights)

    def optimize_hyperparameters(self, algo_name, X, y):
        def objective(trial):
            model = self.get_optuna_model(algo_name, trial)
            # Use 3-fold CV for fast Optuna evaluation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            # Handle XGBoost custom sample weights for CSL
            if algo_name == 'XGBoost':
                scores = []
                for train_idx, test_idx in skf.split(X, y):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_te, y_te = X[test_idx], y[test_idx]
                    
                    # Convert to labels starting from 0 for XGBoost
                    unique_labels = np.unique(y_tr)
                    mapping = {val: idx for idx, val in enumerate(unique_labels)}
                    y_tr_mapped = np.vectorize(mapping.get)(y_tr)
                    y_te_mapped = np.vectorize(mapping.get)(y_te)
                    
                    sample_w = compute_sample_weight('balanced', y_tr_mapped)
                    model.fit(X_tr, y_tr_mapped, sample_weight=sample_w)
                    preds = model.predict(X_te)
                    scores.append(accuracy_score(y_te_mapped, preds))
                return np.mean(scores)
            else:
                return np.mean(cross_val_score(model, X, y, cv=skf, scoring='accuracy'))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_optuna_trials)
        
        # Return instantiated model with best params
        best_trial = study.best_trial
        # Re-instantiate model by mimicking a trial object
        class BestTrialWrapper:
            def __init__(self, best_params):
                self.best_params = best_params
            def suggest_loguniform(self, name, low, high): return self.best_params[name]
            def suggest_int(self, name, low, high): return self.best_params[name]
            def suggest_categorical(self, name, choices): return self.best_params[name]
            
        return self.get_optuna_model(algo_name, BestTrialWrapper(best_trial.params))

    def run_grid(self):
        print("==================================================")
        print("PHASE 4-2A: 2-STAGE HIERARCHICAL CLASSICAL ML (WITH OPTUNA)")
        print("==================================================\n")
        
        results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for dataset_name, path in self.dataset_paths.items():
            print(f"--- Processing Dataset: {dataset_name} ---")
            df, label_col = self.load_data(path)
            
            nfr_df = df[df['Stage1_Label'] == 1].copy()
            nfr_df['Stage2_Label'] = pd.factorize(nfr_df[label_col])[0]
            
            df['Stage2_Label'] = -1
            df.loc[nfr_df.index, 'Stage2_Label'] = nfr_df['Stage2_Label']
            
            for embedding in self.embeddings:
                try:
                    # Generate embedding once for all 6 algorithms to save massive time
                    X = self.get_real_embeddings(df['text'].tolist(), embedding)
                except Exception as e:
                    print(f"      [!] Failed to load {embedding}: {e}")
                    continue
                
                for algo in self.algorithms:
                    print(f"    [*] Optuna Tuning & 5-Fold CV: {algo} + {embedding}")
                    
                    # 1. OPTUNA TUNING
                    # Tune Stage 1 model
                    y_stage1 = df['Stage1_Label'].values
                    best_s1_model = self.optimize_hyperparameters(algo, X, y_stage1)
                    
                    # Tune Stage 2 model (only on NFR data)
                    nfr_mask_global = (y_stage1 == 1)
                    X_nfr_global = X[nfr_mask_global]
                    y_stage2_global = df['Stage2_Label'].values[nfr_mask_global]
                    best_s2_model = self.optimize_hyperparameters(algo, X_nfr_global, y_stage2_global)
                    
                    # 2. 5-FOLD CROSS VALIDATION
                    s1_fold_accs, s2_fold_accs, overall_fold_accs = [], [], []
                    
                    y_global = df['Global_Class'].values
                    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_global)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_s1_train, y_s1_test = y_stage1[train_idx], y_stage1[test_idx]
                        y_s2_train, y_s2_test = df['Stage2_Label'].values[train_idx], df['Stage2_Label'].values[test_idx]
                        
                        # Train Stage 1
                        if algo == 'XGBoost':
                            sw1 = compute_sample_weight('balanced', y_s1_train)
                            best_s1_model.fit(X_train, y_s1_train, sample_weight=sw1)
                        else:
                            best_s1_model.fit(X_train, y_s1_train)
                        s1_preds = best_s1_model.predict(X_test)
                        s1_acc = accuracy_score(y_s1_test, s1_preds)
                        s1_fold_accs.append(s1_acc)
                        
                        # Train Stage 2
                        train_nfr_mask = (y_s1_train == 1)
                        X_train_nfr = X_train[train_nfr_mask]
                        y_s2_train_nfr = y_s2_train[train_nfr_mask]
                        
                        if algo == 'XGBoost':
                            # XGBoost requires labels 0 to num_classes-1
                            unique_s2 = np.unique(y_s2_train_nfr)
                            map_s2 = {val: idx for idx, val in enumerate(unique_s2)}
                            y_s2_train_mapped = np.vectorize(map_s2.get)(y_s2_train_nfr)
                            sw2 = compute_sample_weight('balanced', y_s2_train_mapped)
                            best_s2_model.fit(X_train_nfr, y_s2_train_mapped, sample_weight=sw2)
                        else:
                            best_s2_model.fit(X_train_nfr, y_s2_train_nfr)
                            
                        # Evaluate Stage 2 and Overall Pipeline
                        test_nfr_mask = (y_s1_test == 1)
                        X_test_nfr = X_test[test_nfr_mask]
                        y_s2_test_nfr = y_s2_test[test_nfr_mask]
                        
                        s2_acc = 0.0
                        s2_pred_classes_full = np.zeros(len(X_test)) - 1
                        
                        if len(X_test_nfr) > 0:
                            s2_nfr_preds = best_s2_model.predict(X_test_nfr)
                            if algo == 'XGBoost':
                                # Map back to original labels
                                inv_map_s2 = {idx: val for val, idx in map_s2.items()}
                                s2_nfr_preds = np.vectorize(inv_map_s2.get)(s2_nfr_preds)
                            s2_acc = accuracy_score(y_s2_test_nfr, s2_nfr_preds)
                            
                            # End-to-end pipeline predictions
                            predicted_nfr_mask = (s1_preds == 1)
                            X_test_predicted_nfr = X_test[predicted_nfr_mask]
                            if len(X_test_predicted_nfr) > 0:
                                pipeline_s2_preds = best_s2_model.predict(X_test_predicted_nfr)
                                if algo == 'XGBoost':
                                    pipeline_s2_preds = np.vectorize(inv_map_s2.get)(pipeline_s2_preds)
                                s2_pred_classes_full[predicted_nfr_mask] = pipeline_s2_preds
                                
                        s2_fold_accs.append(s2_acc)
                        
                        correct_fr = np.sum((y_s1_test == 0) & (s1_preds == 0))
                        correct_nfr = np.sum((y_s1_test == 1) & (s1_preds == 1) & (y_s2_test == s2_pred_classes_full))
                        overall_acc = (correct_fr + correct_nfr) / len(X_test)
                        overall_fold_accs.append(overall_acc)

                    results.append({'Phase': 'Stage 1', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': np.mean(s1_fold_accs)})
                    results.append({'Phase': 'Stage 2', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': np.mean(s2_fold_accs)})
                    results.append({'Phase': 'Overall', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': np.mean(overall_fold_accs)})

        pd.DataFrame(results).to_csv('FINAL_Phase4_2A_Classical_Optuna_Results.csv', index=False)
        print("\n[SUCCESS] Phase 4-2A (Optuna Tuned Classical ML) complete.")

if __name__ == "__main__":
    Phase4_2A_Classical_Optuna().run_grid()
