import os
import collections
import pandas as pd
import numpy as np
import warnings
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

# --- Custom CSL Implementations (Replicating Phase 2-B) ---
def tempered_weights(y, alpha=0.5):
    counts = collections.Counter(y)
    max_count = max(counts.values())
    return {cls: (max_count / count)**alpha for cls, count in counts.items()}

class MetaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, cost_matrix=None, n_estimators=10):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(random_state=42)
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        bag = BaggingClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators, random_state=42)
        bag.fit(X, y)
        probs = bag.predict_proba(X)
        classes = bag.classes_
        
        y_relabeled = np.zeros(len(y), dtype=y.dtype)
        for i in range(len(y)):
            expected_costs = [probs[i, j] * self.cost_matrix.get(c, 1.0) for j, c in enumerate(classes)]
            y_relabeled[i] = classes[np.argmax(expected_costs)]
            
        self.final_estimator_ = clone(self.base_estimator)
        self.final_estimator_.fit(X, y_relabeled)
        self.classes_ = self.final_estimator_.classes_
        return self

    def predict(self, X):
        return self.final_estimator_.predict(X)

class AdaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, cost_matrix=None, n_estimators=50):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        self.model_ = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=42)
        sample_weight = np.array([self.cost_matrix.get(label, 1.0) for label in y])
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

class CSKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, cost_matrix=None, n_neighbors=5):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.y_train_ = np.array(y)
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_.fit(X, y)
        return self
        
    def predict(self, X):
        distances, indices = self.knn_.kneighbors(X)
        y_pred = []
        for i in range(len(X)):
            votes = {c: 0.0 for c in self.classes_}
            for j in range(self.n_neighbors):
                neighbor_class = self.y_train_[indices[i, j]]
                dist = distances[i, j]
                weight = 1.0 / (dist + 1e-5)
                votes[neighbor_class] += weight * self.cost_matrix.get(neighbor_class, 1.0)
            y_pred.append(max(votes, key=votes.get))
        return np.array(y_pred)

class Phase4_B_NativeCSL_Hierarchical:
    def __init__(self):
        self.algorithms = ['CS-SVM (Linear)', 'CS-SVM (RBF)', 'CS-DT', 'CS-LR', 'CS-RF', 'MetaCost', 'AdaCost', 'CS-KNN']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC20Functional20Non-Functional20Classification.csv"
        }

    def load_data(self, path):
        df = pd.read_csv(path, encoding='latin1')
        label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
        text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            
        df[label_col] = df[label_col].astype(str).str.strip()
        df['text'] = df[text_col].astype(str).fillna("")
        
        fr_labels = ['F', 'FR', 'Functional', 'functional', 'f']
        df['Stage1_Label'] = df[label_col].apply(lambda x: 'FR' if x in fr_labels else 'NFR')
        df['Stage2_Label'] = df[label_col]
        return df

    def get_real_embeddings(self, texts, embed_type):
        if embed_type == 'TF-IDF':
            vec = TfidfVectorizer(max_features=5000)
            return vec.fit_transform(texts).toarray()
        else:
            import sys
            p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
            if p2_path not in sys.path:
                sys.path.append(p2_path)
            from models.deep_embeddings import get_deep_embeddings
            
            # get_deep_embeddings returns X_train_emb, X_test_emb, _ 
            # We will pass the full text array as 'train' and 'test' just to get the full vectors back
            X_emb, _, _ = get_deep_embeddings(texts, texts, embed_type)
            if hasattr(X_emb, "toarray"):
                X_emb = X_emb.toarray()
            elif len(X_emb.shape) == 3:
                X_emb = X_emb.mean(axis=1) # Flatten sequences
            return X_emb

    def get_csl_model(self, algo_name, w_dict):
        if algo_name == 'CS-SVM (Linear)':
            return LinearSVC(class_weight=w_dict, random_state=42)
        elif algo_name == 'CS-SVM (RBF)':
            return SVC(kernel='rbf', class_weight=w_dict, probability=True, random_state=42)
        elif algo_name == 'CS-DT':
            return DecisionTreeClassifier(class_weight=w_dict, random_state=42)
        elif algo_name == 'CS-LR':
            return LogisticRegression(class_weight=w_dict, random_state=42, max_iter=1000)
        elif algo_name == 'CS-RF':
            return RandomForestClassifier(class_weight=w_dict, random_state=42)
        elif algo_name == 'MetaCost':
            return MetaCost(cost_matrix=w_dict, n_estimators=10)
        elif algo_name == 'AdaCost':
            return AdaCost(cost_matrix=w_dict, n_estimators=50)
        elif algo_name == 'CS-KNN':
            return CSKNN(cost_matrix=w_dict, n_neighbors=5)

    def run(self):
        print("=" * 50)
        print("PHASE 4-B: 2-STAGE HIERARCHICAL NATIVE CSL")
        print("=" * 50)
        
        results = []
        for dataset_name, path in self.dataset_paths.items():
            print(f"\n--- Processing Dataset: {dataset_name} ---")
            df = self.load_data(path)
            
            for embedding in self.embeddings:
                print(f"  -> Generating {embedding} Vectors...")
                X = self.get_real_embeddings(df['text'].tolist(), embedding)
                y1 = df['Stage1_Label'].values
                y2 = df['Stage2_Label'].values
                
                for algo in self.algorithms:
                    print(f"    [*] 5-Fold CV: {algo} + {embedding}")
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    fold_accs = []
                    
                    for train_idx, test_idx in skf.split(X, y2): # Stratify by multi-class
                        X_train, X_test = X[train_idx], X[test_idx]
                        y1_train, y1_test = y1[train_idx], y1[test_idx]
                        y2_train, y2_test = y2[train_idx], y2[test_idx]
                        
                        # --- STAGE 1: FR vs NFR ---
                        w1 = tempered_weights(y1_train, alpha=0.5)
                        model_stage1 = self.get_csl_model(algo, w1)
                        model_stage1.fit(X_train, y1_train)
                        stage1_preds = model_stage1.predict(X_test)
                        
                        # --- STAGE 2: Multi-class NFRs ---
                        nfr_mask = (y1_train == 'NFR')
                        if nfr_mask.sum() > 0:
                            X_train_nfr = X_train[nfr_mask]
                            y2_train_nfr = y2_train[nfr_mask]
                            
                            w2 = tempered_weights(y2_train_nfr, alpha=0.5)
                            model_stage2 = self.get_csl_model(algo, w2)
                            model_stage2.fit(X_train_nfr, y2_train_nfr)
                        else:
                            model_stage2 = None
                        
                        # --- HIERARCHICAL PREDICTION ---
                        final_preds = []
                        for i, p1 in enumerate(stage1_preds):
                            if p1 == 'FR':
                                final_preds.append('FR') # Assume original FR class mapping or label
                            else:
                                if model_stage2 is not None:
                                    final_preds.append(model_stage2.predict(X_test[i].reshape(1, -1))[0])
                                else:
                                    final_preds.append('Unknown_NFR')
                                    
                        # We must map True FR back to the dataset's specific FR label ('F' or 'FR' etc)
                        # To keep it simple, we check if true label is FR, but we evaluate on exact y2.
                        # Need to standardize: if final_pred == 'FR', map to the dataset's FR label.
                        fr_label = df[df['Stage1_Label'] == 'FR']['Stage2_Label'].iloc[0] if len(df[df['Stage1_Label'] == 'FR']) > 0 else 'F'
                        mapped_preds = [fr_label if p == 'FR' else p for p in final_preds]
                        
                        fold_accs.append(accuracy_score(y2_test, mapped_preds))
                        
                    avg_acc = np.mean(fold_accs)
                    results.append({'Phase': 'Phase 4-B', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': avg_acc})
        
        res_df = pd.DataFrame(results)
        res_df.to_csv('FINAL_Phase4_B_Native_CSL_Results.csv', index=False)
        print("\n[SUCCESS] Phase 4-B (Native CSL Hierarchical) complete. Results saved to CSV.")

if __name__ == "__main__":
    Phase4_B_NativeCSL_Hierarchical().run()
