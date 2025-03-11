import numpy as np
import pandas as pd
import os
from itertools import product
from collections import Counter
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # for convenience only
import time
import gurobipy as gp
from gurobipy import GRB

###############################################################################
# 1.a Kernels : Spectrum Kernel
###############################################################################

class Spectrum:
    """
    Spectrum Kernel Class.
    The kernel is defined by k-mer counting. For each sequence, we build a
    feature vector of k-mer counts, then the kernel is the dot product of
    these vectors.
    """
    def __init__(self, k=3):
        self.k = k
        self.kmer_list = None

    def _build_vocab(self, X):
        """
        Build a sorted list of all k-mers that appear in X.
        X is a list/array of DNA sequences.
        """
        kmer_set = set()
        for seq in X:
            for i in range(len(seq) - self.k + 1):
                kmer = seq[i:i+self.k]
                kmer_set.add(kmer)
        self.kmer_list = sorted(kmer_set)

    def transform(self, X):
        """
        Transform each sequence in X into a k-spectrum feature vector
        using the previously-built k-mer list (self.kmer_list).
        """
        from collections import Counter
        n_samples = len(X)
        n_kmers = len(self.kmer_list)
        feature_matrix = np.zeros((n_samples, n_kmers), dtype=float)

        for idx, seq in enumerate(X):
            kmer_counts = Counter(seq[i:i+self.k] for i in range(len(seq) - self.k + 1))
            for j, kmer in enumerate(self.kmer_list):
                feature_matrix[idx, j] = kmer_counts[kmer]
        return feature_matrix

    def kernel(self, X, Y,square=False):
        """
        Compute the Gram matrix between sets of sequences X and Y
        in the k-spectrum space.
        For the first call (training), we build the vocabulary from X.
        """
        # If we haven't built the vocabulary yet, build it from X
        if self.kmer_list is None:
            self._build_vocab(X)
        # Convert sequences to feature space
        X_feat = self.transform(X)
        Y_feat = self.transform(Y)
        # Return dot product
        return X_feat @ Y_feat.T

###############################################################################
# 1.b Kernels : Substring Kernel
###############################################################################

class SubstringKernel:
    """
    Implements a substring kernel (Lodhi et al.) with exponential decay for gaps.
    The parameter 'p' is the maximum substring length to consider,
    and 'lambda_' is the decay factor for gaps (0 < lambda_ < 1).
    """

    def __init__(self, p=3, lambda_=0.5):
        """
        p: maximum length of substring (subsequence) to consider
        lambda_: decay factor for gaps (0 < lambda_ < 1).
        """
        self.p = p
        self.lambda_ = lambda_

    def _k_substring(self, x, y):
        """
        Compute the substring kernel K_p(x, y) for two individual strings x and y.
        This uses a dynamic programming approach to accumulate the contribution
        of all common subsequences up to length p.
        
        Returns a float: the kernel value K_p(x, y).
        """
        n_x = len(x)
        n_y = len(y)
        p = self.p
        lam = self.lambda_

        # B matrices, where B[k][i][j] tracks partial sums for subsequences of length k

        B = [np.zeros((n_x , n_y)) for _ in range(p+1)]
        B[0][:, :] = 1.0

        # The main DP to fill up B[k]
        for k in range(1, p+1):
            for i in range(k-1, n_x):
                for j in range(k-1, n_y):
                    # B[k][i][j] depends on B[k][i-1][j], plus the new matches
                    B[k][i][j] = lam * B[k][i-1][j] + lam * B[k][i][j-1] - lam**2 * B[k][i-1][j-1]
                    
                    if x[i] == y[j]:
                        # If characters match, add lam^2 plus the contribution from B[k-1][i-1][j-1]
                        B[k][i][j] += lam**2 * B[k-1][i-1][j-1]
                        

        K = 0
        for i in range(p-1, n_x):
            for j in range(p-1, n_y):
                if x[i] == y[j]:
                    K += lam**2 * B[p-1][i-1][j-1]
        return K

    def kernel(self, X, Y,square=False):
        """
        Compute the NxM kernel matrix for sets of sequences X and Y.
        X: list of strings (size N)
        Y: list of strings (size M)
        Returns an (N x M) NumPy array K where K[i,j] = substring kernel of X[i], Y[j].
        """
        N = len(X)
        M = len(Y)
        Kmat = np.zeros((N, M))
        if square:
            for i in range(N):
                for j in range(i, M):
                    Kmat[i, j] = self._k_substring(X[i], Y[j])
                    Kmat[j, i] = Kmat[i, j]
        else:
            for i in range(N):
                for j in range(M):
                    Kmat[i, j] = self._k_substring(X[i], Y[j])
        return Kmat


###############################################################################
# 2. Custom Kernel SVC (Soft-margin SVM in the Dual)
###############################################################################
class KernelSVC:
    """
    SVM with kernel for classification in {−1, +1}, using a custom solver.
    The dual form is:
       max_{alpha}  sum(alpha_i) - 0.5 * alpha^T (y_i y_j * K(i,j)) alpha
       subject to:  sum_i alpha_i y_i = 0
                    0 <= alpha_i <= C

    Implementation with Gurobi or some other QP solver.
    """

    def __init__(self, C=1.0, kernel=None, epsilon=1e-5):
        self.C = C
        # kernel can be a function kernel(X, X) or an object with .kernel()
        self.kernel = kernel
        self.epsilon = epsilon
        # Learned parameters
        self.alpha = None
        self.support_idx = None
        self.X = None     # Training data
        self.y = None     # Training labels
        self.b = 0.0

    def fit(self, X, y, K=None):
        """
        Fit SVM model using sequences X and labels y in {−1, +1}.

        :param X: training data (list of sequences or array)
        :param y: labels in {−1, +1}
        :param K: (optional) precomputed kernel matrix for X vs X, shape=(n,n).
                  If None, we will call self.kernel(X, X) to compute it.
        """
        import gurobipy as gp
        from gurobipy import GRB
        import time

        self.X = X
        self.y = y
        n = len(y)

        # 1) If K is None, compute Gram matrix with the kernel function
        if K is None:
            # kernel might be a method or an object
            K = self.kernel(X, X, square=True)

        # 2) Build Q = (y_i * y_j) * K[i, j]
        Q = (y.reshape(-1,1) * y.reshape(1,-1)) * K

        # 3) Build Gurobi model
        model = gp.Model("SVM-dual")
        alpha = model.addVars(n, lb=0.0, ub=self.C, name="alpha", vtype=GRB.CONTINUOUS)

        # sum_i alpha_i y_i = 0
        model.addConstr(gp.quicksum(alpha[i] * float(y[i]) for i in range(n)) == 0.0,
                        name="eq_y_alpha")

        # Objective: minimize(0.5 * alpha^T Q alpha - sum(alpha_i))
        quad_expr = gp.QuadExpr()
        for i in range(n):
            for j in range(n):
                if Q[i, j] != 0.0:
                    quad_expr.add(alpha[i] * alpha[j], Q[i, j] * 0.5)

        linear_expr = gp.quicksum(alpha[i] for i in range(n))
        objective_expr = quad_expr - linear_expr
        model.setObjective(objective_expr, GRB.MINIMIZE)

        model.setParam('OutputFlag', 0)  # optional: quiet solver
        # Solve
        start = time.time()
        model.optimize()
        end = time.time()
        print(f"KernelSVC fit done in {end - start:.2f} s")

        if model.status != GRB.OPTIMAL:
            print("Could not solve to optimality.")
            return False

        alpha_opt = np.array([alpha[i].X for i in range(n)])
        self.alpha = alpha_opt

        # 4) Identify support vectors
        self.support_idx = np.where(alpha_opt > self.epsilon)[0]

        # 5) Compute the bias b
        b_candidates = []
        for i in self.support_idx:
            if alpha_opt[i] < (self.C - self.epsilon):
                f_xi = 0.0
                for j in self.support_idx:
                    f_xi += alpha_opt[j] * y[j] * K[i, j]
                b_candidates.append(y[i] - f_xi)

        if len(b_candidates) > 0:
            self.b = np.mean(b_candidates)
        else:
            self.b = 0.0

        return True

    def decision_function(self, Xtest, Ktest=None):
        """
        Compute the SVM decision function f(x) = sum_j alpha_j y_j K(x, x_j) + b for each x in Xtest.
        :param Xtest: test data
        :param Ktest: optional precomputed kernel matrix of shape (len(Xtest), len(Xtrain))
        :return: array of shape (len(Xtest),)
        """
        # If Ktest not given, compute it
        if Ktest is None:
            # shape (n_test, n_train)
            Ktest = self.kernel(Xtest, self.X, square=False)

        # 1) Restrict alpha and y to support vectors, plus the columns of Ktest
        alpha_support = self.alpha[self.support_idx]      # shape (n_sv,)
        y_support = self.y[self.support_idx]              # shape (n_sv,)
        Ktest_support = Ktest[:, self.support_idx]        # shape (n_test, n_sv)

        # 2) Combine alpha and y via broadcasting
        #    multiply elementwise: alpha_j * y_j
        weights = alpha_support * y_support               # shape (n_sv,)

        # 3) Matrix multiply for the sum_i( alpha_j y_j Ktest[i,j] )
        #    f: shape (n_test,)
        f = Ktest_support.dot(weights)

        # 4) Add bias
        f += self.b

        return f

    def predict(self, Xtest, Ktest=None):
        """
        Predict labels in {−1, +1} for each example in Xtest.
        :param Xtest: test data
        :param Ktest: optional precomputed kernel matrix (n_test x n_train)
        """
        fvals = self.decision_function(Xtest, Ktest=Ktest)
        return np.sign(fvals)


###############################################################################
# 3.a Find Best (k,C) for Spectrum
###############################################################################

def transform_labels_01_to_pm1(y):
    """
    Transform y from {0,1} to {−1,+1}.
    """
    return 2*y - 1

def transform_labels_pm1_to_01(y_pm1):
    """
    Transform y from {−1,+1} to {0,1}.
    """
    return ((y_pm1 + 1 )/2).astype(int)

# Path setup
base_path = os.getcwd()
data_path = os.path.join(base_path, "datasets")

datasets = ["0","1", "2"]
k_values = {"0": [3,15,16,17],"1": [8],"2" :[8]} #[[3, 4, 5,8,10]] * 3  
C_values = {"0":[10**ex * cte for ex in range(-2,4) for cte in [1,3,5,7,9]],"1":[10**ex * cte for ex in range(-2,4) for cte in [1,3,5,7,9]],"2": [10**ex * cte for ex in range(-2,4) for cte in [1,3,5,7,9]]}#[[1, 10, 200]] * 3
best_kc_map = {}
# 3.a Find Best (k,C) for Spectrum

def search_best_k_C_spectrum(X_train, y_train, X_valid, y_valid, k_list, C_list, epsilon=1e-5):
    """
    Search the best (k, C) on the given (X_train, y_train, X_valid, y_valid).
    We only compute kernel matrices once for each k, then loop over C.
    """
    best_acc = 0.0
    best_k = None
    best_C = None

    for k in k_list:
        print(f"  Building kernel for k = {k}...")
        spec_kernel = Spectrum(k=k)

        print(f"  Building train kernel...")
        start = time.time()
        # Compute the NxN kernel on training
        K_train = spec_kernel.kernel(X_train, X_train, square=True)
        end = time.time()
        print(f"  Done in {end - start:.2f} s")
        # Compute the kernel for validation vs. training
        print(f"  Building valid kernel...")
        start = time.time()
        K_valid = spec_kernel.kernel(X_valid, X_train, square=False)
        end = time.time()
        print(f"  Done in {end - start:.2f} s")

        for C in C_list:
            print(f"    Training SVC for C={C} (k={k})...")
            # Build model
            model = KernelSVC(C=C, kernel=spec_kernel.kernel, epsilon=epsilon)
            # Fit using the precomputed K_train (we pass it as 'K')
            fitted_ok = model.fit(X_train, y_train, K_train)
            if not fitted_ok:
                print("      Optimization failed or not optimal.")
                continue

            # Predict on validation (K_valid needed)
            # We'll do a manual approach:  model.decision_function uses model.kernel
            # but that re-computes. We want to skip that. We can do the direct approach:
            # We have K_valid of shape (len(X_valid), len(X_train)).
            # f(x_i) = sum_j alpha_j y_j K_valid[i, j] + b
            # We'll just call .predict(X_valid) for simplicity, though it re-calls the kernel.
            # For maximum efficiency, you could store alpha, do your own dot with K_valid, etc.
            print(f"    Predicting on validation...")
            start = time.time()
            y_pred_pm1 = model.predict(X_valid, K_valid)
            end = time.time()
            print(f"    Done in {end - start:.2f} s")

            # Convert back to {0,1} for accuracy
            y_pred_01 = transform_labels_pm1_to_01(y_pred_pm1)
            y_valid_01 = transform_labels_pm1_to_01(y_valid)

            acc = np.mean(y_pred_01 == y_valid_01)
            print(f"      Validation Accuracy = {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_C = C

    return best_k, best_C, best_acc


LOOK_FOR_BEST_KC = False
if LOOK_FOR_BEST_KC:
    for ds in datasets:
        print(f"=== DATASET {ds} ===")
        # Load full training data
        xtr = pd.read_csv(os.path.join(data_path, f'Xtr{ds}.csv'))['seq'].values
        ytr = pd.read_csv(os.path.join(data_path, f'Ytr{ds}.csv'))['Bound'].values
        ytr_pm1 = transform_labels_01_to_pm1(ytr)

        # Train/Valid split
        X_train, X_valid, y_train, y_valid = train_test_split(
            xtr, ytr_pm1, test_size=0.2, random_state=42)

        # We'll get the pre-defined k_values[ds] and C_values[ds]
        k_list = k_values[ds]
        c_list = C_values[ds]

        print(f"Searching best (k, C) among {k_list} x {c_list} ...")
        best_k, best_C, best_acc = search_best_k_C_spectrum(
            X_train, y_train, X_valid, y_valid,
            k_list=k_list, C_list=c_list,
            epsilon=1e-5
        )
        print(f"Dataset {ds}: best (k, C) = ({best_k}, {best_C}), accuracy = {best_acc:.4f}")
        best_kc_map[ds] = (best_k, best_C)

###############################################################################
# 3.b Find lambda,p for Substring
###############################################################################


def search_best_substring(X_train, y_train, X_valid, y_valid,
                          p_list, lambda_list, C_list, epsilon=1e-5):
    """
    Search the best (p, lambda, C).
    We only compute kernel matrices once for each (p, lambda), then loop over C.
    """
    best_acc = 0.0
    best_p = None
    best_lambda = None
    best_C = None

    for p in p_list:
        for lam in lambda_list:
            print(f"  Building substring kernel for p={p}, lambda={lam}...")
            subs_kernel = SubstringKernel(p=p, lambda_=lam)
            print(f"  Building train kernel...")
            start = time.time()
            # Compute the NxN kernel on training
            K_train = subs_kernel.kernel(X_train, X_train, square=True)
            end = time.time()
            print(f"  Done in {end - start:.2f} s")
            # Compute the kernel for validation vs. training
            print(f"  Building valid kernel...")
            start = time.time()
            K_valid = subs_kernel.kernel(X_valid, X_train, square=False)
            end = time.time()

            for C in C_list:
                print(f"    Training SVC for C={C} (p={p}, lambda={lam})...")
                model = KernelSVC(C=C, kernel=subs_kernel.kernel, epsilon=epsilon)

                fitted_ok = model.fit(X_train, y_train, K_train)
                if not fitted_ok:
                    print("      Optimization failed or not optimal.")
                    continue

                print(f"    Predicting on validation...")
                start = time.time()
                y_pred_pm1 = model.predict(X_valid, K_valid)
                end = time.time()
                print(f"    Done in {end - start:.2f} s")
                y_pred_01 = transform_labels_pm1_to_01(y_pred_pm1)
                y_valid_01 = transform_labels_pm1_to_01(y_valid)

                acc = np.mean(y_pred_01 == y_valid_01)
                print(f"      Validation Accuracy = {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_p = p
                    best_lambda = lam
                    best_C = C

    return best_p, best_lambda, best_C, best_acc

lambda_values = {"0": [0.8,0.9],"1": [0.5,0.6,0.7,0.8,0.9],"2" :[0.5,0.6,0.7,0.8,0.9]}   
p_values = {"0":[3,4,5,6],"1":[4,5,6],"2": [2,3,4,5,6]}
C_values = {"0":[0.1,5,100],"1":[0.1,5,100],"2": [0.1,5,100]}

best_lambdap_map = {}

LOOK_FOR_BEST_lambdap = True  # toggle if we want to do the hyperparam search

if LOOK_FOR_BEST_lambdap:
    for ds in datasets:
        print(f"=== DATASET {ds} ===")
        xtr = pd.read_csv(os.path.join(data_path, f'Xtr{ds}.csv'))['seq'].values
        ytr = pd.read_csv(os.path.join(data_path, f'Ytr{ds}.csv'))['Bound'].values
        ytr_pm1 = transform_labels_01_to_pm1(ytr)

        X_train, X_valid, y_train, y_valid = train_test_split(
            xtr, ytr_pm1, test_size=0.2, random_state=42
        )

        # p_values[ds], lambda_values[ds], and C_values[ds]
        p_list = p_values[ds]
        lam_list = lambda_values[ds]
        c_list = C_values[ds]

        print(f"Searching best (p, lambda, C) among {p_list} x {lam_list} x {c_list}...")
        best_p, best_lam, best_C, best_acc = search_best_substring(
            X_train, y_train, X_valid, y_valid,
            p_list, lam_list, c_list, epsilon=1e-5
        )
        print(f"Dataset {ds}: best (p, lambda, C) = ({best_p}, {best_lam}, {best_C}), accuracy = {best_acc:.4f}")
        best_lambdap_map[ds] = (best_p, best_lam, best_C)


###############################################################################
# Make predictions for the best methods and hyperparameters
###############################################################################
best_methods = {"0": "spectrum","1": "spectrum","2":"spectrum"}
best_hyperparams = {"0":[15, 0.6], "1":[8, 1], "2":[8, 1]}  # (k, C) or (p, lambda,C)
# Now do final training on the full dataset & predictions
MAKE_PREDICTIONS = False # toggle to make predictions
if MAKE_PREDICTIONS:
    submission = []
    for ds in datasets:
        # Reload full train data, test data
        xtr = pd.read_csv(os.path.join(data_path, f'Xtr{ds}.csv'))['seq'].values
        ytr = pd.read_csv(os.path.join(data_path, f'Ytr{ds}.csv'))['Bound'].values
        xte = pd.read_csv(os.path.join(data_path, f'Xte{ds}.csv'))['seq'].values

        # Convert labels
        ytr_pm1 = transform_labels_01_to_pm1(ytr)
        method = best_methods[ds]
        if method == "spectrum":
            (k, C) = best_hyperparams[ds]
            spec_kernel = Spectrum(k=k)
        if method == "substring":
            (p, lambda_, C) = best_hyperparams[ds]
            spec_kernel = SubstringKernel(p=p, lambda_=lambda_)
        print(f"Retraining on full data with k={k}, C={C} for dataset {ds}...")

        # Build & fit model
        model = KernelSVC(C=C, kernel=spec_kernel.kernel, epsilon=1e-5)
        model.fit(xtr, ytr_pm1)

        # Predict on test
        y_test_pm1 = model.predict(xte)
        y_test_01 = transform_labels_pm1_to_01(y_test_pm1)

        # Format for submission
        for i, pred in enumerate(y_test_01):
            # ID = 1000*(ds) + i
            row_id = 1000*int(ds) + i
            submission.append([row_id, pred])

    # Save to CSV
    submission_path = os.path.join(base_path, "Yte.csv")
    submission_df = pd.DataFrame(submission, columns=["Id", "Bound"])
    submission_df.to_csv(submission_path, index=False)
    print(f"Final submission saved to {submission_path}")