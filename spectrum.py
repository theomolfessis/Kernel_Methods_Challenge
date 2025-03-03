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

    def kernel(self, X, Y):
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
        # We'll store them in 3D arrays for clarity: B[k][i][j].
        # For memory reasons, one might only keep B for k and k-1, but let's keep it simpler here.
        B = [np.zeros((n_x + 1, n_y + 1)) for _ in range(p+1)]
        # B[0][i][j] = 1 for all i, j, because there's exactly one subsequence of length 0 (the empty subsequence)
        B[0][:, :] = 1.0

        # The main DP to fill up B[k]
        for k in range(1, p+1):
            for i in range(1, n_x + 1):
                # We store partial sums in a running variable to handle the gap weighting
                accum = 0.0
                for j in range(1, n_y + 1):
                    # B[k][i][j] depends on B[k][i-1][j], plus the new matches
                    B[k][i][j] = B[k][i-1][j]

                    if x[i-1] == y[j-1]:
                        # If characters match, add lam^2 plus the contribution from B[k-1][i-1][j-1]
                        # plus gap weighting from all possible expansions.
                        # A standard formulation is:
                        # B[k][i][j] += lambda_ * B[k][i-1][j-1]
                        # But in the Lodhi kernel, we also accumulate partial sums with an exponent of lam
                        # for the distance between i and j.
                        # We'll keep a simpler version:
                        B[k][i][j] += lam * B[k-1][i-1][j-1]

                    # Then multiply by lam to handle the gap weighting along j dimension
                    B[k][i][j] *= lam

        # The kernel value is the sum over all i, j of B[p][i][j],
        # or in some formulations just B[p][n_x][n_y]. We'll do the typical approach
        # that sums up the final matrix:
        return np.sum(B[p])

    def kernel(self, X, Y):
        """
        Compute the NxM kernel matrix for sets of sequences X and Y.
        X: list of strings (size N)
        Y: list of strings (size M)
        Returns an (N x M) NumPy array K where K[i,j] = substring kernel of X[i], Y[j].
        """
        N = len(X)
        M = len(Y)
        Kmat = np.zeros((N, M))

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

    We implement this with SciPy's SLSQP.
    """
    def __init__(self, C=1.0, kernel=None, epsilon=1e-5):
        self.C = C
        self.kernel = kernel  # a function kernel(X, X) or an object with .kernel()
        self.epsilon = epsilon
        # Learned parameters
        self.alpha = None
        self.support_idx = None
        self.X = None
        self.y = None
        self.b = 0.0



    def fit(self, X, y):
        """
        Fit SVM model using sequences X and labels y in {−1, +1}.
        We'll compute the NxN Gram matrix K, then solve the dual problem
        with constraints, but now using Gurobi.
        """
        self.X = X
        self.y = y
        n = len(y)


        # 1) Compute Gram matrix
        K = self.kernel(X, X)  # shape (n, n)

        # 2) Build Q = (y_i * y_j) * K[i, j]
        #    We'll keep it around for computing b, etc.
        Q = (y.reshape(-1,1) * y.reshape(1,-1)) * K

        # -------------------------------------------------
        #         BUILD AND SOLVE THE GUROBI MODEL
        # -------------------------------------------------
        model = gp.Model("SVM-dual")

        # 3) Create alpha variables with bounds: 0 <= alpha_i <= C
        alpha = model.addVars(n, lb=0.0, ub=self.C, name="alpha", vtype=GRB.CONTINUOUS)

        # 4) Add equality constraint: sum_i (alpha_i * y_i) = 0
        model.addConstr(gp.quicksum(alpha[i] * float(y[i]) for i in range(n)) == 0.0,
                        name="eq_y_alpha")

        # 5) Set the objective function:
        #    Minimize:  0.5 * alpha^T Q alpha  - sum_i alpha_i
        #    BUT Gurobi uses a 'minimization' default, so we convert our
        #    original objective (which was a max) by multiplying by -1.
        #
        #    Original:  maximize   sum_i alpha_i  - 0.5 * alpha^T Q alpha
        #    is the same as minimize:  0.5 * alpha^T Q alpha  - sum_i alpha_i
        #
        #    We'll define a QuadExpr for the 0.5 * alpha^T Q alpha part,
        #    then subtract sum(alpha_i).
        quad_expr = gp.QuadExpr()
        # 0.5 * sum_{i,j} Q[i,j] * alpha_i * alpha_j
        for i in range(n):
            for j in range(n):
                if Q[i, j] != 0.0:
                    quad_expr.add(alpha[i] * alpha[j], Q[i, j] * 0.5)

        # Now the linear part: - sum_i alpha_i
        linear_expr = gp.quicksum(alpha[i] for i in range(n))
        objective_expr = quad_expr - linear_expr

        model.setObjective(objective_expr, GRB.MINIMIZE)

        # 6) Solve
        model.setParam('OutputFlag', 0)  # to suppress solver output, optional
        model.optimize()

        # Check if we have an optimal solution
        if model.status != GRB.OPTIMAL:
            # If not solved to optimality, return False or handle error
            return False

        # 7) Retrieve alpha_i
        alpha_opt = np.array([alpha[i].X for i in range(n)])
        self.alpha = alpha_opt

        # 8) Identify support vectors: alpha_i > epsilon
        self.support_idx = np.where(alpha_opt > self.epsilon)[0]

        # 9) Compute the bias b
        #    For each support vector i with 0 < alpha_i < C, we have:
        #    b_i = y_i - sum_j alpha_j y_j K(i,j).
        #    We'll average them.
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
            # rare corner case: no alpha strictly less than C in the support
            self.b = 0.0

        return True


    def decision_function(self, Xtest):
        """
        Compute the SVM decision function f(x) = sum_j alpha_j y_j K(x, x_j) + b
        for each x in Xtest.
        """
        # Ktest[i, j] = kernel(Xtest[i], X[j])
        Ktest = self.kernel(Xtest, self.X)
        # Weighted sum over support vectors
        f = np.zeros(len(Xtest))
        for i, xt in enumerate(Xtest):
            # sum_j alpha_j y_j K(xt, x_j)
            val = 0.0
            for j in self.support_idx:
                val += self.alpha[j] * self.y[j] * Ktest[i, j]
            f[i] = val
        return f + self.b

    def predict(self, Xtest):
        """
        Predict labels in {−1, +1} for each example in Xtest.
        """
        return np.sign(self.decision_function(Xtest))

###############################################################################
# 3.a Find Best (k,C) for Spectrum, Then Make Predictions
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

datasets = ["0", "1", "2"]
k_values = {"0": [2,3,15],"1": [12,15],"2" :[8,15]} #[[3, 4, 5,8,10]] * 3  
C_values = {"0":[0.1,0.3,0.6],"1":[50,500],"2": [5,50]}#[[1, 10, 200]] * 3

best_kc_map = {}

LOOK_FOR_BEST_KC = False  # toggle if we want to do the hyperparam search


if LOOK_FOR_BEST_KC:
    for ds in datasets:
        print(f"=== DATASET {ds} ===")
        # Load full training data
        xtr = pd.read_csv(os.path.join(data_path, f'Xtr{ds}.csv'))['seq'].values
        ytr = pd.read_csv(os.path.join(data_path, f'Ytr{ds}.csv'))['Bound'].values

        # Convert labels from {0,1} to {−1,+1}
        ytr_pm1 = transform_labels_01_to_pm1(ytr)

        print("Searching for best (k, C)...")
        # Split train/validation
        X_train, X_valid, y_train, y_valid = train_test_split(xtr, ytr_pm1, test_size=0.2, random_state=42)
        best_acc = 0.0
        best_k = None
        best_C = None

        for k, C in product(k_values[ds], C_values[ds]):
            print('Starting optimization for k =', k, ', C =', C)
            start = time.time()
            # Create kernel
            spec_kernel = Spectrum(k=k)
            # Create SVC
            model = KernelSVC(C=C, kernel=spec_kernel.kernel, epsilon=1e-5)
            # Fit on the training subset
            successfullyfitted = model.fit(X_train, y_train)
            if not successfullyfitted:
                print("Optimization failed.")
                continue
            # Evaluate on validation
            y_pred_pm1 = model.predict(X_valid)
            # Convert back to {0,1} if we want an accuracy measure vs. original labels
            y_pred_01 = transform_labels_pm1_to_01(y_pred_pm1)
            y_valid_01 = transform_labels_pm1_to_01(y_valid)

            acc = np.mean(y_pred_01 == y_valid_01)
            end = time.time()
            print(f"Validation accuracy = {acc:.4f} (time: {end-start:.2f}s)")
        
            # Track best
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_C = C

        print(f"Best (k, C) = ({best_k}, {best_C}) with validation accuracy = {best_acc:.4f}")
        best_kc_map[ds] = (best_k, best_C)
    



###############################################################################
# 3.b Find lambda,p for Substring
###############################################################################

lambda_values = {"0": [0.5,0.6,0.7,0.8,0.9],"1": [0.5,0.6,0.7,0.8,0.9],"2" :[0.5,0.6,0.7,0.8,0.9]}   
p_values = {"0":[2,3,4,5,6],"1":[2,3,4,5,6],"2": [2,3,4,5,6]}
C_values = {"0":[0.1,0.3,0.6],"1":[50,500],"2": [5,50]}

best_lambdap_map = {}

LOOK_FOR_BEST_lambdap = True  # toggle if we want to do the hyperparam search

if LOOK_FOR_BEST_lambdap:
    for ds in datasets:
        print(f"=== DATASET {ds} ===")
        # Load full training data
        xtr = pd.read_csv(os.path.join(data_path, f'Xtr{ds}.csv'))['seq'].values
        ytr = pd.read_csv(os.path.join(data_path, f'Ytr{ds}.csv'))['Bound'].values

        # Convert labels from {0,1} to {−1,+1}
        ytr_pm1 = transform_labels_01_to_pm1(ytr)

        print("Searching for best (lambda,p, C)...")
        # Split train/validation
        X_train, X_valid, y_train, y_valid = train_test_split(xtr, ytr_pm1, test_size=0.2, random_state=42)
        best_acc = 0.0
        best_k = None
        best_C = None

        for k, C in product(k_values[ds], C_values[ds]):
            print('Starting optimization for k =', k, ', C =', C)
            start = time.time()
            # Create kernel
            spec_kernel = SubstringKernel(p=p, lambda_=lambda_)
            # Create SVC
            model = KernelSVC(C=C, kernel=spec_kernel.kernel, epsilon=1e-5)
            # Fit on the training subset
            successfullyfitted = model.fit(X_train, y_train)
            if not successfullyfitted:
                print("Optimization failed.")
                continue
            # Evaluate on validation
            y_pred_pm1 = model.predict(X_valid)
            # Convert back to {0,1} if we want an accuracy measure vs. original labels
            y_pred_01 = transform_labels_pm1_to_01(y_pred_pm1)
            y_valid_01 = transform_labels_pm1_to_01(y_valid)

            acc = np.mean(y_pred_01 == y_valid_01)
            end = time.time()
            print(f"Validation accuracy = {acc:.4f} (time: {end-start:.2f}s)")
        
            # Track best
            if acc > best_acc:
                best_acc = acc
                best_k = k
                best_C = C

        print(f"Best (k, C) = ({best_k}, {best_C}) with validation accuracy = {best_acc:.4f}")
        best_kc_map[ds] = (best_k, best_C)

###############################################################################
# Make predictions for the best methods and hyperparameters
###############################################################################
best_methods = {"0": "spectrum","1": "substring","2":"spectrum"}
best_hyperparams = {"0":[3, 0.1], "1":[3, 0.1], "2":[3, 0.1]}  # (k, C) or (p, lambda,C)
# Now do final training on the full dataset & predictions
if MAKE_PREDICTIONS:
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
    submission_path = os.path.join(base_path, "submission.csv")
    submission_df = pd.DataFrame(submission, columns=["Id", "Bound"])
    submission_df.to_csv(submission_path, index=False)
    print(f"Final submission saved to {submission_path}")