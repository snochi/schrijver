import numpy as np

from numpy import linalg as la
from functools import reduce


class IncorrectMatrixDimensionError(BaseException):
    pass


def nextIndices(indices, n):
    if indices[-1] == n-1:
        sub = nextIndices(indices[:-1], n-1)
        return sub+[sub[-1]+1]
    return indices[:-1]+[indices[-1]+1]

def linIndependenceCheck(columns):
    matrix = np.column_stack(tuple(columns))
    eigvals = la.eigvals(matrix)
    return not (0 in eigvals)

def sch(columns, b):
    # find what m,n are
    m, n = len(b), len(columns)
    # choose m linearly independent vectors
    indices = list(range(m))
    basis = columns[:m]
    while not linIndependenceCheck(basis):
        indices = nextIndices(indices, n)
        basis = []
        for idx in indices:
            basis.append(columns[idx])
    # main iteration
    while True:
        # solve for \lambda_{i_1},...,\lambda_{i_m}
        matrix = np.column_stack(tuple(basis))
        lambdas = la.solve(matrix, b)
        # first if
        if np.all(lambdas>0):
            coefficients = []
            counter = 0
            for idx in range(n):
                if idx in indices:
                    coefficients.append(lambdas[counter])
                else:
                    coefficients.append(0)
            return True, np.array(coefficients)
        # find h
        h = min(filter(lambda idx: lambdas[idx]<0, indices))
        # find c
        subbasis = []
        for idx in indices:
            if idx!=h:
                subbasis.append(columns[idx])
        submatrix = np.column_stack(tuple(subbasis + [np.zeros(m)]))
        eigvals, eigvecs = la.eig(np.transpose(submatrix))
        zero_idx = np.where(eigvals==0)
        c = eigvecs[:,zero_idx[0][0]]
        if np.inner(c,columns[h]) < 0: # this cannot be 0
            c = -c
        # second if
        # -1e-8 is to account for the imprecision in floating-point system
        if reduce(lambda x, y: x and y, map(lambda a: np.inner(c,a) >= -1e-8, basis), True): 
            return False, c
        # find s
        s = min(filter(lambda idx: np.inner(c,columns[idx])<0, range(n)))
        # replace a_h with a_s
        basis = subbasis.append(columns[s])


# essentially sch, but uses matrix instead of list of column vectors
def schrijver(A, b):
    if la.matrix_rank(A) < A.shape[0]:
        raise IncorrectMatrixDimensionError("A must have full row rank")
    columns = []
    for i in range(A.shape[1]):
        columns.append(A[:,i])
    return sch(columns, b)



# this is for demonstration purpose
# this program generates random examples until it sees
# A, b such that b\in\conv(a_1,...,a_n)
if __name__ == "__main__":
    while True:
        m = np.random.randint(3, 11) # 3, 4, ..., 10
        n = np.random.randint(3, 11)
        A = np.random.randint(-10, 11, (m,n))
        b = np.random.randint(-10, 11, (m,1))

        print("A = ")
        print(A)
        print("b = ")
        print(b)
        print()

        try: 
            isin, vec = schrijver(A,b)
            if isin:
                print(b,end=" ")
                print("is in conv(A)")
                print("coefficients:",end=" ")
                print(vec)
                break
            else:
                print(b,end=" ")
                print("is not in conv(A)")
                print("certificate:",end=" ")
                print(vec)
        except IncorrectMatrixDimensionError as e:
            print(e)
