from cvxopt import spmatrix, matrix, solvers
import numpy

def basis_pursuit(A, b):
    '''
    min ||x||_1
    subject to: Ax = b
    '''
    (n, p) = A.shape
    _c = numpy.ones(2 * p)
    _A = numpy.concatenate((A, -A), axis=1)
    _b = b
    _bounds = []
    for ii in range(2 * p):
        _bounds.append((0, None))
    G = spmatrix(-1.0, range(2*p), range(2*p))
    h = matrix(numpy.zeros((2*p, 1)), tc='d')
    _c = matrix(_c, tc='d')
    _A = matrix(_A, tc='d')
    _b = matrix(_b, tc='d')
    solvers.options['show_progress'] = False
    res = solvers.lp(_c, G, h, _A, _b)

    
    x = []
    for ii in range(p):
        x.append([res['x'][ii] - res['x'][ii+p]])
    return x

if __name__ == '__main__':
    A = numpy.matrix([[1, 1, 1], [1, -1, 0]])
    b = numpy.matrix([[3],[5]])
    # A = numpy.matrix([[1, 3, 2]])
    # b = [[3]]
    print(basis_pursuit(A, b))
    # print(BP.basis_pursuit(A, b))