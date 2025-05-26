import numpy as onp
# Scipy
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import identity
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pickle


def solve_Eikonal(N, epsilon):
    hg = onp.array(1/(N+1))
    x_grid = (onp.arange(1,N+1,1))*hg
    a1 = onp.ones((N,N+1))
    a2 = onp.ones((N+1,N))

    # diagonal element of A
    a_diag = onp.reshape(a1[:,:N]+a1[:,1:]+a2[:N,:]+a2[1:,:], (1,-1))
    
    # off-diagonals
    a_super1 = onp.reshape(onp.append(a1[:,1:N], onp.zeros((N,1)), axis = 1), (1,-1))
    a_super2 = onp.reshape(a2[1:N,:], (1,-1))
    
    A = diags([[-a_super2[onp.newaxis, :]], [-a_super1[onp.newaxis, :]], [a_diag], [-a_super1[onp.newaxis, :]], [-a_super2[onp.newaxis, :]]], [-N,-1,0,1,N], shape=(N**2, N**2), format = 'csr')
    XX, YY = onp.meshgrid(x_grid, x_grid)
    f = onp.zeros((N,N))
    f[0,:] = f[0,:] + epsilon**2 / (hg**2)
    f[N-1,:] = f[N-1,:] + epsilon**2 / (hg**2)
    f[:, 0] = f[:, 0] + epsilon**2 / (hg**2)
    f[:, N-1] = f[:, N-1] + epsilon**2 / (hg**2)
    fv = f.flatten()
    fv = fv[:, onp.newaxis]
    
    mtx = identity(N**2)+(epsilon**2)*A/(hg**2)
    sol_v = scipy.sparse.linalg.spsolve(mtx, fv)
    # sol_v, exitCode = scipy.sparse.linalg.cg(mtx, fv)
    # print(exitCode)
    sol_u = -epsilon*onp.log(sol_v)
    sol_u = onp.reshape(sol_u, (N,N))
    return XX, YY, sol_u


if __name__ == "__main__":
    num = 2000
    N = num - 2
    p1, p2, Y_test = solve_Eikonal(N, 0.1)
    X_test = onp.concatenate(((p1.T).reshape(-1, 1), (p2.T).reshape(-1, 1)), axis=1)
    truth = {
        "X_test": X_test,
        "Y_test": Y_test
    }
    with open('truth_10.pickle', 'wb') as handle:
        pickle.dump(truth, handle, protocol=pickle.HIGHEST_PROTOCOL)
    hg = onp.array(1 / (N + 1))
    X = (onp.arange(1, N + 1, 1)) * hg
    T = (onp.arange(1, N + 1, 1)) * hg
    xx, tt = onp.meshgrid(X, T)
    xx = xx.T
    tt = tt.T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    err_contourf = ax.contourf(xx, tt, Y_test.reshape(-1).reshape(xx.shape), 100,
                               cmap=plt.cm.coolwarm)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Solver value')
    fig.colorbar(err_contourf)
    plt.show()
    print()
