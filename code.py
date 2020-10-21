import numpy as np
import matplotlib.pyplot as plt
import math
import basis_pursuit_cvx as basis_pursuit
import time

# fig 1
def compare_BP_minL2_different_beta_norm(s, n, sigma, p_list, seed, WB2, WL2, beta_norm, display=False):
    np.random.seed(seed)
    start_time = time.time()
    p_largest = p_list[-1]
    np.random.seed(seed)
    X_all = np.random.randn(n, int(p_largest))
    X_all /= np.linalg.norm(X_all, axis=0)
    Z = np.random.randn(n, 1)
    Z /= np.linalg.norm(Z)
    Z *= sigma
    beta = np.random.randn(n, 1)
    beta[s:, :] = 0
    beta /= np.linalg.norm(beta)
    beta *= beta_norm
    Y = np.matmul(X_all[:, :s], beta[:s, :]) + Z
    for p in p_list:
        p = int(p)
        X = X_all[:, :p]
        # min l2 norm
        t = np.matmul(np.linalg.pinv(X), Y)
        t[:s] -= beta[:s, :]
        WL2.append(np.linalg.norm(t, ord=2))
        # BP
        t = basis_pursuit.basis_pursuit(X, Y)
        t[:s] -= beta[:s, :]
        WB2.append(np.linalg.norm(t, ord=2))
        if display:
            print("s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p, time.time() - start_time))
    print("Done! s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p_largest, time.time() - start_time))
p_list = [500]+[500+3**i for i in range(11)]
WB2 = []
WL2 = []
seed = 12
compare_BP_minL2_different_beta_norm(1, 500, 0.01, p_list, seed, WB2, WL2, 1, True)
WB2_another = []
WL2_another = []
compare_BP_minL2_different_beta_norm(100, 500, 0.01, p_list, seed, WB2_another, WL2_another, 1, True)
WB2_middle = []
WL2_middle = []
compare_BP_minL2_different_beta_norm(100, 500, 0.01, p_list, seed, WB2_middle, WL2_middle, 0.1, True)

fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1.5]})
fig.set_figwidth(8)
plt.rcParams.update({'font.size': 14})

def draw_sub_plot(ax, zoom_in=False):

    ax.plot(p_list, WB2, label=r'$BP, ||\beta||_2$=1, s=1', marker='4', markersize=12, linewidth=2)
    ax.plot(p_list, WB2_another, label=r'BP, $||\beta||_2$=1, s=100', marker='x', markersize=12, linewidth=2)
    ax.plot(p_list, WB2_middle, label=r'BP, $||\beta||_2$=0.1, s=100', marker='1', markersize=12, linewidth=2)

    ax.plot(p_list, WL2, '--', label=r'min l2, $||\beta||_2$=1, s=1', fillstyle='none', marker='^', markersize=8, linewidth=2)
    ax.plot(p_list, WL2_another, '--', label=r'min l2, $||\beta||_2$=1, s=100', fillstyle='none', marker='v', markersize=8, linewidth=2)
    ax.plot(p_list, WL2_middle, '--', label=r'min l2, $||\beta||_2$=0.1, s=100', fillstyle='none', marker='>', markersize=8, linewidth=2)
    ax.set(xlabel=r"$p$", ylabel=r'$||w||_2$', xscale='log', yscale='log')
    if zoom_in:
        ax.set_xlim([490, 600])
        ax.set(xscale='linear')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set(ylabel='', xlabel='p (zoomed in)')
    ax.grid(b=True, which='both')

draw_sub_plot(ax1)
draw_sub_plot(ax2, True)
plt.show()

# fig 2
def compare_BP_minL2(s, n, sigma, p_list, seed, WB2, display=False):
    start_time = time.time()
    p_largest = p_list[-1]
    np.random.seed(seed)
    X_all = np.random.randn(n, int(p_largest))
    X_all /= np.linalg.norm(X_all, axis=0)
    Z = np.random.randn(n, 1)
    Z /= np.linalg.norm(Z)
    Z *= sigma
    beta = np.random.randn(n, 1)
    beta[s:, :] = 0
    beta /= np.linalg.norm(beta)
    Y = np.matmul(X_all[:, :s], beta[:s, :]) + Z
    for p in p_list:
        p = int(p)
        X = X_all[:, :p]
        if p >= n:
            t = basis_pursuit.basis_pursuit(X, Y)
            t[:s] -= beta[:s, :]
            WB2.append(np.linalg.norm(t, ord=2))
        else:
            # min l2 norm
            t = np.matmul(np.linalg.pinv(X), Y)
            temp = beta.copy()
            temp[:p, :] -= t[:p]
            WB2.append(np.linalg.norm(temp, ord=2))
            
        if display:
            print("s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p, time.time() - start_time))
    print("Done! s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p_largest, time.time() - start_time))
p_list_1 = [80, 100, 120, 140] + [150+3**i-1 for i in range(11)]
WB2 = []
compare_BP_minL2(1, 150, 0.01, p_list_1, 3, WB2, True)
WB2_another = []
p_list_2 = [80, 150, 300, 350, 400, 450, 500, 550]+[600+3**i-1 for i in range(11)]
compare_BP_minL2(1, 600, 0.01, p_list_2, 3, WB2_another, True)
WB2_middle = []
p_list_3 = [80, 150, 200, 250, 270]+[300+3**i-1 for i in range(11)]
compare_BP_minL2(1, 300, 0.01, p_list_3, 3, WB2_middle, True)
plt.rcParams.update({'font.size': 12})
plt.plot(p_list_1, WB2, label='n=150', marker='x', markersize=12, linewidth=2)
plt.plot(p_list_3, WB2_middle, label='n=300', marker='1', markersize=12, linewidth=2)
plt.plot(p_list_2, WB2_another, label='n=600', marker='+', markersize=12, linewidth=2)
plt.xlabel(r"$p$")
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$||w^{BP}||_2$')
plt.legend()
plt.grid(b=True, which='both')
plt.savefig('change_n.eps', format='eps')
plt.show()

# fig 3
def BP_change_n(WB2, s=1, sigma=0.1, n_list=[i*50 for i in range(4, 12)], beta_norm=1, seed=0, display=True):
    start_time = time.time()
    p = 5000
    n_largest = n_list[-1]
    np.random.seed(seed)
    X_all = np.random.randn(n_largest, p)
    Z_all = np.random.randn(n_largest, 1)
    beta = np.random.randn(s, 1)
    beta[s:, :] = 0
    beta /= np.linalg.norm(beta)
    beta *= beta_norm
    for n in n_list:
        p = int(p)
        X = np.copy(X_all[:n, :])
        X /= np.linalg.norm(X, axis=0)
        Z = np.copy(Z_all[:n, :])
        Z /= np.linalg.norm(Z)
        Z *= sigma
        beta = np.resize(beta, (n, 1))
        Y = np.matmul(X[:, :s], beta[:s, :]) + Z
        
        t = basis_pursuit.basis_pursuit(X, Y)
        t[:s] -= beta[:s, :]
        WB2.append(np.linalg.norm(t, ord=2))
        if display:
            print("s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p, time.time() - start_time))
    print("Done! s={}, n={}, p={}, finished in {:.2f} seconds".format(s, n, p, time.time() - start_time))

WB2 = []
WB2_s2 = []
WB2_epsilon = []
n_list = [100*i for i in range(1, 15)]
num_trial = 1
BP_change_n(WB2, n_list=n_list, seed=1, s=1, sigma=0.15)
BP_change_n(WB2_s2, n_list=n_list, seed=1, s=20, sigma=0.15)
BP_change_n(WB2_epsilon, n_list=n_list, seed=1, s=20, sigma=0.6)
def divided_by_n_1_4(_w, _n_list=n_list):
    l = len(_n_list)
    return [_w[i] / (_n_list[i] ** 0.25) for i in range(l)]

plt.rcParams.update({'font.size': 14})
plt.plot(n_list, divided_by_n_1_4(WB2, n_list), marker='+', label=r's=1, $||\epsilon_{train}||_2$=0.15', markersize=15, linewidth=3)
plt.plot(n_list, divided_by_n_1_4(WB2_s2, n_list),'--', marker='x', label=r's=20, $||\epsilon_{train}||_2$=0.15', markersize=10, linewidth=3)
plt.plot(n_list, divided_by_n_1_4(WB2_epsilon, n_list),'--', marker='4', label=r's=20, $||\epsilon_{train}||_2$=0.6', markersize=15, linewidth=3)
plt.ylim(bottom=0, top=0.4)
plt.xlabel(r'$n$')
plt.ylabel(r'$n^{-1/4}\cdot||w^{BP}||_2$')
plt.grid('both')
plt.xscale('log')
plt.legend()
plt.show()