import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

############ load, processed, save #############
# vis = np.load("visualization-[7, 4, 2].npy", allow_pickle=True) # Xi, Xv, Xi_genre, Xv_genre, y, y_pred, edge_weight
# num_vis = len(vis) #73902
#
# fr = open("feature.txt",'r+')
# dict = eval(fr.read())
# fr.close()
#
# samples = []
# for i in range(num_vis):
#     v = vis[i]
#     sample = []
#
#     xi = []
#     for j in v[0]:
#         xi.append(dict[j])
#     xi[-1] = xi[-1] + "-%4f"%v[1][-1] # the numerical value of timestamp, e.g. "Timestamp-0.2112"
#     sample.append(xi) # xi
#
#     xi_genre = []
#     for j in v[2]:
#         xi_genre.append(dict[j])
#     sample.append(xi_genre) # xi_genre
#
#     sample.append(float(v[4][0])) # y
#     sample.append(float(v[5][0])) # y_pred
#
#     # forget to transpose when creating the visualization_test.npy, so we transpose here
#     # weights = np.transpose(v[6], (0, 2, 1))
#     weights = v[6]
#     sample.append(weights)
#
#     samples.append(sample)
#
# np.save("processed_vis-[7,4,2].npy", samples)

############ load, use #############
# samples = np.load("processed_vis-[7,4,2].npy", allow_pickle=True)
# idxs = list(range(len(samples)))
# random.shuffle(idxs)
#
# for idx in idxs[:100]:
#     sample = samples[idx]
#     labels = []
#     for i in sample[0]:
#         labels.append(i.split('-', 1)[-1])
#     genre = ''
#     for i, s in enumerate(sample[1]):
#         g = s.split("-", 1)[-1]
#         if g != 'NULL':
#             if i == 0:
#                 genre += g
#             else:
#                 genre += ('/' + g)
#         else:
#             break
#     labels.append(genre)
#
#     sns.set(font_scale=0.8)
#     for b in range(3):
#         weights = sample[-1][b]
#         ax = sns.heatmap(weights, vmin=0, vmax=1, cmap="YlGnBu", linewidths=.5, annot=True, xticklabels=labels, yticklabels=labels)
#         ax.set_title('order=%d' % (b+2))
#         # plt.yticks(rotation='90')
#         plt.xticks(rotation=45, verticalalignment="top", horizontalalignment="right", rotation_mode="anchor")
#         plt.subplots_adjust(left=0.25, bottom=0.25)
#         fig = ax.get_figure()
#         fig.savefig("./pic0-1-rating/idx%d-y%d-p%.4f-order%d.png"%(idx, sample[2], sample[3], b))
#         plt.close()
#         # plt.show()


############ 3d #############
from scipy.stats import kde
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import pandas as pd
import numpy as np

# a = np.concatenate([np.random.normal(2, 4, 1000), np.random.normal(4, 4, 1000), np.random.normal(1, 2, 500),
#                     np.random.normal(10, 2, 500), np.random.normal(8, 4, 1000), np.random.normal(10, 4, 1000)])
# df = pd.DataFrame({'x': np.repeat(range(1, 6), 1000), 'y': a})
# nbins = 7
# k = kde.gaussian_kde([df.x, df.y])
# xi, yi = np.mgrid[df.x.min():df.x.max():nbins * 1j, df.y.min():df.y.max():nbins * 1j]
# zi = k(np.vstack([xi.flatten(), yi.flatten()]))
# data = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'z': zi})
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(data.x, data.y, data.z, cmap=plt.cm.Spectral, linewidth=0.2, antialiased=True)
# ax.view_init(30, 80)
# plt.show()



criteo_auc_list = [0.8085,0.8086,0.8082,0.8087,0.8085,0.8083,0.8081,0.8084,0.8083,0.8082,0.8081,0.8082,0.8082,0.8083,0.8082,0.8085, 0.8082, 0.8084,0.8083,0.8081, 0.8082,0.8087, 0.8085,
            0.8083, 0.8083, 0.8086,0.8084,0.8082,0.8086, 0.8085,0.8083,0.8085,0.8082,0.8082,0.8081,0.8087, 0.8083,0.8083,0.8084,0.8083,0.8081,0.8080,0.8083,0.8083,0.8082,0.8084,
            0.8086,0.8083,0.08081]
criteo_logloss_list = [0.4430,0.4429,0.4436,0.4428,0.4435,0.4440,0.4441,0.4432,0.4438,0.4434,0.4438,0.4435,0.4439,0.4434,0.4435,0.4430,0.4435,0.4437,0.4437,0.4441,0.4440,0.4429,0.4431,
                0.4436,0.4440,0.4431,0.4435,0.4437,0.4434,0.4434,0.4438,0.4433,0.4445,0.4441,0.4443,0.4429,0.4436,0.4433,0.4432,0.4436,0.4435,0.4438,0.4432,0.4434,0.4437,0.4431,
                0.4429,0.4433,0.4436]

avazu_auc_list = [0.7759, 0.7759, 0.7761, 0.7760, 0.7761, 0.7758, 0.7761, 0.7762, 0.7761, 0.7755, 0.7757, 0.7756, 0.7760, 0.7759, 0.7759, 0.7759]
avazu_logloss_list = [0.3818,0.3819,0.3821,0.3821,0.3818,0.3819,0.3821,0.3817,0.3817,0.3822,0.3820,0.3820,0.3819,0.3821,0.3822,0.3818]

movielens_auc_list = [0.8500,0.8518,0.8516,0.8528,0.8504,0.8509,0.8483,0.8513,0.8500,0.8523,0.8512,0.8514,0.8497,0.8525,0.8511,0.8514,0.8541,0.8524,0.8512,0.8501,0.8521,0.8525,0.8533,
                 0.8513,0.8503,0.8504,0.8511,0.8504,0.8507,0.8512,0.8503,0.8523,0.8510,0.8495,0.8506,0.8508,0.8518,0.8510,0.8517,0.8519,0.8511,0.8518,0.8523,0.8521,0.8535,0.8500,
                 0.8499,0.8509,0.8508]
movielens_logloss_list = [0.3772,0.3753,0.3736,0.3718,0.3770,0.3747,0.3797,0.3738,0.3736,0.3721,0.3759,0.3751,0.3800,0.3741,0.3744,0.3734,0.3713,0.3725,0.3755,0.3765,0.3733,0.3711,0.3710,
                          0.3738,0.3751,0.3778,0.3757,0.3761,0.3735,0.3745,0.3760,0.3723,0.3767,0.3754,0.3752,0.3733,0.3725,0.3746,0.3736,0.3725,0.3758,0.3745,0.3724,0.3724,0.3716,0.3755,
                          0.3746,0.3757,0.3757]

print (len(movielens_auc_list))
print (len(movielens_logloss_list))

def load_auc(x, y):
    return movielens_auc_list[7*int(x-1)+int(y-1)]
def load_logloss(x, y):
    return movielens_logloss_list[7*int(x-1) + int(y-1)]

nodes = [1, 2, 3, 4, 5, 6, 7]
X = np.array(nodes)
Y = np.array(nodes)
X, Y = np.meshgrid(X, Y)
print (X)
print (Y)
n_samples = X.shape[0]
Z = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        x = X[i][j]
        y = Y[i][j]
        Z[i][j] = load_auc(x, y)
fig = plt.figure(figsize=(8,8))
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z,
    alpha=0.6, cmap="Blues",
    rstride=1, cstride=1, linewidth=0.25, edgecolors='black')
ax.set_zlim(0.8475, 0.8550)
ax.set_xlabel('$m_2$')
plt.xticks([1, 2, 3, 4, 5, 6, 7])
ax.set_ylabel('$m_3$')
plt.yticks([1, 2, 3, 4, 5, 6, 7])
ax.set_zlabel(f'AUC')
ax.view_init(20, 50)
ax.dist = 10

# surf = ax.plot_surface(X, Y, Z, cmap="Blues",
#                        linewidth=0, antialiased=False)

# fig.colorbar(surf, shrink=0.8, aspect=10, format='%.2f', pad=0)
fig.savefig('movielens_auc_m.pdf', bbox_inches='tight')


