import tensorflow as tf
import numpy as np
import scipy.sparse as sp

# for testing
n = 2
k = 4
d = 3
u = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]],
[[-2., -2., -2.], [-4., -4., -4.], [-6., -6., -6.], [-8., -8., -8.]]] # n=2,k=4,d=3
i = [[[1., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 1., 1.]],
[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]],
] # n=2, k=4, d=3

u = tf.constant(u)
i = tf.constant(i)

u1 = tf.expand_dims(u, 1)
u2 = tf.expand_dims(u, 2)
u12 = tf.multiply(u2, u1)


num_heads = 2
num_units = 4

a = tf.layers.dense(u, 1, activation=tf.nn.relu) # n=2,k=4,d=1
a = tf.nn.softmax(a, axis=1)
# output = tf.multiply(u, a) # n=2,k=4,d=3

output = tf.expand_dims(u, 1) * tf.expand_dims(u, 2)

# a1 = tf.ones([d, num_heads])
# a2 = tf.ones([d, num_units])
#
# A = tf.layers.dense(u12, num_heads, activation=tf.nn.leaky_relu) # [n, k, k, 2]
# H = tf.layers.dense(u12, num_units, activation=None) # [n, k, k, num_units]
# Q_res = tf.layers.dense(u, num_units, activation=tf.nn.leaky_relu) # [n, k, num_units]
# # Split and concat
# A_ = tf.concat(tf.split(A, num_heads, axis=-1), axis=0) # [num_heads*batch_size, field_size, field_size, 1]
# H_ = tf.concat(tf.split(H, num_heads, axis=-1), axis=0)  # [num_heads*batch_size, field_size, field_size, num_units/num_heads]
# # Activation
# weights = tf.nn.softmax(A_, axis=2)
# # Dropouts
# weights = tf.layers.dropout(weights, rate=0.1)
# # Weighted sum
# outputs = tf.multiply(weights, H_) # [num_heads*batch_size, field_size, field_size, num_units/num_heads]
# outputs = tf.reduce_sum(outputs, axis=2, keepdims=False)  # [num_heads*batch_size, field_size, num_units/num_heads]
# # Restore shape
# outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2) # [batch_size, field_size, num_units]
# # Residual connection
# outputs += Q_res
# outputs = tf.nn.relu(outputs)


# ui = tf.reshape(ui, [n, k, m, k, d])
# ui = tf.transpose(ui, [0, 2, 1, 3, 4]) #[n, m, k, k, d]
# ui = tf.reshape(ui, [n*m*k*k, d])
# ui_score = tf.reduce_sum(ui, axis=-1, keepdims=True) # [nm_kk, 1] score without attention
# ui_score = tf.reduce_sum(tf.reshape(ui_score, [n, m, -1]), axis=-1)

#
# ui_att_score = tf.multiply(tf.reshape(ui_score, [-1, k * k]), tf.ones([k*k]))  # [nm_, k*k]
# ui_score = tf.reshape(tf.reduce_mean(ui_att_score, axis=-1, keepdims=False),
#                                              [n, m]) # [n, m_]

# ui = tf.matmul(ui, p) #[n, m, k, k, 1]
# ui = tf.reshape(ui, [n, m, k*k])
# ui = tf.nn.softmax(ui, axis=-1)
# ui = tf.reduce_sum(ui, axis=-1, keepdims=False)

# for training
# b = 2
# k = 4
# d = 3
# u = [[[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]],
# [[-1., -1., -1.], [-2., -2., -2.], [-3., -3., -3.], [-4., -4., -4.]]] # b=2,k=4,d=3
# i = [[[1., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 1., 1.]],
# [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]]] # b=2, k=4, d=3
#
#
# u = tf.reshape(u, [b, 1, k, d])
# i = tf.reshape(i, [b, k, 1, d])
# ui = tf.multiply(u, i)
# ui = tf.reshape(ui, [b, k*k, d])


# h = [[1., 2., 3.], [-1., -2., -3.]] # 2, 3
# p = [[0., 1., 0.], [1., 1., 0.]] # 3, 3
# h = tf.constant(h)
# p = tf.constant(p)
# hp = tf.reduce_sum(tf.multiply(h, p), 1)

initializer = tf.contrib.layers.xavier_initializer()

session = tf.Session()
session.run(tf.global_variables_initializer())

# print (session.run(embedding_inputs))
# print (session.run(conv).shape)

b = 2
k = 3
f = 4

s = [[[1., 2., 3., 0.], [2., 1., 3., 0.], [0., 3., 2., 1.], [3., 0., 2., 1.]],
     [[1, 3., 2., 0.], [2., 0., 3., 1.], [3, 0., 1, 2], [3, 1., 0, 2]]]

s = tf.constant(s)
values, indices = tf.nn.top_k(s, k)
kth = tf.reduce_min(values, -1, keepdims=True)
topk = tf.cast(tf.greater_equal(s, kth), tf.float32)


# print (session.run(update))
print (session.run(indices))
print ('stack:')
print (session.run(topk))
# print (session.run(s))



# a = np.array([[1., 0., 1.], [1., 1., 0.], [0., 1., 1.], [1., 1., 1.]])
# a = np.array([0., 1., 1.])
# b = np.where(a>0.5)
# min = a.min(axis=1)
# max = a.max(axis=1)
# print (b)
# print (max)
# a = np.median(min)
# print (a)
