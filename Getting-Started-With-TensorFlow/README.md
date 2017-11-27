# TensorFlow 是个什么玩意儿

从字面意思理解，TensorFlow就是数据流，很多很多的Tensor形成的Flow。从计算机角度理解，TensorFlow是一个深度学习的API库，提供多种语言接口，不过最方便的是python语言接口，python可以很简单的使用。

目前，TensorFlow很方便使用，在Windows上也开始支持python接口，可以不用折腾Linux，在Windows上就能很方便的使用python接口的TensorFlow。


## 什么是Tensor

学习数学的时候，曾经了解过数据的表示方法，最简单的是标量，就是一个数，再往后，还有向量，表示有方向的量，实际上是多个标量组成一个维度的量，在学习矩阵分析的时候，还会遇到矩阵，是向量的组合，在深一层，就是张量，也就是tensor。

用数据维度表示，标量的维度为0，向量是1维的，矩阵是2维的，张量就是3维，不知道是不是更高维度的量表示什么，也可能是3维以上的量都叫做张量。

在Tensor flow中，所有的数据都表示张量，其实救赎维度不同的张量。比如在Tensor Flow中举得例子：

- 3 # a rank 0 tensor; a scalar with shape []
- [1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
- [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
- [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

## 在Python中使用Tensor Flow
在Python程序中使用Tensor Flow的接口很简单，只需要引入Tensor flow的包，比如；
```
import tensorflow as tf
```
这样就可以使用Tensor Flow的类、方法、一些符号等，在文档的说明中，都是默认程序员已经使用这样的语句包含TensorFlow的包了。

## Tensor Flow是怎么工作的
我的理解，Tensor Flow是一个数据流，整个程序是一个有很多张量节点构成的图，Tensor Flow程序的执行就是数据流在整个图中“流动”。

完成一个TensorFlow的程序需要两个步骤：第一构建这样的一个图，第二执行构建的流图。Tensor Flow的这些图被称作是计算图。

在Tensor Flow中，图的每个节点输入和输出的数据都是Tensor，也就是在TensorFlow中，所有的数据都是张量的形式，一个节点有0个或者多个输入，有一个张量作为输出。

**简单常量节点**
```
a = tf.constant(3.0, dtype=tf.float32)
```
或者省去数据类型，默认类型就是tf.float32，
b = tf.constant(4.0)

直接使用print函数打印这些节点，只会输出节点的信息，但是不会输出节点的值，这些节点看作是一个过程，只有执行的时候，才会有输出，静止的时候，只会得到节点静态信息。

比如，下面顶一个session，并且调用run函数执行两个节点

```
sess = tf.Session()
print(sess.run([node1, node2]))
```

这样会输出节点的值。

## 稍微复杂一点TensorFlow程序

构建简单的图，比如下面的程序，使用简单的加法运算，构建了一个简单计算图。

```
from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
```

## 占位符
占位符也是一种节点，没有数值，通常被用作输入占位符。
比如，下面定义了加法的占位符：

```
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```

a 和b是两个占位符，adder_node是两个占位符的加法组合的一个节点，在运行的时候需要给定两个输入量。

```
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```

继续增加一点难度，
```
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
```

## 变量节点
在机器学习过程中，通常是学习得到的一个参数模型，参数在学习过程是不断地变化的，以达到最优的结果，变量在机器学习中使用非常多。Tensor Flow的变量节点定义如下：

```
W = tf.Variable([.3], dtype=float32)
b = tf.Variable([-.3], dtype=float32)
x= tf.placeholder(tf.float32)
linear_model=W*x+b
```

常量节点在使用语句定义的时候，就已经被初始化值了，而变量的值，需要在调用函数才能初始化，比如通过下面的方式：

```
init = tf.global_variable_initializer()
sess.run(init)
```

上面定义了一个线性模型，可以计算他的误差：

```
y = tf.placeholder(tf.float32)
square_delta = tf.square(linear_model-y)
loss = tf.reduce_sum(square_delta)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```

## 训练API

前面的内容简单的介绍了Tensor Flow中的数据类型，以及Tensor Flow程序的结构、还有几种常用的节点，以及基本的使用，应该具备基本的构建Tensor FLow计算图并且能够进行图计算的能力。

这一部分内容将介绍如何使用TensorFlow中的API，训练，优化模型。

Tensor Flow提供了optimizers对象，优化定义的损失函数，其中最简单的就是梯度下降，比如

```
optimizer = tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

sess.run(init)
for i int range(1000)
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W,b]))
```

这段代码，会接受x和y的输入，并且执行1000迭代梯度下降，返回w和b的输出，也就是优化的参数W和偏执b。

完整的程序如：

```
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # initialize variables with incorrect defaults.
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```

## estimator
这个对象是Tensor Flow中的高级API，能够简化机器学习的技术，比如包括：
- 运行训练循环；
- 运行评价循环
- 管理数据集

这部分用到的时候在看。

