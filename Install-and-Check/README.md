# 安装TensorFlow

## 安装

### 环境

1. Windows10
1. NVIDA GF 1080ti
1. 256 固态盘
1. 10T 机械盘
1. E5处理器 * 2

> 本想用另一块盘做Ubuntu系统，但是为了测试GPU在Win7 系统上的运行状态，临时改装了Win7系统，所以只能在当前的Win10系统上配置TensorFlow，用来可见光部分的天空背景目标检测与识别，与红外设备配合做天空背景运动小目标检测与跟踪。

### 网络

连不上TensorFlow的官网一个大问题，两种解决方法：用公司的VPN服务，连接外网；用个人使用SS加速科学上网。

### 过程

打开TensorFlow主页，安装步骤简洁，简答的几个步骤就完事了，前提是网络问题解决了。

默认已经安装好了Anaconda，修改了国内的镜像源，并且已经下载了cuDNN，并且添加到了环境变量。

- 创建TensorFlow的虚拟环境

``` shell
conda create -n tensorflow python=3.5 
```

- 激活环境

``` shell
activate tensorflow
```

- 在Windows上powershell与anaconda不兼容，我使用的git bash，并且使用的是bash解释器，激活环境的命令需要添加`source`在前面。

``` shell
source activate tensorflow
```

- 安装TensorFlow

``` shell
pip install --ignore-installed --upgrade tensorflow-gpu 
```

## 测试

创建一个Python脚本，把安装主页上的几行测试代码写进去，运行这个脚本就行了，也可以在bash里面交互式运行，但是我喜欢把它弄成脚本。

``` python
import tensorflow as tf
hello=tf.constant('Hello, tensorflow')
sess = tf.Session()
print(sess.run(hello))
```

下面是运行结果：

``` shell
$ python tf.py
b'Hello, tensorflow'
2017-10-20 11:18:58.221443: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-10-20 11:18:58.221467: W C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-10-20 11:18:58.692986: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:955] Found device 0 with properties:
name: GeForce GTX 1080 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.607
pciBusID 0000:02:00.0
Total memory: 11.00GiB
Free memory: 9.08GiB
2017-10-20 11:18:58.693012: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:976] DMA: 0
2017-10-20 11:18:58.693018: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:986] 0:   Y
2017-10-20 11:18:58.693052: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\35\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0)
```

另外，官网还提供可一个Checker文件，地址在[GithubGist](https://gist.github.com/mrry/ee5dbcfdd045fa48a27d56664411d41c).

要是安装没有问题的话，会输出下面的结果：

``` shell
$ python tensorflow_self_check.py
TensorFlow successfully installed.
The installed version of TensorFlow includes GPU support.
```