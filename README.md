### CNN for CIFAR10 in C++（神经网络底层原理&算法，源代码介绍）





![image-20211127120201839](C:\Users\Wang Zihan\AppData\Roaming\Typora\typora-user-images\image-20211127120201839.png) 

#### Forward Pass

1. Hidden Layer得到上一层的输出a'作为本层的输入，或Input Layer得到整个网络的输入。
2. 当前层得到输入后首先经过线性计算(如Convolution Layer的卷积运算或Fully Connected Layer)，得到Z。
3. 线性运算的结果Z经过Activation Function(激活函数如Sigmoid、ReLU、tanh等)，得到a。
4. 激活函数的结果a最终作为本层输出并作为下一层的输入。

#### Backward Pass

总目标：求得Loss函数对本层Unknown Parameter的梯度(所有未知参数的偏导数组成的向量)，$\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$

然后对参数更新。如上图所示，W和b在线性运算中。我们首先想到，可以像做高数题那样，先通过网络计算出一个以W和b为未知数的Loss函数，然后分别对每一个Unknown Parameter求偏导，但这不是计算思维，带有大量未知参数的表达式计算机无法存储也难以计算偏导，于是出现了Backpropagation，它包括Forward Pass(先)和Backward Pass(后)，那我们下面来看看为了求得$\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$，Backward Pass都做了哪些工作。

1. 对于任意一层，我们首先假设，$\frac{\partial L}{\partial a}$是已知的。为什么可以作这样的假设？如果我们是在Output Layer，a就是整个网络的输出y，而y和Label可以得到Loss函数的表达式，用该表达式对y求偏导，再代入Forward Pass中求得的y和已知的label，就可以求得最后一层Loss对a的偏导，Backward Pass也正是从这里开始的。我们会逐步得到任意一层的$\frac{\partial L}{\partial a}$，$\frac{\partial L}{\partial a}$也叫作Backward Error。

2. 有了$\frac{\partial L}{\partial a}$，我们再求当前层的$\frac{\partial L}{\partial Z}$，根据链式法则，显然我们需要$\frac{\partial a}{\partial Z}$，这实际上就是Activation Function的导数，值得注意的是，求$\frac{\partial a}{\partial Z}$需要Forward Pass中求得的当前层的Z的值。$\frac{\partial L}{\partial Z}$也叫做Layer Error或Delta。

   [$\frac{\partial L}{\partial Z}$=$\frac{\partial L}{\partial a}$ *$\frac{\partial a}{\partial Z}$]

3. 有了$\frac{\partial L}{\partial Z}$，离我们的目标$\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$更近了，显然我们需要$\frac{\partial Z}{\partial W}$和$\frac{\partial Z}{\partial b}$。我们先忽略$\frac{\partial Z}{\partial b}$的计算，关注$\frac{\partial Z}{\partial W}$。在全连接层中，Z = W * a；在卷积层中，因为Receptive Field和Parameter Sharing，W的某些部分与a的某些部分相乘得到Z，总之，W和a通过简单的乘加运算得到Z，因此$\frac{\partial Z}{\partial W}$的结果就来自a。有了$\frac{\partial Z}{\partial W}$，我们就可以更新W，W‘ = W - Learning Rate * $\frac{\partial L}{\partial W}$。$\frac{\partial L}{\partial W}$、$\frac{\partial L}{\partial b}$也叫dW、db。

   [$\frac{\partial L}{\partial W}$=$\frac{\partial L}{\partial Z}$ *$\frac{\partial Z}{\partial W}$]

4. 既然得到了本层的$\frac{\partial Z}{\partial W}$，那Backward Pass是否可以结束了？显然不可以，如果结束了，那该层的上一层怎么办呢？于是我们希望再次得到$\frac{\partial L}{\partial a'}$，上一层再继续重复1，2，3步骤，因为2中我们得到了$\frac{\partial L}{\partial Z}$，显然我们需要$\frac{\partial Z}{\partial a'}$，与3类似，$\frac{\partial Z}{\partial a'}$就来自于W，$\frac{\partial L}{\partial a'}$也叫作Backward Error，至此我们可以对上一层继续Backward Pass。

   [$\frac{\partial L}{\partial a'}$=$\frac{\partial L}{\partial Z}$ *$\frac{\partial Z}{\partial a'}$]

值得注意的是：

- Backward Pass过程中，需要Forward Pass中计算的Z和a，使用Z计算$\frac{\partial a}{\partial Z}$进而[$\frac{\partial L}{\partial Z}$=$\frac{\partial L}{\partial a}$ *$\frac{\partial a}{\partial Z}$]得到Layer Error(Delta)，使用上一层的a计算$\frac{\partial Z}{\partial a'}$进而[$\frac{\partial L}{\partial a'}$=$\frac{\partial L}{\partial Z}$ *$\frac{\partial Z}{\partial a'}$]得到Backward Error。因此Z和a需要在Forward Pass中保存下来。
- 首先有本层的Backward Error，然后结合本层的Z和Activation Funtion获得本层的Layer Error，使用上一层的a和本层的Layer Error进行update，使用本层的Unknown Parameter和本层的Layer Error获得上一层的Backward Error(Maxpooling特殊)。
- 无论是怎样的Layer，在当前层的Backward Error已知的情况下，求得Layer Error的过程都是根据激活函数，都是类似的[$\frac{\partial L}{\partial Z}$=$\frac{\partial L}{\partial a}$ *$\frac{\partial a}{\partial Z}$]。
- 但不同的Layer得到$\frac{\partial L}{\partial W}$和$\frac{\partial L}{\partial a'}$是有区别的，原因是$\frac{\partial Z}{\partial W}$和$\frac{\partial Z}{\partial a'}$得到的方式不同，下面开始逐个介绍。

#### Convolution Layer Backward Pass

在卷积层中，上一层的输入Input a'是一个矩阵X(我们以单通道即1 channel为例)，待更新的参数W叫做卷积核(Kernel或Filter，假设只有一个卷积核，卷积核的厚度应当等于输入通道数 = 1)，输出是一个单通道矩阵O，我们通过上述的1，2操作，可以得到$\frac{\partial L}{\partial Z}$，也就是$\frac{\partial L}{\partial O}$。有下面公式：
$$
\frac{\partial L}{\partial F} = Convolution(Input\ X,\  Layer\ Error\ \frac{\partial L}{\partial O})
$$

$$
\frac{\partial L}{\partial X} = Full\ Convolution(180°\ rotated\ F,\  Layer\ Error\ \frac{\partial L}{\partial O})
$$

具体可见：[How does Backpropagation work in a CNN?](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c) 

在我的代码中并没有在卷积层的反向传播过程中使用卷积操作，因为没能准确地把握多通道，多核的情况，网上的例子大都是单通道和单核的，后续捋清楚了再更新代码。

这份代码中用的多层循环，因为都是加乘操作，O对X的偏导都来自Filter，O对Filter的偏导都来自X，通过循环找到待求导变量的系数累加，效率较低。

 #### Pooling Layer Backward Pass

1. 池化层没有Activation Funtion，因此Z = a，$\frac{\partial L}{\partial Z}$ = $\frac{\partial L}{\partial a}$，即Backward Error = Layer Error。
2. 池化层没有Unknown Parameters，因此不需要求$\frac{\partial L}{\partial W}$。
3. 但池化层必须完成$\frac{\partial L}{\partial a'}$,求出上一层的Backward Error进行反向传播。
4. 对于最大池化，得到$\frac{\partial L}{\partial Z}$ 即 $\frac{\partial L}{\partial a}$后，求$\frac{\partial L}{\partial a'}$，a'的尺寸大于a，因此$\frac{\partial L}{\partial a'}$的尺寸要大于$\frac{\partial L}{\partial a'}$ ，$\frac{\partial L}{\partial a'}$中，在a'被采样的位置的导数值与$\frac{\partial L}{\partial a}$相等，其余位置为0。
5. 对于平均池化，将$\frac{\partial L}{\partial a}$的值平均到池化窗口中再填回到$\frac{\partial L}{\partial a'}$中即可。

具体可见：[序号3](https://www.cnblogs.com/pinard/p/6494810.html)

#### Fully Connected Layer Backward Pass

<img src="C:\Users\Wang Zihan\AppData\Roaming\Typora\typora-user-images\image-20211127172035176.png" alt="image-20211127172035176" style="zoom:80%;" />

  

#### About Padding in Backward Pass

1. Padding出现在Convolution Layer，当然Pooling Layer也可以Padding，但我的代码中没有考虑。
2. 如果Forward Pass时有Padding，在求上一层的Backward Error的时候，求得的矩阵中是包含对Padding的Zero求导的，要先把这些Zero的偏导数求出来然后再去掉送给上一层。
3. 如果Forward Pass时有Padding，在求dW(即对Filter求偏导)时，要把a'(即X)先Padding再与本层的Layer Error作相关操作。

#### the Neural Network Structure for CIFAR10

<img src="C:\Users\Wang Zihan\AppData\Roaming\Typora\typora-user-images\image-20211127174656382.png" alt="image-20211127174656382" style="zoom:50%;" />

<img src="C:\Users\Wang Zihan\AppData\Roaming\Typora\typora-user-images\image-20211127175001292.png" alt="image-20211127175001292" style="zoom:60%;" />

代码中在Convolution Layer和Fully Connected Layer后都加了一个Activation Funtion，卷积层加了ReLU，全连接层加了Sigmoid。

#### Process CIFAR10 Dataset For C++

CIFAR10的数据集文件是二进制文件，C++直接读取比较麻烦，我先用Pytoch的DataLoader将图片转化为Tensor直接存到文本文件中，C++只需要从文本文件中读RGB对应的数值即可。相关程序在CIFAR10_for_C++.py中。

#### Code structure

<img src="C:\Users\Wang Zihan\AppData\Roaming\Typora\typora-user-images\image-20211127175446546.png" alt="image-20211127175446546" style="zoom:50%;" />

1. Array2d用于全连接层，具有行、列属性，用一个vector存储全部值，将二维索引映射到一维；其中实现了向量的一些基本操作，如行点积，列点积。
2. Array3d用于卷积层和池化层，具有宽、高、通道属性，用一个vector存储全部值，将三维索引映射到一维；其中实现了一些矩阵基本操作，加减乘除等。
3. 三种层，Convolution Layer、Fully Connected Layer、Maxpooling Layer，每一层Compute函数和activate函数进行Forward Pass，通过gradient_L_to_Z函数来获取Layer Error，通过Backward函数计算上一层的Backward Error，通过Update函数更新Unknown Parameter(除了Maxpooling)。
4. ReLU和Sigmoid不单独作为层，分别内置在Convolution Layer和Fully Connected Layer中。
5. MSE和Cross Entropy是两个Loss Funtion，其中的start_backward函数计算Output Layer的Backward Error，由此开始反向传播过程。
6. CNN负责将main声明的网络模型中的层衔接起来。train函数负责在训练集上训练，包括了正向和反向传播；Predict函数和test_accuracy函数共同在测试集上计算正确率，仅有正向传播过程。
7. db_handler处理数据集，读取CIFAR10_for_C++.py处理后的文件。





