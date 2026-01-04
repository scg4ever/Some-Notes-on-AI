# Transformer

**输入**

 传统network问题：只能处理单个输入，不能处理批量输入（sequence）

我们想用sequence作为输入

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251128162620618.png" alt="image-20251128162620618" style="zoom:33%;" />

Vector Set as Input:

用向量表示词汇？

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129105350435.png" alt="image-20251129105350435" style="zoom:50%;" />

Sound,Graph...都可以分解为向量

**输出**

![image-20251129110042390](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110042390.png)

输入n个向量，输出n个label

![image-20251129110053163](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110053163.png)

仅输出1个label

![image-20251129110158862](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110158862.png)

输出由模型决定

Sequence Labeling(第一种)

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110436305.png" alt="image-20251129110436305" style="zoom:50%;" />

对于Fully-connected Network，两个saw的输出应当是一样的。有没有可能不一样呢？考虑上下文。

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110606278.png" alt="image-20251129110606278" style="zoom:50%;" />

## Self-attention

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129110821744.png" alt="image-20251129110821744" style="zoom:50%;" />

Self-attention处理整个sequence的信息，FC处理某一个位置的信息。

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112145204.png" alt="image-20251129112145204" style="zoom:50%;" />

如何产生b1？

1.根据a1，找出输入序列中和a1相关的其他向量（用$\alpha$形容关联程度）

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112319582.png" alt="image-20251129112319582" style="zoom:50%;" />



<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112420542.png" alt="image-20251129112420542" style="zoom:50%;" />

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112604659.png" alt="image-20251129112604659" style="zoom:50%;" />

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112636922.png" alt="image-20251129112636922" style="zoom:50%;" />

q:query k:key v:value

$\alpha $实际上就是q，k的距离（相关程度）。每一个输入都有对应的q，k，v。$\alpha$ 大就说明q，k离得更近，输出中，这一组对应的v所占的权重就更大

![image-20251129112743912](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129112743912.png)
$$
b^1 = \sum_{i=1}^{n}\alpha'_{1,i}v^i
$$
<img src="C:\Users\sc_4ever\Documents\WeChat Files\wxid_m0g389jbs1qg12\FileStorage\Temp\52e2e69f0a40294f514f8b70789f3f0.jpg" alt="52e2e69f0a40294f514f8b70789f3f0" style="zoom:50%;" />

总结：

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129113603364.png" alt="image-20251129113603364" style="zoom:50%;" />

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129113910587.png" alt="image-20251129113910587" style="zoom:50%;" />

A:Attention

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129114107245.png" alt="image-20251129114107245" style="zoom:50%;" />`

O:Output

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251130122228118.png" alt="image-20251130122228118" style="zoom:50%;" />

## Multi-head Self-attention

可能有不同的q，负责不同的相关性

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129122248872.png" alt="image-20251129122248872" style="zoom:50%;" />

各channel分开计算

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251130123009370.png" alt="image-20251130123009370" style="zoom:50%;" />

**positional encoding**

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129122620058.png" alt="image-20251129122620058" style="zoom:50%;" />

每个position有一个独立的ei

在输入中添加时序信息

**Self-attention vs CNN**

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129123939380.png" alt="image-20251129123939380" style="zoom:50%;" />

CNN在训练数据小的时候效果更好

Self-attention在训练数据多的时候效果更好，训练数据少的话容易产生过拟合

## Transformer

a <u>seq2seq</u> model

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129151114003.png" alt="image-20251129151114003" style="zoom:50%;" />

**Encoder：**

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129151634637.png" alt="image-20251129151634637" style="zoom:50%;" />

一个block的工作可以通俗理解为：

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129151756004.png" alt="image-20251129151756004" style="zoom:50%;" />

实际上是

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129151824781.png" alt="image-20251129151824781" style="zoom:50%;" />

**Decoder**（Auto-regressive）

![image-20251129214530295](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129214530295.png)

masked Self-attention:不能看到未来的query 

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129214939884.png" alt="image-20251129214939884" style="zoom:50%;" />

为什么decoder要用masked self-attention呢

因为在输入时(encoder 层)，可以一次接收所有sequence的输入，进行输出。但是decoder的输入是一个一个的，具有时序。

decoder不会自己停下来，需要END 

**Encoder-Decoder**

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129220303516.png" alt="image-20251129220303516" style="zoom:50%;" />

cross-attention:q来自decoder，k&v来自encoder，进行一些weighted sum，得到接下来decoder中FC层的input

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251129220707596.png" alt="image-20251129220707596" style="zoom:50%;" />

**FFN**

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251130153918958.png" alt="image-20251130153918958" style="zoom:50%;" />

**Embedding**





















