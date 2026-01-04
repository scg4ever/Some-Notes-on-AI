# World Models

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204105901081.png" alt="image-20251204105901081" style="zoom: 80%;" />

1.VAE Model（V）

![image-20251204110134357](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204110134357.png)

Input：2D image frame from a video sequence

主要负责视觉，当下。产生latent action $z$

2.MDN-RNN Model（M）

![image-20251204110352616](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204110352616.png)

RNN: model $P(z_{t+1}|a_t,z_t,h_t)$

$a_t$: action at time t,from controller model C

$h_t$: hidden state of the RNN at time t

$\tau$: temperature parameter,和系统随机性有关。

3.Controller Model (C)

决定如何行动才能使累计回报最大。

![image-20251204111349067](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204111349067.png)

将$z_t,h_t$ 映射到 $a_t$ , 用一个单线性层。

4.Putting together

![image-20251204111512633](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204111512633.png)

# VideoGPT

对于学习视频生成来说，选择什么模型比较好？

likelihood-based VS adversarial ? 前者好，在优化和估计上更容易。

autoregressive？在离散的数据上表现很好。

要不要在压缩了冗余之后的潜在空间中执行自回归？要。因为自然图像/视频包含很多冗余，可以训练编码器进行降采样，数据量减小以后可以提高计算速度。

<img src="C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251204154844151.png" alt="image-20251204154844151" style="zoom: 80%;" />

**Learning Latent Codes**

训练一个VQ-VAE来学习视频的数据。包括一系列时空降采样，后面跟着注意力模块和残差链接模块

Axial Attention：标准的global注意力机制在处理3D数据时计算量很大。轴向注意力机制将计算分解，沿不同的轴分别进行注意力计算。这样既能捕捉长距离的全局依赖，也降低了计算成本

**Learning a Prior**

在上一步已经学到的code的基础上（单词），这一步期望学到这些code是如何组合的（语法），也就是这些潜在代码的概率分布。

这一步中应用了一些条件（conditioning）：

​	1.交叉注意力 cross attention

​		对于视频帧的生成来说，首先把作为条件的视频帧输入到一个 3D ResNet中，然后使用交叉注意力机制，在训练网络时确保生成的内容和输入的视频帧在视觉上是连贯的。

​	2.条件归一化 conditional norms

​		对于动作或类别这种标签时，通过一个仿射函数，根据输入的条件向量，改变Transformer中的Layer Norm 中的Gain参数或者Bias参数。



# VAE

定义一下问题

<img src="C:\Users\sc_4ever\Documents\WeChat Files\wxid_m0g389jbs1qg12\FileStorage\Temp\9c062b09aa756722c7ab1234e3c2867.jpg" alt="9c062b09aa756722c7ab1234e3c2867" style="zoom:33%;" />

AE-Autoencoder：可以做特征提取，但是不好做生成。

为什么不好做生成呢？核心问题是它对隐空间的特征没有进行约束，这样我们没办法从隐空间采集"新"的 z 来生成新图像。

——假设我们有一个set : $\{x_1,x_2,...,x_n\}$, 对应着latent space$\{z_1,z_2,...,z_n\}$。要生成的话，我们期望通过一个新的、未出现过的 $z_{n+1}$ 来解码 。这个$x \to z$ 的过程不是"确定"的，所以我们很难预测 $z$  的分布。我们不好随便挑 $z_{n+1}$ ,我们期望 $z$ 最好服从一个分布。

可以假设 $p(z) \sim N(0,1)$ , 但是直接假设的话我们得不到隐空间的点z和真实数据集x的对应关系，很难训encoder。

可以再考虑 $p_\theta(z|x_i)$, 如果我们清楚了对于每一个输入 $x_i$ ,$z$ 的分布，我们就可以从这个分布中取一个$z_i$ ，输入到解码器中，得到 $x$ 的一个预测值 $\hat{x}$ , 可以让这个预测值和输入$x_i$ 做重构损失。这个损失为什么是合理的呢？因为它来自于输入 $x_i$ 的条件概率。

我们的任务，从找 $z_i$ 变成了找这个分布:  $p_\theta(z|x_i)$。



通过变分贝叶斯（Variational Bayes）的方法来找这个分布。

<img src="C:\Users\sc_4ever\Documents\WeChat Files\wxid_m0g389jbs1qg12\FileStorage\Temp\a21e091a7ce5e6192b8bafe06225002.jpg" alt="a21e091a7ce5e6192b8bafe06225002" style="zoom:50%;" />

想找到离 $p_\theta(z|x_i)$ 这个分布最近的分布：$q_\phi(z|x_i)$

 衡量两个分布的距离可以通过KL散度（相对熵）。

![image-20251205203645364](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251205203645364.png)

![image-20251205203708226](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251205203708226.png)
$$
\begin{aligned}
KL(q_\phi(z|x_i)||p_\theta(z|x_i)) &= \int q_\phi(z|x_i) log \frac{q_\phi(z|x_i)}{p_\theta(z|x_i)}dz \\
&=\int q_\phi(z|x_i)logq_\phi(z|x_i)dz-\int q_\phi(z|x_i)logp_\theta(z|x_i)dz \\ 
&=E_q[logq_\phi(z|x_i)]-E_q[q_\phi(z|x_i)logp_\theta(z,x_i)]+E_q[logp_\theta(x_i)]\\
&=-ELBO + E_q[logp_\theta(x_i)]
\end{aligned}
$$
即
$$
logp(x_i) = KL +ELBO
$$
$logp(x_i)$ 是定值，如果我们想最小化KL散度（让两个分布离得最近），就要最大化ELBO。
$$
\begin{aligned}
ELBO &= E_q[logp_\theta(z,x_i)]-E_q[logq_\phi(z|x_i)]\\
&=E_q[logp_\theta(x_i|z)p_\theta(z)]-E_q[logq_\phi(z|x_i)]\\
&=E_q[logp_\theta(x_i|z)]+E_q[logp_\theta(z)]-E_q[logq_\phi(z|x_i)]\\
&=-E_q[log\frac{q_\phi(z|x_i)}{p(z)}]+E[logp_\theta(x_i|z)]\\
&=-KL(q_\phi(z|x_i)||p(z))+E[logp_\theta(x_i|z)]
\end{aligned}
$$
最大化ELBO，就要最小化KL，最大化$E[logp_\theta(x_i|z)]$

我们最开始想要$q_\phi(z|x_i)$逼近于$p_\theta(z|x_i)$，事实上，经过推导现在已经等价于让$q_\phi(z|x_i)$逼近这个先验分布$p(z)\sim N(0,1)$,并且$E[logp_\theta(x_i|z)]$最大。注意到$p_\theta(x_i|z)$ 这部分实际上就是decoder。

<img src="C:\Users\sc_4ever\Documents\WeChat Files\wxid_m0g389jbs1qg12\FileStorage\Temp\435006a9e76b76f212c72d2a503decf.jpg" alt="435006a9e76b76f212c72d2a503decf" style="zoom: 50%;" />

