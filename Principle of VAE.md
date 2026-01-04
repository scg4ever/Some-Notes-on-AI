# Principle of VAE

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