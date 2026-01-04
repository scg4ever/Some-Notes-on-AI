# Diffusion

## DDPM——Denoising Diffusion Probabilistic Models

 a parameterized Markov chain

前向过程：给图片加噪声；反向过程：给图片去噪声

可以理解为一种更具体的VAE：编码过程：加噪声；解码过程：学习怎样消除之前添加的噪声



### Forward

来自训练集的 $x_0$ 被加 $T$ 次噪声.

加噪声：从均值与上一时刻有关的一个正态分布中<u>重新采样</u>，即$x_t$ 是从均值与$x_{t-1}$ 有关的正态分布中采样得到。

即
$$
x_t \sim N(\mu_t(x_{t-1}),\sigma_tI)
$$
通常可以设置为：
$$
x_t \sim N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)
$$
已知 $x_0$ , 如何得到 $x_t$ 呢？由 $x_t$ 的分布，可以表示出它的形式
$$
\begin{aligned}
x_t &= \sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\varepsilon_{t-1}\\
&=\sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}}\varepsilon_{t-2})+\sqrt{\beta_t}\varepsilon_{t-1}\\
&=...\\

\end{aligned}
$$
令$\alpha_t = 1 - \beta_t$
$$
\begin{aligned}
x_t &= \sqrt{\alpha_t\alpha_{t-1}...\alpha_1}x_0+\sqrt{1-\alpha_t\alpha_{t-1}...\alpha_1}\varepsilon\\
&=\sqrt{\overline{\alpha_t}}x_0+\sqrt{1-\overline{\alpha_t}}\varepsilon
\end{aligned}
$$
**Discussion**

经过这一步，我们就完成了从原图加噪声的过程。

可以看出 : $0<\alpha_t,\beta_t<1$

单个加噪过程中 $\beta_t$ 是恒定的。随着$\beta$ 增大，$\alpha$ 减小，结果中 $\varepsilon$ （噪声）的占比越来越多，原图片的信息减少。

### Backward

对被加噪声的图片，执行加噪声的“逆操作”。让每一步的“去噪”对应先前的“加噪”，最终达到去除噪音恢复原图的效果。

现在的case是，我们已知 $x_t$（刚刚被加完噪音的图片），已知 $x_0$ （原图片，并且是去噪后的目标图片），想要一步一步从 $x_t$ 推回 $x_0$

也就是想求这个分布：$q(x_{t-1}|x_0,x_t)$ 

我们目前已知的是正向过程，也就是这些分布：$q(x_t|x_0)$,$q(x_{t-1}|x_0)$,$q(x_{t}|x_{t-1})$ 想办法和$q(x_{t-1}|x_0,x_t)$ 联系起来

在此之前先推一个贝叶斯公式：
$$
\begin{aligned}
P(A|B,C) &= \frac{P(A,B,C)}{P(B,C)}(条件概率)\\
&=\frac{P(C|A,B)P(A,B)}{P(C|B)P(B)}\\
&=P(C|A,B)\frac{P(A|B)}{P(C|B)}
\end{aligned}
$$
对应回上式，就是说
$$
q(x_{t-1}|x_0,x_t)=q(x_t|x_{t-1},x_0)\frac{q(x_{t-1}|x_0)}{q(x_{t}|x_0)}
$$
实际上，加噪声是一个Markov过程，也就是t时刻只和(t-1)时刻有关，和0时刻无关。上式可以改写为
$$
q(x_{t-1}|x_0,x_t)=q(x_t|x_{t-1})\frac{q(x_{t-1}|x_0)}{q(x_{t}|x_0)}
$$
右侧的三个分布我们都可以写出来：
$$
q(x_t|x_{t-1}) \sim N(\sqrt{1-\beta_t}x_{t-1},\beta_tI)
$$

$$
q(x_{t-1}|x_0) \sim N(\sqrt{\bar{\alpha}_{t-1}}x_0,\sqrt{1-\bar{\alpha}_{t-1}}I)
$$

$$
q(x_{t}|x_0) \sim N(\sqrt{\bar{\alpha}_{t}}x_0,\sqrt{1-\bar{\alpha}_{t}}I)
$$

右侧就是三个高斯分布的乘积，经过展开后系数比对，得到：
$$
q(x_{t-1}|x_0,x_t)\sim N(\tilde{\mu}_t,\tilde{\beta}_tI)
$$
其中

![image-20251210180017105](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251210180017105.png)

![image-20251210180029949](C:\Users\sc_4ever\AppData\Roaming\Typora\typora-user-images\image-20251210180029949.png)