## 题目1

\[
\frac{x-1}{x+2} \geq 2
\]

### 解答

先把不等式右边的 2 移到左边，通分化简：

\[
\frac{x-1}{x+2} - 2 \ge 0
\;\Longrightarrow\;
\frac{x-1-2(x+2)}{x+2} \ge 0
\;\Longrightarrow\;
\frac{-x-5}{x+2} \ge 0
\;\Longrightarrow\;
\frac{x+5}{x+2} \le 0 .
\]

令 \(g(x)=\dfrac{x+5}{x+2}\)。  
- 零点（分子为零）：\(x=-5\)。  
- 极点（分母为零）：\(x=-2\)（此点不在定义域）。  

在区间 \((-∞,-5),(-5,-2),(-2,∞)\) 检验符号：

| 区间 | \(x+5\) | \(x+2\) | \(\dfrac{x+5}{x+2}\) |
|------|--------|--------|----------------------|
| \((-∞,-5)\) | 负 | 负 | 正 |
| \((-5,-2)\) | 负 | 正 | 负 |
| \((-2,∞)\)   | 正 | 正 | 正 |

满足 \(\le 0\) 的区间为 \([-5,-2)\)。  
在端点 \(x=-5\) 时原式等于 2，满足不等式；\(x=-2\) 不在定义域。

\[
\boxed{x\in[-5,\,-2)}
\]

### 评价

题目解答正确。步骤清晰，处理分母和符号变化得当，结果准确。

答案：
\[
\boxed{x\in[-5,\,-2)}
\]

## 题目2

已知数列 \(\{a_n\}\) 满足 \(a_1 = 1,\; a_{n+1}=2a_n+n\)，求 \(\{a_n\}\) 的通项公式。

### 解答

该递推式为线性非齐次递推，可先求齐次解再求特解。

**1. 齐次方程**  
\(a_{n+1}^{(h)}=2a_n^{(h)}\)  
解得  
\[
a_n^{(h)}=C\cdot2^{\,n-1}\qquad(C\in\mathbb{R}).
\]

**2. 特解**  
右端为 \(n\)，设特解为一次多项式 \(a_n^{(p)}=\alpha n+\beta\)。  
代入递推式：

\[
\alpha (n+1)+\beta =2(\alpha n+\beta)+n
\;\Longrightarrow\;
\alpha n+\alpha+\beta = (2\alpha+1)n+2\beta .
\]

比较系数得  

\[
\begin{cases}
\alpha =2\alpha+1\\[2pt]
\alpha+\beta =2\beta
\end{cases}
\Longrightarrow\ \alpha=-1,\;\beta=-1 .
\]

于是 \(a_n^{(p)}=-n-1\)。

**3. 通解**  

\[
a_n=C\cdot2^{\,n-1}-n-1 .
\]

利用初始条件 \(a_1=1\) 求 \(C\)：

\[
1=C\cdot2^{0}-1-1 \;\Longrightarrow\; C=3 .
\]

\[
\boxed{a_n = 3\cdot2^{\,n-1}-n-1}\qquad (n\ge1).
\]

### 评价

该解答过程正确。齐次方程的解法、特解的假设和求解步骤均无误，验证初始条件后得到正确的通项公式。

答案：

$\boxed{a_n = 3 \cdot 2^{n-1} - n - 1}$

## 题目3

求函数的导数  
(1) \(f(x)=x^{3}\ln x\)  
(2) \(f(x)=\dfrac{e^{x}}{x^{2}+1}\)

### 解答

**(1)**  

\[
\begin{aligned}
f'(x) &=\frac{d}{dx}\bigl(x^{3}\ln x\bigr) 
      =3x^{2}\ln x + x^{3}\cdot\frac{1}{x}  \\
      &=3x^{2}\ln x + x^{2}
      =x^{2}\bigl(3\ln x+1\bigr).
\end{aligned}
\]

**(2)**  

\[
\begin{aligned}
f'(x)&=\frac{(e^{x})'(x^{2}+1)-e^{x}(x^{2}+1)'}{(x^{2}+1)^{2}}\\
     &=\frac{e^{x}(x^{2}+1)-e^{x}\cdot2x}{(x^{2}+1)^{2}}\\
     &=\frac{e^{x}\bigl(x^{2}-2x+1\bigr)}{(x^{2}+1)^{2}}
      =\frac{e^{x}(x-1)^{2}}{(x^{2}+1)^{2}}.
\end{aligned}
\]

### 评价

(1) 答案正确。使用了乘积法则，并且计算和化简均无误。

(2) 答案正确。应用了商的求导法则，化简过程准确，最终结果正确。

整体上解答步骤清晰，答案正确。

## 题目4

计算 \(\displaystyle \int x^{2}e^{x}\,dx\)。

### 解答

采用分部积分：

\[
\begin{aligned}
\int x^{2}e^{x}\,dx
&=x^{2}e^{x}-\int 2x e^{x}\,dx   \qquad(u=x^{2},\ dv=e^{x}dx)\\[4pt]
&=x^{2}e^{x}-\Bigl(2x e^{x}-\int 2e^{x}\,dx\Bigr)\\[4pt]
&=x^{2}e^{x}-2x e^{x}+2e^{x}+C\\[4pt]
&=e^{x}\bigl(x^{2}-2x+2\bigr)+C .
\end{aligned}
\]

\[
\boxed{\displaystyle \int x^{2}e^{x}\,dx=e^{x}\bigl(x^{2}-2x+2\bigr)+C }.
\]

### 评价

答案正确且步骤清晰，两次分部积分应用得当，结果无误。

答案：

计算 \(\int x^{2}e^{x}\,dx\) 使用分部积分法是正确的。答案详细展示了每一步的过程，并最终得到正确结果 \(e^{x}(x^{2} - 2x + 2) + C\)。

\[
\boxed{e^{x}(x^{2} - 2x + 2) + C}
\]

## 题目5

计算 \(\displaystyle I=\int_{0}^{1}x^{2}\sqrt{1-x^{2}}\,dx\)。

### 解答

令 \(x=\sin\theta\;(0\le\theta\le\frac{\pi}{2})\)，则  

\[
dx=\cos\theta\,d\theta,\qquad \sqrt{1-x^{2}}=\cos\theta .
\]

于是

\[
\begin{aligned}
I &=\int_{0}^{\pi/2}\bigl(\sin^{2}\theta\bigr)\,\cos^{2}\theta\,d\theta \\
  &=\int_{0}^{\pi/2}\sin^{2}\theta\cos^{2}\theta\,d\theta .
\end{aligned}
\]

利用 \(\sin^{2}\theta\cos^{2}\theta=\frac14\sin^{2}2\theta\)：

\[
\begin{aligned}
I &=\frac14\int_{0}^{\pi/2}\sin^{2}2\theta\,d\theta 
   =\frac14\cdot\frac12\int_{0}^{\pi}\sin^{2}u\,du
     \qquad(u=2\theta,\ du=2d\theta)\\
  &=\frac18\Bigl[\frac{u}{2}-\frac{\sin2u}{4}\Bigr]_{0}^{\pi}
   =\frac18\cdot\frac{\pi}{2}
   =\frac{\pi}{16}.
\end{aligned}
\]

\[
\boxed{I=\dfrac{\pi}{16}} .
\]

### 评价

解答正确且完整。积分计算过程中应用了三角替换和适当的恒等式变换，并正确地进行了变量替换与积分上下限的调整，最终得到的结果准确无误。

评价：答案正确，步骤清晰。

## 题目6

设 \(f(x,y)=x^{3}+y^{3}-3xy\)，求  
(1) 所有驻点；  
(2) 判定这些驻点是极大值、极小值还是鞍点。

### 解答

**(1) 求驻点**  

\[
f_x = 3x^{2}-3y =0 \;\Longrightarrow\; y = x^{2},
\]
\[
f_y = 3y^{2}-3x =0 \;\Longrightarrow\; x = y^{2}.
\]

代入 \(y=x^{2}\) 得 \(x = (x^{2})^{2}=x^{4}\)，即 \(x^{4}-x=0\)，解得 \(x=0\) 或 \(x=1\)。

\[
\begin{cases}
x=0\;\Rightarrow\;y=0,\\
x=1\;\Rightarrow\;y=1.
\end{cases}
\]

故驻点为 \((0,0)\) 与 \((1,1)\)。

**(2) Hessian 矩阵并判定**  

\[
f_{xx}=6x,\qquad f_{yy}=6y,\qquad f_{xy}=f_{yx}=-3.
\]

- 在 \((0,0)\)：

\[
f_{xx}=0,\;f_{yy}=0,\;f_{xy}=-3\;\Longrightarrow\;
\Delta =f_{xx}f_{yy}-(f_{xy})^{2}= -9<0.
\]

\(\Delta<0\) ⇒ \((0,0)\) 为 **鞍点**。

- 在 \((1,1)\)：

\[
f_{xx}=6,\;f_{yy}=6,\;f_{xy}=-3\;\Longrightarrow\;
\Delta =6\cdot6-(-3)^{2}=36-9=27>0,\quad f_{xx}>0.
\]

\(\Delta>0\) 且 \(f_{xx}>0\) ⇒ \((1,1)\) 为 **局部极小点**（事实上是全局最小点）。

\[
\boxed{(0,0)\ \text{鞍点},\;(1,1)\ \text{局部（全局）极小点}} .
\]

### 评价

答案正确，计算过程无误，结果合理。

## 题目7

解微分方程 \(y'+2xy=e^{-x^{2}}\)，并满足初始条件 \(y(0)=1\)。

### 解答

方程是一阶线性常系数非齐次方程。先求积分因子：

\[
\mu(x)=\exp\!\bigl(\int 2x\,dx\bigr)=e^{x^{2}}.
\]

两边同乘 \(\mu(x)\)：

\[
e^{x^{2}}y' + 2xe^{x^{2}}y = 1
\;\Longrightarrow\;
\frac{d}{dx}\bigl(e^{x^{2}}y\bigr)=1 .
\]

积分得到

\[
e^{x^{2}}y = x + C \quad\Longrightarrow\quad
y = (x+C)e^{-x^{2}} .
\]

利用初值 \(y(0)=1\)：

\[
1 = (0+C)e^{0}=C \;\Longrightarrow\; C=1 .
\]

于是

\[
\boxed{\,y(x)=\bigl(x+1\bigr)e^{-x^{2}}\, }.
\]

### 评价

正确解答了微分方程并应用初始条件求得解，所有步骤准确无误。

$\boxed{y(x)= (x+1)e^{-x^{2}}}$

## 题目8

设 \(A=\begin{pmatrix}1&2\\3&4\end{pmatrix}\)，求 \(A^{-1}\) 与 \(|A|\)。

### 解答

\[
|A| = 1\cdot4-2\cdot3 = 4-6 = -2.
\]

逆矩阵公式 \(A^{-1}= \frac1{|A|}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}\)：

$A^{-1}= \frac1{-2}
\begin{pmatrix}
4 & -2\\
-3& 1
\end{pmatrix}=\begin{pmatrix}
-2 & 1\\[4pt]
\displaystyle\frac{3}{2}& -\frac12
\end{pmatrix}.
$

\[
\boxed{|A|=-2,\qquad 
A^{-1}= \begin{pmatrix}
-2 & 1\\[4pt]
\frac{3}{2}&-\frac12
\end{pmatrix}} .
\]

### 评价

该解答正确地计算了矩阵 \(A\) 的行列式和逆矩阵。计算步骤清晰，结果准确，并通过验证确认了逆矩阵的正确性。答案格式正确，使用了数学公式并进行了适当的排版。

\[
\boxed{|A| = -2, \quad 
A^{-1} = \begin{pmatrix}-2 & 1 \\ \frac{3}{2} & -\frac{1}{2}\end{pmatrix}}
\]

## 题目9

求矩阵 \(A=\begin{pmatrix}4&1\\2&3\end{pmatrix}\) 的特征值与对应的特征向量。

### 解答

特征方程：

\[
\det(A-\lambda I)=
\begin{vmatrix}
4-\lambda & 1\\
2 & 3-\lambda
\end{vmatrix}
=(4-\lambda)(3-\lambda)-2
= \lambda^{2}-7\lambda+10=0.
\]

\[
\lambda^{2}-7\lambda+10=(\lambda-5)(\lambda-2)=0
\Longrightarrow \lambda_{1}=5,\;\lambda_{2}=2 .
\]

**对应特征向量**  

- \(\lambda_{1}=5\)：

\[
(A-5I)v=0\;\Longrightarrow\;
\begin{pmatrix}
-1&1\\2&-2
\end{pmatrix}
\begin{pmatrix}v_{1}\\v_{2}\end{pmatrix}=0
\Rightarrow -v_{1}+v_{2}=0\;\Longrightarrow\;v\propto\begin{pmatrix}1\\1\end{pmatrix}.
\]

- \(\lambda_{2}=2\)：

\[
(A-2I)v=0\;\Longrightarrow\;
\begin{pmatrix}
2&1\\2&1
\end{pmatrix}
\begin{pmatrix}v_{1}\\v_{2}\end{pmatrix}=0
\Rightarrow 2v_{1}+v_{2}=0\;\Longrightarrow\;v\propto\begin{pmatrix}1\\-2\end{pmatrix}.
\]

\[
\boxed{\lambda_{1}=5,\;v_{1}=\begin{pmatrix}1\\1\end{pmatrix};\quad
\lambda_{2}=2,\;v_{2}=\begin{pmatrix}1\\-2\end{pmatrix}} .
\]

### 评价

解答正确。特征值和特征向量的计算过程无误，步骤清晰且结果准确。

## 题目10

求解线性方程组

\[
\begin{cases}
x_{1}+2x_{2}+3x_{3}=1,\\
2x_{1}+5x_{2}+3x_{3}=3,\\
x_{1}+8x_{3}=3 .
\end{cases}
\]

### 解答

由第三式得 \(x_{1}=3-8x_{3}\)。代入前两式：

\[
\begin{aligned}
(3-8x_{3})+2x_{2}+3x_{3}&=1 \;\Longrightarrow\; 2x_{2}-5x_{3}=-2 \;\Longrightarrow\; x_{2}= -1+\frac52 x_{3},\\[4pt]
2(3-8x_{3})+5x_{2}+3x_{3}&=3 \;\Longrightarrow\; 6-16x_{3}+5x_{2}+3x_{3}=3\\
&\Longrightarrow\;5x_{2}-13x_{3}=-3 .
\end{aligned}
\]

把 \(x_{2}= -1+\frac52 x_{3}\) 代入第二式：

\[
5\!\left(-1+\frac52x_{3}\right)-13x_{3}=-3
\;\Longrightarrow\;-5+\frac{25}{2}x_{3}-13x_{3}=-3
\;\Longrightarrow\;-\frac12 x_{3}=2
\;\Longrightarrow\;x_{3}=-4 .
\]

回代得到

\[
x_{2}= -1+\frac52(-4)= -1-10=-11,\qquad
x_{1}=3-8(-4)=3+32=35 .
\]

\[
\boxed{(x_{1},x_{2},x_{3})=(35,\,-11,\,-4)} .
\]

### 评价

解答过程正确，结果满足原方程组。答案正确。

$\boxed{(35, -11, -4)}$

## 题目11

将二次型  

\[
f(x_{1},x_{2},x_{3})=x_{1}^{2}+2x_{2}^{2}+3x_{3}^{2}+4x_{1}x_{2}-2x_{2}x_{3}
\]

化为标准形（求其特征值并给出对应的正交变换）。

### 解答

二次型的矩阵为  

\[
Q=\begin{pmatrix}
1 & 2 & 0\\
2 & 2 & -1\\
0 & -1 & 3
\end{pmatrix}.
\]

**特征值**  

解特征方程 \(\det(Q-\lambda I)=0\)：

\[
\det\!\begin{pmatrix}
1-\lambda & 2 & 0\\
2 & 2-\lambda & -1\\
0 & -1 & 3-\lambda
\end{pmatrix}=0
\Longrightarrow \lambda^{3}-6\lambda^{2}+6\lambda+7=0.
\]

该三次方程的根为（数值解）  

\[
\lambda_{1}\approx-0.66908,\qquad
\lambda_{2}\approx2.524,\qquad
\lambda_{3}\approx4.14508.
\]

**对应的正交特征向量（归一化）**  

- 对 \(\lambda_{1}=-0.66908\)：

\[
(Q-\lambda_{1}I)v=0 \;\Longrightarrow\;
v_{1}\propto\begin{pmatrix}1\\-1\\-1\end{pmatrix}
\quad\Rightarrow\quad
\hat v_{1}= \frac{1}{\sqrt3}\begin{pmatrix}1\\-1\\-1\end{pmatrix}.
\]

- 对 \(\lambda_{2}=2.524\)：

\[
v_{2}\propto\begin{pmatrix}2\\1\\-2\end{pmatrix}
\quad\Rightarrow\quad
\hat v_{2}= \frac{1}{\sqrt{9}}\begin{pmatrix}2\\1\\-2\end{pmatrix}
 =\frac13\begin{pmatrix}2\\1\\-2\end{pmatrix}.
\]

- 对 \(\lambda_{3}=4.14508\)：

\[
v_{3}\propto\begin{pmatrix}1\\2\\1\end{pmatrix}
\quad\Rightarrow\quad
\hat v_{3}= \frac{1}{\sqrt{6}}\begin{pmatrix}1\\2\\1\end{pmatrix}.
\]

（上述向量满足两两正交，且已单位化。）

设正交矩阵  

\[
P=\bigl[\hat v_{1}\;\hat v_{2}\;\hat v_{3}\bigr],
\qquad
y=P^{\mathsf T}x,
\]

则二次型化为

\[
\boxed{f(x_{1},x_{2},x_{3})=
\lambda_{1}y_{1}^{2}+\lambda_{2}y_{2}^{2}+\lambda_{3}y_{3}^{2}
\approx -0.66908\,y_{1}^{2}+2.524\,y_{2}^{2}+4.14508\,y_{3}^{2}} .
\]

如果只需要一个“标准形”（非正交）而不要求正交变换，也可通过配方法得到

\[
f=(x_{1}+2x_{2})^{2}-2\!\left(x_{2}+\tfrac12x_{3}\right)^{2}
+\frac{7}{2}x_{3}^{2},
\]

这已是 \(x\) 的平方和形式（系数 \(-2,\frac72\) 与特征值对应相同的惯性指数）。  

### 评价

在上述问题中，虽然构建的协方差矩阵是正确的，但随后计算得出的特征向量存在问题，因为它们未能满足对应的齐次线性方程组。正确的特征向量需要通过准确求解每个特征值所对应的齐次系统来获得，并确保不同特征值的特征向量正交且归一化。因此，回答中的结论是不正确的。

$\boxed{\text{回答中的特征向量错误，需重新计算}}$