## 题目1

解不等式：$\frac{x-1}{x+2} \geq 2$

### 解答

将不等式移项：

$$\frac{x-1}{x+2} - 2 \geq 0$$

$$\frac{x-1 - 2(x+2)}{x+2} \geq 0$$

$$\frac{x-1-2x-4}{x+2} \geq 0$$

$$\frac{-x-5}{x+2} \geq 0$$

即：

$$\frac{x+5}{x+2} \leq 0$$

令分子、分母分别为零：$x = -5$，$x = -2$

分析符号：

| 区间 | $x+5$ | $x+2$ | 商 |
|------|-------|-------|---|
| $x < -5$ | $-$ | $-$ | $+$ |
| $-5 < x < -2$ | $+$ | $-$ | $-$ |
| $x > -2$ | $+$ | $+$ | $+$ |

需要商 $\leq 0$，且 $x \neq -2$（分母不为零），$x=-5$ 时分子为零满足 $\leq 0$。

$$\boxed{-5 \leq x < -2}$$

### 评价


解答正确。步骤清晰且逻辑严谨，特别是在处理分数不等式和符号变化时表现良好。答案范围正确：$\boxed{[-5, -2)}$。

## 题目2

已知数列 $\{a_n\}$ 满足 $a_1 = 1$，$a_{n+1} = 2a_n + n$，求 $\{a_n\}$ 的通项公式

### 解答

**第一步：求齐次通解**

齐次递推 $a_{n+1} = 2a_n$ 的通解为 $C \cdot 2^n$。

**第二步：求特解**

设特解为 $a_n^* = \alpha n + \beta$，代入递推式：

$$\alpha(n+1) + \beta = 2(\alpha n + \beta) + n$$

$$\alpha n + \alpha + \beta = (2\alpha + 1)n + 2\beta$$

比较系数：
- $n^1$：$\alpha = 2\alpha + 1 \Rightarrow \alpha = -1$
- 常数项：$\alpha + \beta = 2\beta \Rightarrow \beta = \alpha = -1$

特解为 $a_n^* = -n - 1$。

**第三步：通解**

$$a_n = C \cdot 2^n - n - 1$$

**第四步：代入初始条件** $a_1 = 1$：

$$1 = C \cdot 2^1 - 1 - 1 = 2C - 2 \Rightarrow C = \frac{3}{2}$$

$$\boxed{a_n = 3 \cdot 2^{n-1} - n - 1}$$

**验证**：$a_1 = 3 \cdot 1 - 2 = 1$ ✓；$a_2 = 2a_1 + 1 = 3$，公式：$3\cdot2 - 3 = 3$ ✓

---

### 评价

答案完整且正确。

## 题目3

求下列函数的导数：
(1) $f(x) = x^3 \ln x$
(2) $f(x) = \frac{e^x}{x^2+1}$

### 解答

**(1)** 使用乘积法则 $(uv)' = u'v + uv'$，令 $u = x^3,\ v = \ln x$：

$$f'(x) = 3x^2 \cdot \ln x + x^3 \cdot \frac{1}{x} = 3x^2 \ln x + x^2$$

$$\boxed{f'(x) = x^2(3\ln x + 1)}$$

**(2)** 使用商法则 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$，令 $u = e^x,\ v = x^2+1$：

$$f'(x) = \frac{e^x(x^2+1) - e^x \cdot 2x}{(x^2+1)^2} = \frac{e^x(x^2 - 2x + 1)}{(x^2+1)^2}$$

$$\boxed{f'(x) = \frac{e^x(x-1)^2}{(x^2+1)^2}}$$

---

### 评价

(1) 正确。使用了乘积法则，计算和化简步骤正确，答案正确。

(2) 正确。使用了商法则，分子化简过程中 $x^2 - 2x + 1$ 化为 $(x-1)^2$ 是正确的，答案正确。

## 题目4

计算 $\int x^2 e^x \, dx$

### 解答

连续使用分部积分法 $\int u\,dv = uv - \int v\,du$：

**第一次**：令 $u = x^2,\ dv = e^x dx$，则 $du = 2x\,dx,\ v = e^x$：

$$\int x^2 e^x\,dx = x^2 e^x - 2\int x e^x\,dx$$

**第二次**：令 $u = x,\ dv = e^x dx$，则 $du = dx,\ v = e^x$：

$$\int x e^x\,dx = xe^x - \int e^x\,dx = xe^x - e^x$$

**合并**：

$$\int x^2 e^x\,dx = x^2 e^x - 2(xe^x - e^x) + C$$

$$\boxed{\int x^2 e^x\,dx = e^x(x^2 - 2x + 2) + C}$$

### 评价

解答正确且详细。答案两次使用分部积分法，步骤清晰，结果无误。

## 题目5

计算 $\int_0^1 x^2\sqrt{1-x^2}\,dx$

### 解答

令 $x = \sin\theta$，则 $dx = \cos\theta\,d\theta$，$\sqrt{1-x^2} = \cos\theta$。

换限：$x=0 \Rightarrow \theta=0$；$x=1 \Rightarrow \theta=\pi/2$。

$$I = \int_0^{\pi/2} \sin^2\theta \cdot \cos\theta \cdot \cos\theta\,d\theta = \int_0^{\pi/2} \sin^2\theta\cos^2\theta\,d\theta$$

利用二倍角公式 $\sin^2\theta\cos^2\theta = \frac{1}{4}\sin^2 2\theta = \frac{1}{4} \cdot \frac{1-\cos 4\theta}{2} = \frac{1-\cos 4\theta}{8}$：

$$I = \int_0^{\pi/2} \frac{1-\cos 4\theta}{8}\,d\theta = \frac{1}{8}\left[\theta - \frac{\sin 4\theta}{4}\right]_0^{\pi/2}$$

$$= \frac{1}{8}\left(\frac{\pi}{2} - \frac{\sin 2\pi}{4} - 0\right) = \frac{1}{8} \cdot \frac{\pi}{2}$$

$$\boxed{I = \frac{\pi}{16}}$$

### 评价

步骤正确。被积函数通过三角替换和二倍角公式成功简化，计算过程无误，结果准确。

评价：好。解答详细且正确。

## 题目6

设 $f(x,y) = x^3 + y^3 - 3xy$，求：(1) 所有驻点；(2) 判定极值类型

### 解答

**(1) 求驻点**

令偏导数为零：

$$f_x = 3x^2 - 3y = 0 \Rightarrow y = x^2$$

$$f_y = 3y^2 - 3x = 0 \Rightarrow x = y^2$$

将 $y = x^2$ 代入 $x = y^2 = x^4$，得 $x^4 - x = 0$，即 $x(x^3-1)=0$。

故 $x = 0$ 或 $x = 1$，对应 $y = 0$ 或 $y = 1$。

**驻点为 $(0,0)$ 和 $(1,1)$。**

**(2) 判定极值类型**

计算二阶偏导：$f_{xx} = 6x,\quad f_{xy} = -3,\quad f_{yy} = 6y$

Hessian 判别式：$H = f_{xx}f_{yy} - f_{xy}^2 = 36xy - 9$

**在 $(0,0)$：**

$$H = 36 \cdot 0 \cdot 0 - 9 = -9 < 0$$

$\Rightarrow$ **(0,0) 是鞍点**。

**在 $(1,1)$：**

$$H = 36 \cdot 1 \cdot 1 - 9 = 27 > 0，\quad f_{xx}(1,1) = 6 > 0$$

$\Rightarrow$ **(1,1) 是极小值点**，极小值 $f(1,1) = 1+1-3 = -1$。

### 评价

答案正确。所有步骤和计算都是合理的，没有发现错误。驻点求解和二阶导数的使用都符合数学分析的方法。

**评价：**
解答过程正确，结果无误。

详细回答：

**(1) 求驻点**

令偏导数为零：

$$f_x = 3x^2 - 3y = 0 \Rightarrow y = x^2$$

$$f_y = 3y^2 - 3x = 0 \Rightarrow x = y^2$$

将 $y = x^2$ 代入 $x = y^2 = (x^2)^2 = x^4$，得到方程：

$$x^4 - x = 0 \Rightarrow x(x^3 - 1) = 0$$

解得 $x = 0$ 或 $x = 1$。对应的 $y$ 值为 $y = 0$ 和 $y = 1$。

**驻点为 $(0,0)$ 和 $(1,1)$。**

**(2) 判定极值类型**

计算二阶偏导数：

$$f_{xx} = 6x,\quad f_{xy} = -3,\quad f_{yy} = 6y$$

Hessian 判别式为：

$$H = f_{xx}f_{yy} - f_{xy}^2 = (6x)(6y) - (-3)^2 = 36xy - 9$$

- **在 $(0,0)$：**

  $$H = 36 \cdot 0 \cdot 0 - 9 = -9 < 0$$

  因此，$(0,0)$ 是鞍点。

- **在 $(1,1)$：**

  $$H = 36 \cdot 1 \cdot 1 - 9 = 27 > 0$$
  
  且 $f_{xx}(1,1) = 6 > 0$，因此，$(1,1)$ 是极小值点。

计算函数在 $(1,1)$ 处的值：

$$f(1,1) = 1^3 + 1^3 - 3 \cdot 1 \cdot 1 = 1 + 1 - 3 = -1$$

**最终答案：**

驻点为 $(0,0)$ 和 $(1,1)$。其中，$(0,0)$ 是鞍点，$(1,1)$ 是极小值点，且 $f(1,1) = -1$。

**(1) 驻点:**  
$\boxed{(0,0)}$ 和 $\boxed{(1,1)}$

**(2) 极值类型:**  
- $(0,0)$ 是鞍点。  
- $(1,1)$ 是极小值点，极小值为 $\boxed{-1}$。

## 题目7

解微分方程：$y' + 2xy = e^{-x^2}$，初始条件 $y(0)=1$

### 解答

这是一阶线性微分方程，标准形式 $y' + P(x)y = Q(x)$，其中 $P(x)=2x$，$Q(x)=e^{-x^2}$。

**积分因子：**

$$\mu = e^{\int 2x\,dx} = e^{x^2}$$

两边乘以 $e^{x^2}$：

$$e^{x^2}y' + 2xe^{x^2}y = e^{x^2} \cdot e^{-x^2} = 1$$

即：

$$\frac{d}{dx}\left(e^{x^2}y\right) = 1$$

两边积分：

$$e^{x^2}y = x + C$$

$$y = (x+C)e^{-x^2}$$

**代入初始条件** $y(0)=1$：

$$1 = (0+C)e^0 = C \Rightarrow C = 1$$

$$\boxed{y = (x+1)e^{-x^2}}$$

### 评价

这是一阶线性微分方程，标准形式为 $y' + P(x)y = Q(x)$，其中 $P(x) = 2x$ 和 $Q(x) = e^{-x^2}$。通过计算积分因子和求解，得到的通解满足初始条件，并且推导过程中每一步都正确无误。

答案正确。

**评价：**

该解答过程完整且准确。步骤清晰，应用了一阶线性微分方程的标准解法，包括正确计算积分因子并代入初始条件求出常数。最终答案正确。

## 题目8

设 $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$，求 $A^{-1}$ 和 $|A|$

### 解答

**行列式：**

$$|A| = 1\cdot4 - 2\cdot3 = 4-6 = \boxed{-2}$$

**逆矩阵**（$2\times2$ 矩阵公式：交换主对角元素，对角取反，除以行列式）：

$$A^{-1} = \frac{1}{|A|}\begin{pmatrix}4&-2\\-3&1\end{pmatrix} = \frac{1}{-2}\begin{pmatrix}4&-2\\-3&1\end{pmatrix}$$

$$\boxed{A^{-1} = \begin{pmatrix}-2&1\\\frac{3}{2}&-\frac{1}{2}\end{pmatrix}}$$

**验证**：$AA^{-1} = \begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}-2&1\\\frac{3}{2}&-\frac{1}{2}\end{pmatrix} = \begin{pmatrix}-2+3&1-1\\-6+6&3-2\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix}$ ✓

### 评价

答案正确。行列式和逆矩阵均按公式准确计算，且通过验证确认无误。

## 题目9

求矩阵 $A = \begin{pmatrix}4&1\\2&3\end{pmatrix}$ 的特征值和特征向量

### 解答

**特征值：** 解 $\det(A - \lambda I) = 0$：

$$\begin{vmatrix}4-\lambda&1\\2&3-\lambda\end{vmatrix} = (4-\lambda)(3-\lambda)-2 = \lambda^2-7\lambda+10 = 0$$

$$(\lambda-5)(\lambda-2)=0 \Rightarrow \boxed{\lambda_1=5,\quad \lambda_2=2}$$

**对 $\lambda_1 = 5$：** 解 $(A-5I)\mathbf{x}=\mathbf{0}$：

$$\begin{pmatrix}-1&1\\2&-2\end{pmatrix}\rightarrow \begin{pmatrix}1&-1\\0&0\end{pmatrix}$$

$x_1 = x_2$，取 $x_2=1$，特征向量为 $\boxed{\xi_1 = \begin{pmatrix}1\\1\end{pmatrix}}$

**对 $\lambda_2 = 2$：** 解 $(A-2I)\mathbf{x}=\mathbf{0}$：

$$\begin{pmatrix}2&1\\2&1\end{pmatrix}\rightarrow \begin{pmatrix}2&1\\0&0\end{pmatrix}$$

$2x_1 = -x_2$，取 $x_2=2$，特征向量为 $\boxed{\xi_2 = \begin{pmatrix}-1\\2\end{pmatrix}}$

### 评价

**题目：**  
求矩阵 \( A = \begin{pmatrix}4 & 1 \\ 2 & 3\end{pmatrix} \) 的特征值和特征向量。  
**答案：**  
- 特征值：5 和 2，正确。
- 对应的特征向量计算正确，分别为 \(\begin{pmatrix}1\\1\end{pmatrix}\) 和 \(\begin{pmatrix}-1\\2\end{pmatrix}\)，但后者的表示可以不同。  

**评价：**  
解题过程和答案均正确。

## 题目10

求解线性方程组：$\begin{cases}x_1+2x_2+3x_3=1\\2x_1+5x_2+3x_3=3\\x_1+8x_3=3\end{cases}$

### 解答

写出增广矩阵并进行行变换：

$$\left(\begin{array}{ccc|c}1&2&3&1\\2&5&3&3\\1&0&8&3\end{array}\right)$$

$R_2 \leftarrow R_2 - 2R_1$，$R_3 \leftarrow R_3 - R_1$：

$$\left(\begin{array}{ccc|c}1&2&3&1\\0&1&-3&1\\0&-2&5&2\end{array}\right)$$

$R_3 \leftarrow R_3 + 2R_2$：

$$\left(\begin{array}{ccc|c}1&2&3&1\\0&1&-3&1\\0&0&-1&4\end{array}\right)$$

由第三行：$-x_3 = 4 \Rightarrow x_3 = -4$

由第二行：$x_2 - 3(-4) = 1 \Rightarrow x_2 = 1-12 = -11$

由第一行：$x_1 + 2(-11) + 3(-4) = 1 \Rightarrow x_1 = 1+22+12 = 35$

$$\boxed{x_1=35,\quad x_2=-11,\quad x_3=-4}$$

### 评价

解答过程正确且清晰，使用了高斯消元法通过行变换将增广矩阵化为阶梯形，并回代求得变量值。答案正确验证了原方程组的所有方程，结果无误。

$$
\boxed{x_1=35,\quad x_2=-11,\quad x_3=-4}
$$

## 题目11

将二次型 $f(x_1,x_2,x_3) = x_1^2+2x_2^2+3x_3^2+4x_1x_2-2x_2x_3$ 化为标准形

### 解答

**写出矩阵：**

$$A = \begin{pmatrix}1&2&0\\2&2&-1\\0&-1&3\end{pmatrix}$$

**配方法：**

**第一步**，对含 $x_1$ 的项配方：

$$f = (x_1^2 + 4x_1x_2) + 2x_2^2 + 3x_3^2 - 2x_2x_3$$

$$= (x_1+2x_2)^2 - 4x_2^2 + 2x_2^2 + 3x_3^2 - 2x_2x_3$$

$$= (x_1+2x_2)^2 - 2x_2^2 + 3x_3^2 - 2x_2x_3$$

令 $y_1 = x_1+2x_2$，处理余下部分 $g = -2x_2^2 - 2x_2x_3 + 3x_3^2$。

**第二步**，对含 $x_2$ 的项配方：

$$g = -2\left(x_2^2 + x_2x_3\right) + 3x_3^2$$

$$= -2\left(x_2+\frac{1}{2}x_3\right)^2 + \frac{1}{2}x_3^2 + 3x_3^2$$

$$= -2\left(x_2+\frac{1}{2}x_3\right)^2 + \frac{7}{2}x_3^2$$

令 $y_2 = x_2+\dfrac{1}{2}x_3$，$y_3 = x_3$。

**标准形：**

$$\boxed{f = y_1^2 - 2y_2^2 + \frac{7}{2}y_3^2}$$

对应的变换为：

$$\begin{cases}y_1 = x_1+2x_2\\y_2 = x_2+\dfrac{1}{2}x_3\\y_3 = x_3\end{cases}$$

二次型的**惯性指数**：正惯性指数 $p=2$，负惯性指数 $q=1$，符合实二次型的惯性定理。

### 评价

答案正确。矩阵A、配方法步骤以及标准形和惯性指数的计算均无误。