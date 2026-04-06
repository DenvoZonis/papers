# 五大类任务测试题库

## 一、论文大纲撰写

### 【学术论文类】

**题目1（研究型论文 - 困难）**
> 主题：气候变化背景下的城市韧性建设路径研究
> 要求：撰写跨学科研究论文大纲，需整合城市规划、公共管理、环境科学三个学科视角

### 【综述类】

**题目2（综述论文 - 中等）**
> 主题：数字经济时代下消费者行为研究进展
> 要求：设计系统性文献综述的大纲，需包含文献检索策略、分类框架、研究趋势分析

### 【应用型论文】

**题目3（案例分析 - 简单）**
> 主题：某新能源汽车企业的竞争战略分析
> 要求：设计案例分析论文大纲，需包含理论框架选择、案例描述、分析框架

### 【不同学科领域】

**题目4（工程类）**
> 主题：基于物联网的智能家居系统设计与实现
> 要求：撰写工程类学位论文大纲，需包含需求分析、系统设计、实现方案、测试验证

**题目5（人文社科类）**
> 主题：宋代市井文化对当代城市空间设计的启示
> 要求：撰写跨时空研究论文大纲，需体现历史学与建筑学的交叉


---

## 二、代码调试（13题）

### 【Python】

**题目1（语法错误 - 简单）**
```python
def calculate_average(numbers):
    sum = 0
    for i in range(len(numbers)):
        sum += numbers[i]
    average = sum / len(numbers)
    return average

result = calculate_average([10, 20, 30, 40, 50])
print("平均值是: " + result)
```
> 要求：找出所有错误并修复

**题目2（逻辑错误 - 中等）**
```python
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if lst[i] == lst[j] and lst[i] not in duplicates:
                duplicates.append(lst[i])
    return duplicates

print(find_duplicates([1, 2, 3, 2, 5, 3, 7, 3]))
```
> 要求：代码应返回 [2, 3]，但实际输出可能不正确，找出并修复bug

**题目3（边界条件 - 中等）**
```python
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```
> 要求：测试用例 [1,3,5,7,9] 查找 1、5、9、10，指出并修复问题

**题目4（算法效率 - 困难）**
```python
def longest_substring_without_repeating(s):
    max_length = 0
    for i in range(len(s)):
        seen = []
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.append(s[j])
            max_length = max(max_length, len(seen))
    return max_length
```
> 要求：分析时间复杂度问题，并给出O(n)解法

### 【Java】

**题目5（异常处理 - 中等）**
```java
public class UserService {
    public User getUserById(int id) {
        String query = "SELECT * FROM users WHERE id = " + id;
        ResultSet rs = database.execute(query);
        User user = new User();
        user.setId(rs.getInt("id"));
        user.setName(rs.getString("name"));
        return user;
    }
}
```
> 要求：指出代码中的问题并修复

**题目6（并发问题 - 困难）**
```java
public class Counter {
    private int count = 0;
    
    public void increment() {
        count++;
    }
    
    public int getCount() {
        return count;
    }
}

public class Main {
    public static void main(String[] args) throws InterruptedException {
        Counter counter = new Counter();
        Thread t1 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        });
        Thread t2 = new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                counter.increment();
            }
        });
        t1.start();
        t2.start();
        t1.join();
        t2.join();
        System.out.println(counter.getCount());
    }
}
```
> 要求：分析并修复并发问题

### 【JavaScript】

**题目7（异步编程 - 中等）**
```javascript
function fetchUserData(userId) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            if (userId > 0) {
                resolve({ id: userId, name: "User" + userId });
            } else {
                reject(new Error("Invalid user ID"));
            }
        }, 100);
    });
}

async function getUserAndPosts(userId) {
    const user = await fetchUserData(userId);
    const posts = await fetchPosts(user.id);
    return { user, posts };
}

getUserAndPosts(1).then(result => console.log(result));
```
> 要求：假设fetchPosts在用户数据返回前就被调用，找出并修复问题

**题目8（作用域 - 中等）**
```javascript
for (var i = 0; i < 3; i++) {
    setTimeout(() => console.log(i), 100);
}

for (let j = 0; j < 3; j++) {
    setTimeout(() => console.log(j), 100);
}
```
> 要求：分析两段代码的输出差异，并解释原因

**题目9（类型转换 - 简单）**
```javascript
console.log([] == false);
console.log(![]);
console.log([] + {});
console.log({} + []);
```
> 要求：预测每行代码的输出并解释JavaScript的类型转换规则

### 【C/C++】

**题目10（内存管理 - 困难）**
```c
#include <stdio.h>
#include <stdlib.h>

char* get greeting() {
    char greeting[] = "Hello, World!";
    return greeting;
}

int main() {
    char* msg = get_greeting();
    printf("%s\n", msg);
    return 0;
}
```
> 要求：识别内存相关问题并修复

**题目11（指针操作 - 中等）**
```c
#include <stdio.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(&x, &y);
    printf("x=%d, y=%d\n", x, y);
    
    int arr[] = {1, 2, 3, 4, 5};
    int* p = arr;
    printf("*(p+2): %d\n", *(p + 2));
    printf("p[2]: %d\n", p[2]);
    printf("*p + 2: %d\n", *p + 2);
    
    return 0;
}
```
> 要求：分析代码并解释输出

### 【综合调试题】

**题目12（多语言集成 - 困难）**
> 场景：一个Web应用使用Python后端和JavaScript前端，需要调试用户登录功能
> 要求：分析以下场景中的问题
> - 前端发送的密码包含特殊字符导致JSON解析失败
> - Python后端对空用户名的验证逻辑缺失
> - Session管理中的CSRF token验证漏洞

**题目13（性能优化 - 困难）**
```python
import time

def slow_function(data):
    result = []
    for item in data:
        if item not in result:
            result.append(item)
    return result

def another_function(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i * j
    return total

start = time.time()
data = list(range(10000))
slow_function(data)
another_function(1000)
print(f"耗时: {time.time() - start}")
```
> 要求：分析性能瓶颈并优化代码

---

## 三、数学解题（11题）

### 【代数】

**题目1（不等式 - 中等）**
> 解不等式：\(\frac{x-1}{x+2} \geq 2\)

**题目2（数列 - 困难）**
> 已知数列 \(\{a_n\}\) 满足 \(a_1 = 1\)，\(a_{n+1} = 2a_n + n\)，求 \(\{a_n\}\) 的通项公式

### 【微积分】

**题目3（求导 - 简单）**
> 求下列函数的导数：
> (1) \(f(x) = x^3 \ln x\)
> (2) \(f(x) = \frac{e^x}{x^2 + 1}\)

**题目4（积分 - 简单）**
> 计算 \(\int x^2 e^x \, dx\)

**题目5（定积分 - 中等）**
> 计算 \(\int_0^1 x^2 \sqrt{1-x^2} \, dx\)

**题目6（多元微积分 - 困难）**
> 设 \(f(x,y) = x^3 + y^3 - 3xy\)，求：
> (1) 所有驻点
> (2) 判定这些驻点是极大值、极小值还是鞍点

**题目7（微分方程 - 困难）**
> 解微分方程：\(y' + 2xy = e^{-x^2}\)，初始条件 \(y(0) = 1\)

### 【线性代数】

**题目8（矩阵运算 - 简单）**
> 设 \(A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}\)，求 \(A^{-1}\) 和 \(|A|\)

**题目9（特征值 - 中等）**
> 求矩阵 \(A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}\) 的特征值和特征向量

**题目10（线性方程组 - 中等）**
> 求解线性方程组：
> \(\begin{cases} x_1 + 2x_2 + 3x_3 = 1 \\ 2x_1 + 5x_2 + 3x_3 = 3 \\ x_1 + 8x_3 = 3 \end{cases}\)

**题目11（二次型 - 困难）**
> 将二次型 \(f(x_1, x_2, x_3) = x_1^2 + 2x_2^2 + 3x_3^2 + 4x_1x_2 - 2x_2x_3\)化为标准形

---

## 四、英文翻译（15题）

### 【中译英】

**题目1（科技类 - 简单）**
> 人工智能技术正在深刻改变我们的生活方式和工作模式。从智能家居到自动驾驶汽车，从医疗诊断到金融分析，AI的应用已经渗透到各个领域。

**题目2（商业类 - 中等）**
> 该公司的年度报告显示，尽管面临全球经济放缓的挑战，公司通过实施数字化转型战略，成功实现了营收增长15%和利润率提升2个百分点的业绩。管理层表示，这一成果得益于对创新研发的持续投入以及对客户体验的优化。

**题目3（文学类 - 困难）**
> "人生天地之间，若白驹之过隙，忽然而已。"这句话道出了时间的流逝之快，让人不由得感叹生命的短暂与珍贵。我们在有限的时间里，如何才能活出无限的价值，这是每一个时代的人都在思考的问题。

**题目4（法律类 - 中等）**
> 本协议自双方签字之日起生效，有效期为五年。协议期满后，如双方无异议，可自动续期一年。除非经双方书面协商一致，任何一方不得单方面解除或终止本协议。

**题目5（学术类 - 困难）**
> 本研究采用混合研究方法，结合定量分析与定性访谈，深入探讨了数字经济背景下中小企业组织变革的内在机制。研究发现，外部环境压力与内部资源禀赋的交互作用共同驱动了企业的战略调整，而领导力在其中扮演着关键的调节角色。

### 【英译中】

**题目6（新闻类 - 简单）**
> "The global economy is facing unprecedented challenges, but emerging markets continue to show resilience," said the IMF Managing Director in her opening remarks at the annual meeting.

**题目7（商务类 - 中等）**
> Our company has been committed to sustainable development for over a decade. We believe that long-term business success cannot be achieved at the expense of environmental protection or social responsibility. Therefore, we have integrated ESG principles into our core business strategy.

**题目8（文学类 - 困难）**
> "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness." — Charles Dickens, A Tale of Two Cities

**题目9（技术类 - 中等）**
> Machine learning algorithms require large amounts of labeled data to train effectively. However, data annotation is often time-consuming and expensive. Transfer learning offers a promising solution by allowing models trained on one task to be adapted for another related task with minimal additional training data.

**题目10（学术类 - 困难）**
> The relationship between socioeconomic status and educational attainment has been extensively documented in the literature. However, less attention has been paid to the mediating mechanisms through which family background influences children's academic achievement. This study aims to fill this gap by examining the role of parental involvement and learning resources as potential mediators.

### 【领域专项翻译】

**题目11（医学类 - 中等）**
> The patient presented with acute onset of chest pain radiating to the left arm, accompanied by diaphoresis and shortness of breath. Electrocardiogram revealed ST-segment elevation in leads V1-V4, consistent with anterior wall myocardial infarction. Troponin levels were significantly elevated at 2.5 ng/mL (normal < 0.04 ng/mL).

**题目12（法律合同 - 中等）**
> Each party represents and warrants that it has the full right, power, and authority to enter into this Agreement and to perform the acts required of it hereunder. Each party's representations and warranties shall survive the termination or expiration of this Agreement for a period of two (2) years.

**题目13（金融类 - 困难）**
> The fund employs a multi-factor investment approach, combining quantitative screening with fundamental analysis. Risk management is achieved through diversification across asset classes, geographic regions, and sectors. The portfolio is rebalanced quarterly to maintain target allocations and minimize tracking error.

**题目14（文化类 - 中等）**
> Chinese calligraphy, known as "shūfǎ" in Mandarin, is an art form that has been cultivated for over two thousand years. It is not merely a means of communication but also a reflection of the calligrapher's personality, emotions, and spiritual state. The practice requires patience, discipline, and a deep understanding of Chinese aesthetics.

**题目15（广告文案 - 简单）**
> "Innovation that empowers. Technology that inspires. Together, we are building the future." — [Brand tagline]

---

## 五、通识知识问答（27题）

### 【自然科学】

**题目1（物理 - 简单）**
> 为什么天空是蓝色的？

**题目2（地理 - 中等）**
> 什么是温室效应？它与全球气候变化有什么关系？

**题目3（物理 - 中等）**
> 请解释牛顿三大运动定律，并用日常生活中的例子加以说明

**题目4（天文 - 困难）**
> 请比较黑洞与中子星的异同，并说明它们各自的形成过程

**题目5（化学 - 中等）**
> 什么是同位素？同位素在医学和考古学中有什么应用？

### 【社会科学】

**题目6（经济学 - 简单）**
> 什么是通货膨胀？请举出一个可能导致通货膨胀的原因

**题目7（社会学 - 中等）**
> 什么是"社会资本"？它在个人发展和社区建设中起什么作用？

**题目8（政治学 - 中等）**
> 请解释三权分立原则及其在美国政府体制中的体现

**题目9（心理学 - 困难）**
> 比较行为主义理论和认知心理学理论对"学习"的不同解释

**题目10（法学 - 中等）**
> 什么是"无罪推定"原则？它为什么被认为是现代法治的基石之一？

**题目11（经济学 - 困难）**
> 解释"菲利普斯曲线"的含义，它描述了哪两个经济变量之间的关系？

### 【人文历史】

**题目12（历史 - 简单）**
> 中国古代四大发明是什么？它们对世界文明有什么影响？

**题目13（文学 - 中等）**
> 请介绍《哈姆雷特》的主要情节和莎士比亚通过这部剧作探讨的主题

**题目14（艺术 - 简单）**
> 什么是"文艺复兴"？请列举文艺复兴时期的三位代表艺术家及其代表作

**题目15（哲学 - 困难）**
> 康德的"绝对命令"是什么意思？请评价这一伦理学说

**题目16（历史 - 中等）**
> 比较英国资产阶级革命和法国大革命的原因、过程和历史意义

**题目17（语言学 - 中等）**
> 什么是语言的"萨皮尔-沃尔夫假说"？它对跨文化交际有什么启示？

**题目18（文学 - 困难）**
> 鲁迅的《狂人日记》被视为中国现代文学的开山之作，请分析其艺术特色和思想意义

### 【生活常识】

**题目19（健康 - 简单）**
> 什么是BMI指数？如何计算？它有什么局限性？

**题目20（法律 - 中等）**
> 在中国，如果发生了交通事故，正确的处理流程是什么？

**题目21（金融 - 简单）**
> 什么是复利？请举例说明复利与单利的区别

**题目22（环保 - 中等）**
> 什么是垃圾分类？中国为什么要推行垃圾分类？

**题目23（急救 - 简单）**
> 有人突然心脏骤停，在等待救护车到来之前，普通人可以采取哪些急救措施？

### 【综合分析】

**题目24（跨学科 - 困难）**
> 人工智能技术的发展引发了关于"人工智能是否应该拥有权利"的讨论。请从技术、伦理、法律和社会角度分析这一问题

**题目25（批判性思维 - 困难）**
> "读书无用论"认为大学生毕业后找不到工作、读书浪费时间和金钱。请从经济学、社会学、个人发展等角度反驳这一观点

**题目26（时事分析 - 中等）**
> 请分析数字货币（如比特币、各国央行数字货币）对传统金融体系可能产生的影响

**题目27（逻辑推理 - 困难）**
> "所有的A都是B"和"有些A不是B"这两个命题之间的关系是什么？请用文氏图说明，并判断以下推理是否有效：
> 前提1：所有的鸟都会飞
> 前提2：企鹅是鸟
> 结论：企鹅会飞

---

## 评分参考建议

| 任务类型 | 评判维度 | 权重建议 |
|---------|---------|---------|
| 论文大纲 | 结构完整性、逻辑性、创新性、学术规范性 | 30%/25%/25%/20% |
| 代码调试 | 错误识别准确性、修复方案合理性、代码质量 | 40%/35%/25% |
| 数学解题 | 解题思路、计算准确性、答案完整度 | 35%/45%/20% |
| 英文翻译 | 准确性、流畅性、文化适应性、术语专业性 | 40%/30%/15%/15% |
| 通识问答 | 准确性、深度、广度、表达能力 | 35%/30%/20%/15% |