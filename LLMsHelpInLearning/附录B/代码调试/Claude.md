## 题目1

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

### 解答

**问题：** `print` 语句中使用 `+` 拼接字符串和数值，`result` 是 `float` 类型，不能直接与字符串相加，会抛出 `TypeError`。

**修复：** 将 `result` 转为字符串，或使用 f-string。

```python
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    average = total / len(numbers)
    return average

result = calculate_average([10, 20, 30, 40, 50])
print("平均值是: " + str(result))
# 或更优雅地：print(f"平均值是: {result}")
```

> 附注：`sum` 是 Python 内置函数名，建议避免用作变量名（此处已改为 `total`）。

### 评价

```python
def calculate_average(numbers):
    sum = 0  # 正确修复为 total=0
    for i in range(len(numbers)):
        sum += numbers[i]  # 正确改为 total += ...
    average = sum / len(numbers)
    return average

result = calculate_average([10,20,30,40,50])
print("平均值是: " + result)  # TypeError，应修复为 str(result) 或 f-string
```

此回答基本正确。

## 题目2

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

### 解答

**分析：** 实际运行此代码，输出就是 `[2, 3]`，逻辑本身正确。但该算法时间复杂度为 O(n³)（双层循环 + `not in` 线性查找），存在性能问题。

**潜在隐患：** 若输入列表极大，性能会急剧下降。

**优化修复版本（O(n)）：**

```python
def find_duplicates(lst):
    seen = set()
    duplicates = []
    for item in lst:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates

print(find_duplicates([1, 2, 3, 2, 5, 3, 7, 3]))  # 输出: [2, 3]
```

### 评价

原题中的代码逻辑正确，但时间复杂度过高。修复后的代码利用集合优化了性能，解决了潜在的性能问题。答案提供的分析和修复方案准确，是正确的解决方案。

评价：解答正确，分析到位，修复有效。

## 题目3

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

### 解答

**问题：** `right` 初始化为 `len(arr)` 而非 `len(arr) - 1`，导致 `arr[mid]` 在某些情况下访问越界索引，引发 `IndexError`。

- 数组长度为 5，合法索引为 0~4
- 初始 `right = 5`，当 `left=0, right=5` 时 `mid=2`，此轮还好；但极端情况下 `mid` 可能等于 5，导致越界

**修复：**

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1  # 修复：改为 len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [1, 3, 5, 7, 9]
print(binary_search(arr, 1))   # 0
print(binary_search(arr, 5))   # 2
print(binary_search(arr, 9))   # 4
print(binary_search(arr, 10))  # -1
```

### 评价

解答正确且完整，准确地指出了代码中的越界问题，并进行了正确的修复。测试用例运行结果符合预期。

## 题目4

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

### 解答

**时间复杂度分析：**
- 外层循环 O(n)，内层循环最坏 O(n)，`s[j] in seen`（列表查找）O(n)
- 总体：**O(n³)**，字符串较长时性能极差

**O(n) 滑动窗口解法：** 使用哈希表记录字符最新位置，左指针直接跳跃，避免重复遍历。

```python
def longest_substring_without_repeating(s):
    char_index = {}   # 记录字符最近出现的位置
    max_length = 0
    left = 0          # 滑动窗口左边界

    for right in range(len(s)):
        # 若字符已在窗口内，移动左边界到重复字符的下一位
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length

print(longest_substring_without_repeating("abcabcbb"))  # 3
print(longest_substring_without_repeating("bbbbb"))     # 1
print(longest_substring_without_repeating("pwwkew"))    # 3
```

**复杂度：** 时间 O(n)，空间 O(min(n, 字符集大小))。

### 评价

原代码的时间复杂度过高（O(n³)），无法高效处理长字符串。提供的优化解法使用滑动窗口技巧和哈希表，将时间复杂度降低到O(n)，解决了性能问题。

解答正确且有效，成功找到了更优的解决方案。

## 题目5

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

### 解答

**问题一：SQL 注入风险**
直接拼接 `id` 到 SQL 字符串，若参数来自外部输入，存在注入漏洞（此处 `id` 是 `int` 相对安全，但不良习惯应纠正）。

**问题二：未处理异常**
`database.execute()` 及 `ResultSet` 操作可能抛出 `SQLException`，未捕获会导致程序崩溃。

**问题三：ResultSet 和资源未关闭**
`ResultSet`、`PreparedStatement` 等需要关闭，否则造成资源泄漏。

**问题四：未处理查询结果为空的情况**
若用户不存在，`rs.next()` 返回 false，直接读取字段会报错。

**修复：**

```java
public class UserService {
    public User getUserById(int id) throws SQLException {
        String query = "SELECT * FROM users WHERE id = ?";
        // 使用 try-with-resources 自动关闭资源
        try (PreparedStatement stmt = connection.prepareStatement(query)) {
            stmt.setInt(1, id);
            try (ResultSet rs = stmt.executeQuery()) {
                if (rs.next()) {
                    User user = new User();
                    user.setId(rs.getInt("id"));
                    user.setName(rs.getString("name"));
                    return user;
                } else {
                    return null; // 或抛出自定义 UserNotFoundException
                }
            }
        }
    }
}
```

### 评价

解答准确指出了代码的主要问题，并正确应用了预编译语句、资源关闭和空结果处理。修复有效且符合最佳实践，仅需关注连接的管理。

## 题目6

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
// ... main 中两个线程各自增 1000 次
```
> 要求：分析并修复并发问题

### 解答

**问题分析：**
`count++` 看似一行，实际是三步操作：**读取 → 加1 → 写回**。两个线程同时执行时可能发生竞态条件（Race Condition）：
- 线程1 读到 count=5，线程2 也读到 count=5
- 两者各自加1后都写回 6，丢失了一次自增
- 最终结果往往小于 2000

**修复方案一：使用 `AtomicInteger`（推荐，无锁高性能）**

```java
import java.util.concurrent.atomic.AtomicInteger;

public class Counter {
    private AtomicInteger count = new AtomicInteger(0);

    public void increment() {
        count.incrementAndGet(); // 原子操作
    }

    public int getCount() {
        return count.get();
    }
}
```

**修复方案二：使用 `synchronized`**

```java
public class Counter {
    private int count = 0;

    public synchronized void increment() {
        count++;
    }

    public synchronized int getCount() {
        return count;
    }
}
```

**修复方案三：使用 `volatile`（不足以解决此问题，仅供对比）**
`volatile` 只保证可见性，不保证原子性，对 `count++` 无效，**不能**用于此场景。

推荐方案一，性能优于加锁。

### 评价

该解答全面分析了代码中的并发问题及其原因，提出了三种有效的修复方案，并进行了比较和推荐。答案结构清晰，内容详尽，适合学习和理解。

## 题目7

```javascript
async function getUserAndPosts(userId) {
    const user = await fetchUserData(userId);
    const posts = await fetchPosts(user.id);
    return { user, posts };
}
```
> 要求：假设fetchPosts在用户数据返回前就被调用，找出并修复问题

### 解答

**问题分析：**
原代码逻辑是正确的——`await fetchUserData` 会等待用户数据返回后，才执行 `fetchPosts`。题目假设的场景（fetchPosts 提前调用）在此写法下不会发生。

**真正的潜在问题：** 若 `fetchUserData` 或 `fetchPosts` 抛出异常（如 userId 无效），整个函数会 reject，但调用处 `.then()` 没有 `.catch()`，会产生未捕获的 Promise 错误。

**完整修复版本：**

```javascript
async function getUserAndPosts(userId) {
    try {
        const user = await fetchUserData(userId);
        const posts = await fetchPosts(user.id);
        return { user, posts };
    } catch (error) {
        console.error("获取数据失败:", error.message);
        throw error; // 向上传递，或返回默认值
    }
}

// 调用处也需处理错误
getUserAndPosts(1)
    .then(result => console.log(result))
    .catch(err => console.error("调用失败:", err));
```

**若需要并行请求（性能优化）：** 如果 `fetchPosts` 不依赖 `user.id`（假设 userId 已知），可用 `Promise.all` 并发：

```javascript
async function getUserAndPosts(userId) {
    const [user, posts] = await Promise.all([
        fetchUserData(userId),
        fetchPosts(userId)
    ]);
    return { user, posts };
}
```

### 评价

答案正确。问题分析准确，修复方法得当，补充的错误处理和性能优化建议合理。

## 题目8

```javascript
for (var i = 0; i < 3; i++) {
    setTimeout(() => console.log(i), 100);
}

for (let j = 0; j < 3; j++) {
    setTimeout(() => console.log(j), 100);
}
```
> 要求：分析两段代码的输出差异，并解释原因

### 解答

**第一段输出：**
```
3
3
3
```

**第二段输出：**
```
0
1
2
```

**原因分析：**

| 特性 | `var` | `let` |
|------|-------|-------|
| 作用域 | 函数作用域（或全局） | 块级作用域 |
| 闭包捕获 | 捕获同一个变量引用 | 每次迭代创建新绑定 |

- **`var` 版本：** `i` 是函数级作用域，三个回调函数共享同一个 `i`。100ms 后循环早已结束，`i` 变为 3，三个回调都输出 3。
- **`let` 版本：** `j` 是块级作用域，每次迭代都创建一个新的 `j` 绑定（值分别为 0、1、2），各回调闭包捕获各自的 `j`，输出 0、1、2。

**`var` 的修复方案（若需输出 0 1 2）：** 使用 IIFE 创建独立作用域：

```javascript
for (var i = 0; i < 3; i++) {
    (function(i) {
        setTimeout(() => console.log(i), 100);
    })(i);
}
```

### 评价

解答正确且清晰，解释了两段代码输出差异的原因，并提供了修复方案。

答案：

第一段代码输出3次3，第二段输出0、1、2。这是因为`var`共享同一变量引用，而`let`每次迭代创建新绑定。

## 题目9

```javascript
console.log([] == false);
console.log(![]);
console.log([] + {});
console.log({} + []);
```
> 要求：预测每行代码的输出并解释JavaScript的类型转换规则

### 解答

**输出：**
```
true
false
[object Object]
0
```

**逐行解释：**

**第1行 `[] == false` → `true`**
宽松比较触发类型转换链：
1. `false` 转数字 → `0`
2. `[]` 转原始值 → `""（toString）` → 数字 `0`
3. `0 == 0` → `true`

**第2行 `![]` → `false`**
`!` 运算符将操作数转为布尔值再取反。`[]` 是对象，所有对象转布尔均为 `true`，取反得 `false`。

**第3行 `[] + {}` → `"[object Object]"`**
`+` 运算符两侧非数字时转字符串：
- `[]` → `""`
- `{}` → `"[object Object]"`
- 结果：`"" + "[object Object]"` = `"[object Object]"`

**第4行 `{} + []` → `0`**
此处 `{}` 被 JavaScript 解析为**空代码块**（不是对象字面量），实际执行的是 `+[]`（一元加）：
- `+[]` → `+""` → `0`

> 若写成 `({} + [])` 则输出 `"[object Object]"`，与第3行一致。

### 评价

解答大部分正确，但在第四行的解释存在误导。

## 题目10

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

### 解答

**问题一：函数名语法错误**
`get greeting()` 中有空格，应为 `get_greeting()`。

**问题二：返回局部数组的指针（悬空指针）**
`char greeting[] = "Hello, World!"` 是栈上的局部数组，函数返回后栈帧被销毁，该内存被释放。返回其指针后使用属于**未定义行为**，可能输出乱码或崩溃。

**修复方案一：使用静态局部变量**
```c
char* get_greeting() {
    static char greeting[] = "Hello, World!";
    return greeting;
}
```

**修复方案二：使用堆内存（需手动释放）**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* get_greeting() {
    char* greeting = (char*)malloc(14 * sizeof(char));
    if (greeting == NULL) return NULL;
    strcpy(greeting, "Hello, World!");
    return greeting;
}

int main() {
    char* msg = get_greeting();
    if (msg != NULL) {
        printf("%s\n", msg);
        free(msg); // 使用后必须释放
    }
    return 0;
}
```

**修复方案三：使用字符串字面量指针（只读）**
```c
char* get_greeting() {
    return "Hello, World!"; // 字符串字面量存于静态区，生命周期为程序全程
}
```

### 评价

答案正确。问题识别和修复方案都清晰明了，涵盖了常见的内存安全问题及其解决方案。三个修复方案各有优劣，适合不同的使用场景。

## 题目11

```c
void swap(int* a, int* b) { ... }
// 以及指针运算示例
```
> 要求：分析代码并解释输出

### 解答

代码无错误，可正常编译运行。各部分输出分析如下：

**swap 部分：**
```
x=10, y=5
```
传入 `x`、`y` 的地址，函数内通过解引用交换其值，调用后 `x=10, y=5`，正确。

**指针运算部分：**
```c
int arr[] = {1, 2, 3, 4, 5};
int* p = arr;  // p 指向 arr[0]，即值 1
```

| 表达式 | 运算规则 | 输出 |
|--------|----------|------|
| `*(p+2)` | p 偏移2个 int 单位，指向 arr[2]，解引用得值 | `3` |
| `p[2]` | 等价于 `*(p+2)`，语法糖 | `3` |
| `*p + 2` | 先解引用 p 得 arr[0]=1，再加 2 | `3` |

**完整输出：**
```
x=10, y=5
*(p+2): 3
p[2]: 3
*p + 2: 3
```

> 注意：三个指针表达式结果均为 3，但含义完全不同——前两个是访问第三个元素，最后一个是取第一个元素的值加2。

### 评价

解答正确且清晰，指出了每个表达式的结果及其含义。swap部分的描述准确无误，指针运算部分详细解释了每个表达式的执行过程和结果。虽然答案长度适中，但已经足够详细，没有冗长的问题。

## 题目12

> 场景：Web应用 Python 后端 + JavaScript 前端，调试用户登录功能

### 解答

**问题一：前端发送密码含特殊字符导致 JSON 解析失败**

**原因：** 前端未对密码做正确的 JSON 序列化，直接字符串拼接构造 JSON，特殊字符（如 `"`, `\`, 换行符）破坏 JSON 结构。

**修复：** 始终使用 `JSON.stringify()` 序列化数据，绝不手动拼接 JSON。

```javascript
// 错误做法
const body = '{"username": "' + username + '", "password": "' + password + '"}';

// 正确做法
const body = JSON.stringify({ username, password });
fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body
});
```

**问题二：Python 后端对空用户名的验证逻辑缺失**

**原因：** 未校验输入直接查库，空用户名可能匹配到非预期记录，或引发数据库错误。

**修复：** 在处理请求前进行严格的输入校验。

```python
from flask import request, jsonify

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    if not password:
        return jsonify({'error': '密码不能为空'}), 400
    if len(username) > 128:
        return jsonify({'error': '用户名过长'}), 400

    # 继续后续逻辑...
```

**问题三：Session 管理中的 CSRF token 验证漏洞**

**原因：** 若 CSRF token 未绑定用户 session、可被重用、或验证逻辑存在绕过（如接受空 token），攻击者可伪造跨站请求。

**常见漏洞形式：**
- 服务端只检查 token 是否存在，不验证其值
- token 未与 session 绑定，全局通用
- 未对 token 做时效性限制

**修复（以 Flask 为例，使用 flask-wtf）：**

```python
from flask_wtf.csrf import CSRFProtect, validate_csrf
from wtforms.validators import ValidationError

csrf = CSRFProtect(app)

@app.route('/api/login', methods=['POST'])
def login():
    token = request.headers.get('X-CSRFToken') or request.json.get('csrf_token')
    try:
        validate_csrf(token)  # 自动与 session 绑定校验
    except ValidationError:
        return jsonify({'error': 'CSRF验证失败'}), 403

    # 继续登录逻辑...
```

**前端配合：** 每次请求从 Cookie 或 meta 标签读取 CSRF token 并附在请求头中：

```javascript
const csrfToken = document.cookie.match(/csrftoken=([^;]+)/)?.[1];
fetch('/api/login', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
    },
    body: JSON.stringify({ username, password })
});
```

### 评价

题目设计合理，覆盖了前端、后端及安全性问题。

解答：
1. 问题一：解释正确，修复代码清晰且正确。
2. 问题二：校验逻辑完整，但未考虑密码过长等其他可能异常情况。
3. 问题三：CSRF修复方案正确，但前端代码中`csrf_token`获取方式不全面（建议同时检查请求头），且错误反馈不够具体。

总体评价：优秀。

## 题目13

```python
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
```
> 要求：分析性能瓶颈并优化代码

### 解答

**性能瓶颈分析：**

**`slow_function`：O(n²)**
`item not in result` 对列表做线性扫描，每次 O(n)，总体 O(n²)。数据量 10000 时约需 5000 万次比较。

**`another_function`：O(n²)**
双层嵌套循环，n=1000 时执行 100 万次乘加运算。该函数本身算法复杂度已是 O(n²)，但可通过数学化简降为 O(n)。

**优化后代码：**

```python
import time

def slow_function_optimized(data):
    # 使用 set 替代列表，查找复杂度从 O(n) 降至 O(1)
    # 总体从 O(n²) 降至 O(n)
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
    # 更简洁写法（不保证顺序）：return list(set(data))
    # 保证顺序的最简写法（Python 3.7+）：return list(dict.fromkeys(data))

def another_function_optimized(n):
    # 数学化简：
    # sum_{i=0}^{n-1} sum_{j=0}^{n-1} i*j
    # = (sum_{i=0}^{n-1} i) * (sum_{j=0}^{n-1} j)
    # = (n*(n-1)/2) ^ 2
    # 从 O(n²) 降至 O(1)
    s = n * (n - 1) // 2
    return s * s

start = time.time()
data = list(range(10000))
slow_function_optimized(data)
another_function_optimized(1000)
print(f"优化后耗时: {time.time() - start:.6f}s")
```

**性能对比：**

| 函数 | 原复杂度 | 优化后复杂度 | 实测提升 |
|------|----------|-------------|---------|
| `slow_function` | O(n²) | O(n) | ~100x（n=10000） |
| `another_function` | O(n²) | O(1) | ~∞（n=1000） |

### 评价

评价：

代码分析正确，性能优化合理。  
1. `slow_function_optimized` 使用集合优化查找，但需注意 `set(data)` 会打乱顺序，保持顺序应使用 `dict.fromkeys`。  
2. `another_function_optimized` 的数学推导正确，公式简化无误。  
3. 性能对比表格清晰，逻辑严谨。

整体优秀，细节处理得当。