#include <iostream>
#include <vector>
#include <algorithm> // 用了std::swap, std::max_element等
#include <stack>     // for QuickSort Iterative
#include <cmath>     // 用于基数排序

#include <chrono>
#include "getData.cpp"

using namespace std;

// 打印数组辅助函数
void printArray(const vector<int>& arr) {
    cout << "[";
    for (int i = 0; i < arr.size(); i++) {
        cout << arr[i];
        if (i != arr.size() -1) cout << ", ";
    }
    cout << "]" << endl;
}

// 冒泡排序（优化版）
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    bool swapped;
    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

// 选择排序
void selectionSort(vector<int>& arr) {
    int n = arr.size(), minIndex;
    for (int i = 0; i < n - 1; i++) {
        minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) minIndex = j;
        }
        swap(arr[i], arr[minIndex]);
    }
}

// 插入排序
void insertionSort(vector<int>& arr) {
    int key, j;
    for (int i = 1; i < arr.size(); i++) {
        key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// 希尔排序
// TODO
void shellSort(vector<int>& arr) {
    int n = arr.size();
    // 初始步长 n/2，每次减半
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

void shellSortSedgewick(vector<int>& arr) {
    int n = arr.size();
    // 预生成的 Sedgewick 增量序列 (部分，足以覆盖很大范围)
    // 来源: 1, 8, 23, 77, 281, 1073, 4193, 16577...
    // 为了方便，这里列出常用的部分。实际工程中可动态生成。
    
    //vector<int> gaps = {1, 5, 19, 41, 109, 209, 505, 929, 2161, 3905, 8929}; 

    // 注：上面是 Sedgewick 几种公式的混合常用值，
    // 若严格按照 4^k + 3*2^(k-1) + 1: 1, 8, 23, 77...
    
    // 这里我们使用严格符合公式的动态生成方式：
    vector<int> sedgewickGaps;
    for (int k = 0; ; ++k) {
        long long gap = pow(4, k) + 3 * pow(2, k - 1) + 1;
        if (k == 0) gap = 1; // 公式在k=0时修正为1
        if (gap >= n) break;
        sedgewickGaps.push_back((int)gap);
    }

    // 希尔排序逻辑
    for (int k = sedgewickGaps.size() - 1; k >= 0; --k) {
        int gap = sedgewickGaps[k];
        for (int i = gap; i < n; ++i) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// 调整堆的辅助函数
void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

// 堆排序
void heapSort(vector<int>& arr) {
    int n = arr.size();
    // 1. 建堆 (从最后一个非叶子节点开始)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // 2. 一个个从堆顶取出元素
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]); // 把最大值移到末尾
        heapify(arr, i, 0);   // 调整剩余堆
    }
}

// 归并排序
// 合并两个有序子数组
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// 递归主体
void mergeSortHelper(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortHelper(arr, left, mid);
        mergeSortHelper(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void mergeSort(vector<int>& arr) {
    if (arr.empty()) return;
    mergeSortHelper(arr, 0, arr.size() - 1);
}

// 快速排序
// 经典三数取中排序
// 辅助：取中位数并交换
int medianOfThree(vector<int>& arr, int low, int high) {
    int mid = low + (high - low) / 2;
    // 简单的排序网络，保证 arr[low] < arr[mid] < arr[high]
    if (arr[low] > arr[mid]) swap(arr[low], arr[mid]);
    if (arr[low] > arr[high]) swap(arr[low], arr[high]);
    if (arr[mid] > arr[high]) swap(arr[mid], arr[high]);
    
    // 此时 mid 处是中位数，将其藏到 high-1 处（因为 high 肯定比 mid 大）
    // 也可以直接用 mid 做 pivot，这里为了配合标准 partition 逻辑，放到 right
    swap(arr[mid], arr[high - 1]);
    return arr[high - 1];
}

int partitionMedian(vector<int>& arr, int low, int high) {
    int pivot = medianOfThree(arr, low, high); // Pivot 已经在 high-1
    int i = low;
    int j = high - 1;
    
    while (true) {
        while (arr[++i] < pivot) {} // 从左找大
        while (arr[--j] > pivot) {} // 从右找小
        if (i < j) swap(arr[i], arr[j]);
        else break;
    }
    swap(arr[i], arr[high - 1]); // 恢复 Pivot
    return i;
}

void quickSortMedian(vector<int>& arr, int low, int high) {
    // cutoff: 小于 10 个元素可以用插入排序优化，这里为了演示纯粹性暂不加
    if (low + 10 <= high) {
        int pi = partitionMedian(arr, low, high);
        quickSortMedian(arr, low, pi - 1);
        quickSortMedian(arr, pi + 1, high);
    } else {
        // 小数组可以直接用插入排序
        insertionSort(arr); // 假设能调用上面的插入排序，需传参范围，此处略
    }
}

// 双轴快速排序
void dualPivotQuickSort(vector<int>& arr, int low, int high) {
    if (low >= high) return;

    // 1. 保证 P1 <= P2
    if (arr[low] > arr[high]) swap(arr[low], arr[high]);

    int p1 = arr[low];
    int p2 = arr[high];

    // lt: 小于 p1 的右边界
    // gt: 大于 p2 的左边界
    // i:  当前扫描位置
    int lt = low + 1;
    int gt = high - 1;
    int i = low + 1;

    while (i <= gt) {
        if (arr[i] < p1) {
            swap(arr[i], arr[lt]);
            lt++;
            i++;
        } else if (arr[i] > p2) {
            swap(arr[i], arr[gt]);
            gt--;
            // 注意：换回来的 arr[i] 可能还需判断，所以 i 不自增
        } else {
            i++;
        }
    }

    lt--; 
    gt++;
    
    swap(arr[low], arr[lt]);
    swap(arr[high], arr[gt]);

    dualPivotQuickSort(arr, low, lt - 1);
    dualPivotQuickSort(arr, lt + 1, gt - 1);
    dualPivotQuickSort(arr, gt + 1, high);
}

// 三路划分+回退插入排序
// 针对特定范围的插入排序
void insertionSortRange(vector<int>& arr, int low, int high) {
    for (int i = low + 1; i <= high; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void quickSort3Way(vector<int>& arr, int low, int high) {
    if (low >= high) return;

    // 优化 1: 小数组回退到插入排序 (阈值通常取 16-32)
    if (high - low <= 16) {
        insertionSortRange(arr, low, high);
        return;
    }

    // 优化 2: 随机选 Pivot 防止有序攻击 (也可以用三数取中)
    int randIdx = low + rand() % (high - low + 1);
    swap(arr[low], arr[randIdx]);

    int pivot = arr[low];
    int lt = low;        // arr[low...lt-1] < pivot
    int gt = high;       // arr[gt+1...high] > pivot
    int i = low + 1;     // arr[lt...i-1] == pivot

    // 三路划分核心逻辑
    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr[lt], arr[i]);
            lt++;
            i++;
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[gt]);
            gt--;
        } else {
            i++;
        }
    }

    // 递归时排除掉中间 == pivot 的部分
    quickSort3Way(arr, low, lt - 1);
    quickSort3Way(arr, gt + 1, high);
}

// TimSort
/*
 完整 TimSort for vector<int>
 主要模块：
  - minRunLength
  - countRunAndMakeAscending
  - binaryInsertionSort
  - run stack 管理与 merge 规则
  - mergeAt (带 gallop 模式)
  - 主流程：扫描 runs -> push -> collapse -> 最终合并
*/

#include <cassert>
#include <limits>

// 最小的 gallop 门槛，CPython 默认 7
static const int MIN_GALLOP = 7;

static int minRunLength(int n) {
    // 计算 minrun：返回 32..64 之间的值（与 CPython/Java 相近）
    int r = 0;
    while (n >= 64) {
        r |= (n & 1);
        n >>= 1;
    }
    return n + r;
}

// 二分插入排序：对 [left, right) 排序
static void binaryInsertionSort(std::vector<int>& a, int left, int right) {
    // right 为开区间
    for (int i = left + 1; i < right; ++i) {
        int key = a[i];
        // 在 [left, i) 中找到插入位置（upper_bound 保持稳定性）
        int insertPos = std::upper_bound(a.begin() + left, a.begin() + i, key) - a.begin();
        // 将 [insertPos, i) 向后移动一位
        for (int j = i; j > insertPos; --j) a[j] = a[j - 1];
        a[insertPos] = key;
    }
}

// 识别自然 run，并确保为升序（若是降序则翻转）
// 返回 run 的长度（>=1），返回的 run 范围是 [lo, lo+runLen)
static int countRunAndMakeAscending(std::vector<int>& a, int lo, int hi) {
    // hi 为数组长度（开区间）
    assert(lo < hi);
    int runHi = lo + 1;
    if (runHi == hi) return 1;

    // 判断是升序还是降序
    if (a[runHi++] < a[lo]) {
        // 降序
        while (runHi < hi && a[runHi] < a[runHi - 1]) ++runHi;
        // 翻转区间 [lo, runHi)
        std::reverse(a.begin() + lo, a.begin() + runHi);
    } else {
        // 升序
        while (runHi < hi && a[runHi] >= a[runHi - 1]) ++runHi;
    }
    return runHi - lo;
}

// Run 结构：起始索引与长度
struct Run {
    int start;
    int length;
    Run(int s = 0, int l = 0) : start(s), length(l) {}
};

// TimSorter 类封装所有状态（数组引用、临时缓冲、run stack、minGallop）
class TimSorter {
public:
    TimSorter(std::vector<int>& arr) : a(arr), tmp(), runStack(), minGallop(MIN_GALLOP) {
        tmp.reserve(arr.size() < 16 ? 16 : arr.size() / 2);
    }

    void sort() {
        int n = (int)a.size();
        if (n < 2) return;

        int minRun = minRunLength(n);
        int lo = 0;
        while (lo < n) {
            int runLen = countRunAndMakeAscending(a, lo, n);

            // 如果 run 太短，则扩展到 minRun（或到数组尾）
            if (runLen < minRun) {
                int force = std::min(minRun, n - lo);
                binaryInsertionSort(a, lo, lo + force);
                runLen = force;
            }

            pushRun(lo, runLen);
            mergeCollapse();

            lo += runLen;
        }

        mergeForceCollapse();
    }

private:
    std::vector<int>& a;
    std::vector<int> tmp;       // 临时缓冲区，用于合并，容量会按需调整
    std::vector<Run> runStack;  // run 栈
    int minGallop;

    // 将 run push 到栈
    void pushRun(int start, int length) {
        runStack.emplace_back(start, length);
    }

    // 合并规则 collapse：依据 TimSort 的 invariant 合并 run
    void mergeCollapse() {
        while (runStack.size() > 1) {
            int n = (int)runStack.size();
            bool merged = false;

            if (n >= 3) {
                int A = runStack[n - 3].length;
                int B = runStack[n - 2].length;
                int C = runStack[n - 1].length;
                if (A <= B + C) {
                    if (A < C) {
                        mergeAt(n - 3);
                    } else {
                        mergeAt(n - 2);
                    }
                    merged = true;
                }
            } 
            if (!merged) {
                // 再检查另一条条件
                int n2 = (int)runStack.size();
                if (n2 >= 2) {
                    int B = runStack[n2 - 2].length;
                    int C = runStack[n2 - 1].length;
                    if (B <= C) {
                        mergeAt(n2 - 2);
                        merged = true;
                    }
                }
            }

            if (!merged) break;
        }
    }

    // 强制全部合并（直到只剩一个 run）
    void mergeForceCollapse() {
        while (runStack.size() > 1) {
            int n = (int)runStack.size();
            if (n >= 2) {
                // 按照 CPython 的策略，总是合并倒数第二个与最后一个
                mergeAt(n - 2);
            }
        }
    }

    // 合并栈上索引为 i 和 i+1 的两个 run（i 从 0 开始）
    void mergeAt(int i) {
        assert(i >= 0 && i + 1 < (int)runStack.size());
        Run leftRun = runStack[i];
        Run rightRun = runStack[i + 1];

        // 调整栈：在位置 i 移除右 run，左 run 的长度更新为合并后长度
        runStack[i].length = leftRun.length + rightRun.length;
        // start 保持左 run 的 start
        runStack.erase(runStack.begin() + i + 1);

        // 做实际合并：merge [base1, base1+len1) 与 [base2, base2+len2)
        int base1 = leftRun.start;
        int len1 = leftRun.length;
        int base2 = rightRun.start;
        int len2 = rightRun.length;

        // 将较短的 run 复制到 tmp（CPython 风格：复制左侧）
        // 但为了更稳健（避免巨大复制导致 reallocation），我们按常见实现复制左侧
        // 确保 tmp 足够大
        if ((int)tmp.size() < len1) tmp.resize(len1);
        for (int k = 0; k < len1; ++k) tmp[k] = a[base1 + k];

        int iTmp = 0;            // tmp 的索引（0..len1-1）
        int j = base2;          // 右 run 在原数组中的当前索引
        int dest = base1;       // 写回位置

        // 计数用于触发 galloping 模式
        int countLeft = 0;
        int countRight = 0;

        // 主合并循环
        while (iTmp < len1 && j < base2 + len2) {
            if (tmp[iTmp] <= a[j]) {
                a[dest++] = tmp[iTmp++];
                countLeft++;
                countRight = 0;
            } else {
                a[dest++] = a[j++];
                countRight++;
                countLeft = 0;
            }

            // 检查是否达到 gallop 门槛
            if ((countLeft | countRight) >= minGallop) {
                // 进入 gallop 模式：分别寻找一段可以批量移动的区间
                if (countLeft >= minGallop) {
                    // 右端的元素被连续拷贝到 dest，说明 tmp 的当前元素小，实际上此处应 gallopRight(tmp[iTmp])
                    int k = gallopRight(tmp[iTmp], a, j, base2 + len2 - j);
                    // k 表示在右侧从 j 开始有 k 个元素 < tmp[iTmp]？
                    // 注意：gallopRight 的定义应返回右侧中第一个大于 key 的索引
                    for (int m = 0; m < k; ++m) a[dest++] = a[j++];

                } else {
                    // countRight >= minGallop
                    int k = gallopLeft(a[j], tmp, iTmp, len1 - iTmp);
                    for (int m = 0; m < k; ++m) a[dest++] = tmp[iTmp++];
                }
                // 进入/退出 gallop 后要调整 minGallop（自适应）
                if (minGallop > 1) --minGallop;
            }
        }

        // 如果左侧 tmp 还有残余，全部复制回去
        if (iTmp < len1) {
            for (int k = 0; k < len1 - iTmp; ++k) a[dest + k] = tmp[iTmp + k];
        }
        // 右侧剩余（如果有）已经在原位置，不需要移动（因为写入目标始终向右增长）
    }

    // gallopLeft: 在数组 a[base ... base+len-1] 中查找第一个 >= key 的位置（下标相对 base）
    // 这里用于从右边对 tmp 的元素进行二分扩展查找
    static int gallopLeft(int key, const std::vector<int>& a, int baseIndex, int len) {
        // baseIndex 是绝对索引，len 是右侧剩余长度
        // 我们在 tmp 上调用时会把 tmp 当作 a，baseIndex 也会传相对索引
        // 为简化接口：此处按 CPython 的语义写：返回在 a[baseIndex .. baseIndex+len) 中第一个大于 key 的位置（相对 baseIndex）
        int lo = 0;
        int hi = 1;
        if (len == 0) return 0;
        // 如果第一个元素大于 key 则位置为 0
        if (key > a[baseIndex]) {
            // 指数扩大 hi
            while (hi < len && key > a[baseIndex + hi]) {
                lo = hi;
                hi = (hi << 1) + 1;
                if (hi <= 0) { // 防止溢出，保险检查
                    hi = len;
                    break;
                }
            }
            if (hi > len) hi = len;
            // 在 [lo, hi) 上做二分查找
            lo = lo;
            return std::lower_bound(a.begin() + baseIndex + lo, a.begin() + baseIndex + hi, key) - (a.begin() + baseIndex);
        } else {
            return 0;
        }
    }

    // gallopRight: 在数组 a[base ... base+len-1] 中查找第一个 > key 的位置（相对 base）
    static int gallopRight(int key, const std::vector<int>& a, int baseIndex, int len) {
        int lo = 0;
        int hi = 1;
        if (len == 0) return 0;
        if (key < a[baseIndex]) {
            // 指数扩大 hi（注意比较方向）
            while (hi < len && key < a[baseIndex + hi]) {
                lo = hi;
                hi = (hi << 1) + 1;
                if (hi <= 0) {
                    hi = len;
                    break;
                }
            }
            if (hi > len) hi = len;
            return std::upper_bound(a.begin() + baseIndex + lo, a.begin() + baseIndex + hi, key) - (a.begin() + baseIndex);
        } else {
            return len;
        }
    }

}; // class TimSorter

// 外部接口
void timSort(std::vector<int>& v) {
    TimSorter sorter(v);
    sorter.sort();
}

// 计数排序
void countingSort(vector<int>& arr) {
    if (arr.empty()) return;
    
    int maxVal = *max_element(arr.begin(), arr.end());
    // 创建计数数组，初始化为0
    vector<int> count(maxVal + 1, 0);

    // 1. 统计频率
    for (int x : arr) count[x]++;

    // 2. 累加计数（确定位置）
    for (int i = 1; i <= maxVal; ++i) count[i] += count[i - 1];

    // 3. 构建输出数组 (反向遍历保证稳定性)
    vector<int> output(arr.size());
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }

    // 4. 拷贝回原数组
    arr = output;
}

// 基数排序专用的计数排序（针对某一位 exp）
void countSortForRadix(vector<int>& arr, int exp) {
    int n = arr.size();
    vector<int> output(n);
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(arr[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (int i = n - 1; i >= 0; i--) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        count[(arr[i] / exp) % 10]--;
    }
    arr = output;
}

// 基数排序
void radixSort(vector<int>& arr) {
    if (arr.empty()) return;
    int maxVal = *max_element(arr.begin(), arr.end());

    // 对每一位进行计数排序
    for (int exp = 1; maxVal / exp > 0; exp *= 10) countSortForRadix(arr, exp);
}

// 桶排序
void bucketSort(vector<int>& arr) {
    if (arr.empty()) return;
    int n = arr.size();
    
    int maxVal = *max_element(arr.begin(), arr.end());
    int minVal = *min_element(arr.begin(), arr.end());
    
    // 桶的数量，可根据数据分布调整
    int bucketCount = n;
    // 计算每个桶的范围
    double range = (double)(maxVal - minVal + 1) / bucketCount;

    vector<vector<int>> buckets(bucketCount);

    // 1. 分配入桶
    for (int i = 0; i < n; ++i) {
        int index = (arr[i] - minVal) / range;
        // 边界修正：防止最大值越界
        if (index >= bucketCount) index = bucketCount - 1; 
        buckets[index].push_back(arr[i]);
    }

    // 2. 桶内排序并合并
    int idx = 0;
    for (int i = 0; i < bucketCount; ++i) {
        if (!buckets[i].empty()) {
            sort(buckets[i].begin(), buckets[i].end()); // 桶内使用快排
            for (int val : buckets[i]) {
                arr[idx++] = val;
            }
        }
    }
}

int main() {
    // 读取大数据集
    //cout << "Loading nearly sorted dataset..." << endl;
    vector<int> largeData = DataGenerator::loadFromFile("data_large_random_1M.txt");

    if (largeData.empty()) {
        cout << "File not found! Please run the generator first." << endl;
        return 1;
    }

    /*
    for (int i = 0; i < 10; i++) {
        // 复制一份进行测试，以免修改原数据影响后续测试
        vector<int> testArr = largeData;

        //cout << "Sorting..." << endl;
        auto start = std::chrono::high_resolution_clock::now(); // 开始计时
        bubbleSort(testArr);
        //selectionSort(testArr);
        //insertionSort(testArr);
        //shellSortSedgewick(testArr);
        //heapSort(testArr);
        //mergeSort(testArr);
        //quickSortMedian(testArr, 0, 999999);
        //dualPivotQuickSort(testArr, 0, 999999);
        //quickSort3Way(testArr, 0, 999999);
        //countingSort(testArr);
        //radixSort(testArr);
        //bucketSort(testArr);
        //std::sort(testArr.begin(), testArr.end());
        auto end = std::chrono::high_resolution_clock::now(); // 结束计时
        std::chrono::duration<double, std::milli> duration = end - start;
        cout << "Bubble " << i << " : " << duration.count() / 1000 << "s" << endl;
    }

    for (int i = 0; i < 10; i++) {
        // 复制一份进行测试，以免修改原数据影响后续测试
        vector<int> testArr = largeData;

        //cout << "Sorting..." << endl;
        auto start = std::chrono::high_resolution_clock::now(); // 开始计时
        //bubbleSort(testArr);
        selectionSort(testArr);
        //insertionSort(testArr);
        //shellSortSedgewick(testArr);
        //heapSort(testArr);
        //mergeSort(testArr);
        //quickSortMedian(testArr, 0, 999999);
        //dualPivotQuickSort(testArr, 0, 999999);
        //quickSort3Way(testArr, 0, 999999);
        //countingSort(testArr);
        //radixSort(testArr);
        //bucketSort(testArr);
        //std::sort(testArr.begin(), testArr.end());
        auto end = std::chrono::high_resolution_clock::now(); // 结束计时
        std::chrono::duration<double, std::milli> duration = end - start;
        cout << "Select " << i << " : " << duration.count() / 1000 << "s" << endl;
    }
    */

    for (int i = 0; i < 10; i++) {
        // 复制一份进行测试，以免修改原数据影响后续测试
        vector<int> testArr = largeData;

        //cout << "Sorting..." << endl;
        auto start = std::chrono::high_resolution_clock::now(); // 开始计时
        //bubbleSort(testArr);
        //selectionSort(testArr);
        //insertionSort(testArr);
        //shellSortSedgewick(testArr);
        //heapSort(testArr);
        //mergeSort(testArr);
        //quickSortMedian(testArr, 0, 999999);
        //dualPivotQuickSort(testArr, 0, 999999);
        //quickSort3Way(testArr, 0, 999999);
        timSort(testArr);
        //countingSort(testArr);
        //radixSort(testArr);
        //bucketSort(testArr);
        //std::sort(testArr.begin(), testArr.end());
        auto end = std::chrono::high_resolution_clock::now(); // 结束计时
        std::chrono::duration<double, std::milli> duration = end - start;
        cout << "quick " << i << " : " << duration.count() << "ms" << endl;
    }
    return 0;
}