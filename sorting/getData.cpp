#include <iostream>
#include <vector>
#include <string>
#include <fstream>   // 文件操作
#include <algorithm> // sort, swap, reverse
#include <random>    // 现代随机数库
#include <set>

using namespace std;

class DataGenerator {
private:
    mt19937 rng; // 随机数引擎 (Mersenne Twister)

public:
    DataGenerator() {
        random_device rd;
        rng = mt19937(rd());
    }

    // 1. 保存数组到文件
    // 格式：第一行是元素数量，第二行是空格分隔的数据
    void saveToFile(const vector<int>& data, const string& filename) {
        ofstream outFile(filename);
        if (!outFile.is_open()) {
            cerr << "Error opening file: " << filename << endl;
            return;
        }

        outFile << data.size() << "\n";
        for (size_t i = 0; i < data.size(); ++i) {
            outFile << data[i] << (i == data.size() - 1 ? "" : " ");
        }
        outFile.close();
        cout << "Generated & Saved: " << filename << " (Size: " << data.size() << ")" << endl;
    }

    // 2. 读取文件到数组 (供你的排序程序调用)
    static vector<int> loadFromFile(const string& filename) {
        ifstream inFile(filename);
        vector<int> data;
        if (!inFile.is_open()) {
            cerr << "Error opening file: " << filename << endl;
            return data;
        }

        int n;
        inFile >> n; // 读取第一行的数量
        data.reserve(n);
        int val;
        while (inFile >> val) {
            data.push_back(val);
        }
        inFile.close();
        return data;
    }

    // --- 各类数据生成函数 ---

    // 类型 1 & 2: 完全随机数组
    vector<int> generateRandom(int n, int minVal, int maxVal) {
        vector<int> res(n);
        uniform_int_distribution<int> dist(minVal, maxVal);
        for (int& x : res) x = dist(rng);
        return res;
    }

    // 类型 3: 95% 有序 (基本有序)
    // 策略：先生成完全有序，然后随机挑选 5% 的位置替换为随机噪声，或者进行随机交换
    vector<int> generateNearlySorted(int n, double sortedPercent = 0.95) {
        vector<int> res(n);
        // 1. 先生成有序序列
        for(int i=0; i<n; ++i) res[i] = i;
        
        // 2. 破坏剩余的 (1 - percent) 部分
        // 这里的策略是：随机选取 5% 的对进行交换，或者随机改变 5% 的值
        // 为了保证严格的“无序度”，我们选择随机改变 5% 的元素的值
        int chaosCount = n * (1.0 - sortedPercent);
        uniform_int_distribution<int> distIndex(0, n - 1);
        uniform_int_distribution<int> distVal(0, n);

        for (int i = 0; i < chaosCount; ++i) {
            int idx = distIndex(rng);
            res[idx] = distVal(rng); // 替换为随机数，破坏有序性
        }
        return res;
    }

    // 类型 4: 完全逆序
    vector<int> generateReverse(int n) {
        vector<int> res(n);
        for (int i = 0; i < n; ++i) {
            res[i] = n - i;
        }
        return res;
    }

    // 类型 5: 仅包含少量特定数值 (重复元素极多)
    vector<int> generateFewUnique(int n, int uniqueCount) {
        vector<int> res(n);
        
        // 1. 先选出 uniqueCount 个特定的随机数作为“池子”
        vector<int> pool;
        uniform_int_distribution<int> distVal(0, 100000);
        for(int i=0; i<uniqueCount; ++i) pool.push_back(distVal(rng));

        // 2. 从池子中随机采样填满数组
        uniform_int_distribution<int> distPoolIndex(0, uniqueCount - 1);
        for(int i=0; i<n; ++i) {
            res[i] = pool[distPoolIndex(rng)];
        }
        return res;
    }
};

/*
int main() {
    DataGenerator gen;

    // 1. 元素数量为 100 的随机整数数组
    auto data1 = gen.generateRandom(100, 0, 1000);
    gen.saveToFile(data1, "data_small_random_100.txt");

    // 2. 元素数量为 1,000,000 的随机整数数组
    auto data2 = gen.generateRandom(1000000, 0, 10000000);
    gen.saveToFile(data2, "data_large_random_1M.txt");

    // 3. 95% 元素已有序的、数量 10,000
    auto data3 = gen.generateNearlySorted(10000, 0.95);
    gen.saveToFile(data3, "data_nearly_sorted_10k.txt");

    // 4. 完全逆序的、数量 10,000
    auto data4 = gen.generateReverse(10000);
    gen.saveToFile(data4, "data_reverse_10k.txt");

    // 5. 仅包含 10 种数值的、数量 10,000 (测试三路快排专用)
    auto data5 = gen.generateFewUnique(10000, 10);
    gen.saveToFile(data5, "data_few_unique_10k.txt");

    return 0;
}
*/