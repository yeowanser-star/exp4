#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <clocale>
#ifdef _WIN32
#include <windows.h>
#endif
using namespace std;

static string removeNewlines(const string& s) {
    string out;
    out.reserve(s.size());
    for (char c : s) if (c != '\n' && c != '\r') out.push_back(c);
    return out;
}

// ==================== BoundingBox Class ====================

class BoundingBox {
public:
    float x, y;         // 中心点坐标
    float w, h;         // 宽度和高度
    float confidence;   // 置信度
    
    BoundingBox(float x = 0, float y = 0, float w = 0, float h = 0, float confidence = 0)
        : x(x), y(y), w(w), h(h), confidence(confidence) {}
    
    // Get bounding box coordinates (x1, y1, x2, y2)
    void getCoords(float& x1, float& y1, float& x2, float& y2) const {
        x1 = x - w / 2.0f;
        y1 = y - h / 2.0f;
        x2 = x + w / 2.0f;
        y2 = y + h / 2.0f;
    }
    
    // Compute area
    float area() const {
        return w * h;
    }
    
    // Display info
    string toString() const {
        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "BBox(x=" << x << ", y=" << y << ", w=" << w 
           << ", h=" << h << ", conf=" << confidence << ")";
        return ss.str();
    }
};

// ==================== SortAlgorithm Base Class ====================

class SortAlgorithm {
public:
    virtual ~SortAlgorithm() {}
    virtual string getName() const = 0;
    virtual void sort(vector<float>& arr) const = 0;
    virtual void sort(vector<BoundingBox>& boxes) const = 0;
};

// ==================== Quick Sort ====================

class QuickSort : public SortAlgorithm {
public:
    string getName() const override { return "QuickSort"; }
    
    void sort(vector<float>& arr) const override {
        if (arr.empty()) return;
        quickSort(arr, 0, arr.size() - 1);
    }
    
    void sort(vector<BoundingBox>& boxes) const override {
        if (boxes.empty()) return;
        quickSortBoxes(boxes, 0, boxes.size() - 1);
    }
    
private:
    void quickSort(vector<float>& arr, int low, int high) const {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    int partition(vector<float>& arr, int low, int high) const {
        float pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        return i + 1;
    }
    
    void quickSortBoxes(vector<BoundingBox>& boxes, int low, int high) const {
        if (low < high) {
            int pi = partitionBoxes(boxes, low, high);
            quickSortBoxes(boxes, low, pi - 1);
            quickSortBoxes(boxes, pi + 1, high);
        }
    }
    
    int partitionBoxes(vector<BoundingBox>& boxes, int low, int high) const {
        float pivot = boxes[high].confidence;
        int i = low - 1;
        
        for (int j = low; j <= high - 1; j++) {
            if (boxes[j].confidence < pivot) {
                i++;
                swap(boxes[i], boxes[j]);
            }
        }
        swap(boxes[i + 1], boxes[high]);
        return i + 1;
    }
};

// ==================== Merge Sort ====================

class MergeSort : public SortAlgorithm {
public:
    string getName() const override { return "MergeSort"; }
    
    void sort(vector<float>& arr) const override {
        if (arr.empty()) return;
        mergeSort(arr, 0, arr.size() - 1);
    }
    
    void sort(vector<BoundingBox>& boxes) const override {
        if (boxes.empty()) return;
        mergeSortBoxes(boxes, 0, boxes.size() - 1);
    }
    
private:
    void mergeSort(vector<float>& arr, int left, int right) const {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }
    
    void merge(vector<float>& arr, int left, int mid, int right) const {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        vector<float> leftArr(n1), rightArr(n2);
        
        for (int i = 0; i < n1; i++)
            leftArr[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            rightArr[j] = arr[mid + 1 + j];
        
        int i = 0, j = 0, k = left;
        
        while (i < n1 && j < n2) {
            if (leftArr[i] <= rightArr[j]) {
                arr[k] = leftArr[i];
                i++;
            } else {
                arr[k] = rightArr[j];
                j++;
            }
            k++;
        }
        
        while (i < n1) {
            arr[k] = leftArr[i];
            i++;
            k++;
        }
        
        while (j < n2) {
            arr[k] = rightArr[j];
            j++;
            k++;
        }
    }
    
    void mergeSortBoxes(vector<BoundingBox>& boxes, int left, int right) const {
        if (left < right) {
            int mid = left + (right - left) / 2;
            mergeSortBoxes(boxes, left, mid);
            mergeSortBoxes(boxes, mid + 1, right);
            mergeBoxes(boxes, left, mid, right);
        }
    }
    
    void mergeBoxes(vector<BoundingBox>& boxes, int left, int mid, int right) const {
        int n1 = mid - left + 1;
        int n2 = right - mid;
        
        vector<BoundingBox> leftArr(n1), rightArr(n2);
        
        for (int i = 0; i < n1; i++)
            leftArr[i] = boxes[left + i];
        for (int j = 0; j < n2; j++)
            rightArr[j] = boxes[mid + 1 + j];
        
        int i = 0, j = 0, k = left;
        
        while (i < n1 && j < n2) {
            if (leftArr[i].confidence <= rightArr[j].confidence) {
                boxes[k] = leftArr[i];
                i++;
            } else {
                boxes[k] = rightArr[j];
                j++;
            }
            k++;
        }
        
        while (i < n1) {
            boxes[k] = leftArr[i];
            i++;
            k++;
        }
        
        while (j < n2) {
            boxes[k] = rightArr[j];
            j++;
            k++;
        }
    }
};

// ==================== Heap Sort ====================

class HeapSort : public SortAlgorithm {
public:
    string getName() const override { return "HeapSort"; }
    
    void sort(vector<float>& arr) const override {
        int n = arr.size();
        
        // 构建最大堆
        for (int i = n / 2 - 1; i >= 0; i--)
            heapify(arr, n, i);
        
        // 一个个取出元素
        for (int i = n - 1; i > 0; i--) {
            swap(arr[0], arr[i]);
            heapify(arr, i, 0);
        }
    }
    
    void sort(vector<BoundingBox>& boxes) const override {
        int n = boxes.size();
        
        // 构建最大堆
        for (int i = n / 2 - 1; i >= 0; i--)
            heapifyBoxes(boxes, n, i);
        
        // 一个个取出元素
        for (int i = n - 1; i > 0; i--) {
            swap(boxes[0], boxes[i]);
            heapifyBoxes(boxes, i, 0);
        }
    }
    
private:
    void heapify(vector<float>& arr, int n, int i) const {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && arr[left] > arr[largest])
            largest = left;
        
        if (right < n && arr[right] > arr[largest])
            largest = right;
        
        if (largest != i) {
            swap(arr[i], arr[largest]);
            heapify(arr, n, largest);
        }
    }
    
    void heapifyBoxes(vector<BoundingBox>& boxes, int n, int i) const {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && boxes[left].confidence > boxes[largest].confidence)
            largest = left;
        
        if (right < n && boxes[right].confidence > boxes[largest].confidence)
            largest = right;
        
        if (largest != i) {
            swap(boxes[i], boxes[largest]);
            heapifyBoxes(boxes, n, largest);
        }
    }
};

// ==================== Insertion Sort ====================

class InsertionSort : public SortAlgorithm {
public:
    string getName() const override { return "InsertionSort"; }
    
    void sort(vector<float>& arr) const override {
        int n = arr.size();
        for (int i = 1; i < n; i++) {
            float key = arr[i];
            int j = i - 1;
            
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    
    void sort(vector<BoundingBox>& boxes) const override {
        int n = boxes.size();
        for (int i = 1; i < n; i++) {
            BoundingBox key = boxes[i];
            int j = i - 1;
            
            while (j >= 0 && boxes[j].confidence > key.confidence) {
                boxes[j + 1] = boxes[j];
                j--;
            }
            boxes[j + 1] = key;
        }
    }
};

// ==================== 数据生成器 ====================

class DataGenerator {
public:
    enum DistributionType {
        RANDOM,
        CLUSTERED
    };
    
    static vector<BoundingBox> generateBoxes(int numBoxes, DistributionType type, 
                                           int imageWidth = 1000, int imageHeight = 1000,
                                           float minSize = 10, float maxSize = 200,
                                           int numClusters = 5) {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<float> confDist(0.0f, 1.0f);
        
        if (type == RANDOM) {
            return generateRandomBoxes(numBoxes, imageWidth, imageHeight, minSize, maxSize, gen, confDist);
        } else {
            return generateClusteredBoxes(numBoxes, imageWidth, imageHeight, minSize, maxSize, numClusters, gen, confDist);
        }
    }
    
private:
    static vector<BoundingBox> generateRandomBoxes(int numBoxes, int imageWidth, int imageHeight,
                                                 float minSize, float maxSize,
                                                 mt19937& gen, uniform_real_distribution<float>& confDist) {
        vector<BoundingBox> boxes;
        uniform_real_distribution<float> xDist(maxSize/2, imageWidth - maxSize/2);
        uniform_real_distribution<float> yDist(maxSize/2, imageHeight - maxSize/2);
        uniform_real_distribution<float> sizeDist(minSize, maxSize);
        
        for (int i = 0; i < numBoxes; i++) {
            float x = xDist(gen);
            float y = yDist(gen);
            float w = sizeDist(gen);
            float h = sizeDist(gen);
            float confidence = confDist(gen);
            
            // 确保边界框在图像内
            if (x - w/2 < 0) x = w/2;
            if (x + w/2 > imageWidth) x = imageWidth - w/2;
            if (y - h/2 < 0) y = h/2;
            if (y + h/2 > imageHeight) y = imageHeight - h/2;
            
            boxes.emplace_back(x, y, w, h, confidence);
        }
        
        return boxes;
    }
    
    static vector<BoundingBox> generateClusteredBoxes(int numBoxes, int imageWidth, int imageHeight,
                                                    float minSize, float maxSize, int numClusters,
                                                    mt19937& gen, uniform_real_distribution<float>& confDist) {
        vector<BoundingBox> boxes;
        uniform_real_distribution<float> sizeDist(minSize, maxSize);
        
        // 生成聚集中心
        vector<pair<float, float>> clusterCenters;
        uniform_real_distribution<float> cxDist(maxSize, imageWidth - maxSize);
        uniform_real_distribution<float> cyDist(maxSize, imageHeight - maxSize);
        
        for (int i = 0; i < numClusters; i++) {
            clusterCenters.emplace_back(cxDist(gen), cyDist(gen));
        }
        
        // 根据聚集中心生成边界框
        int boxesPerCluster = numBoxes / numClusters;
        normal_distribution<float> clusterDist(0.0f, imageWidth / (numClusters * 2.0f));
        
        for (int i = 0; i < numClusters; i++) {
            float cx = clusterCenters[i].first;
            float cy = clusterCenters[i].second;
            
            int clusterBoxes = (i == numClusters - 1) ? 
                              numBoxes - boxesPerCluster * (numClusters - 1) : 
                              boxesPerCluster;
            
            for (int j = 0; j < clusterBoxes; j++) {
                float x = cx + clusterDist(gen);
                float y = cy + clusterDist(gen);
                float w = sizeDist(gen);
                float h = sizeDist(gen);
                float confidence = confDist(gen);
                
                // 确保边界框在图像内
                x = max(maxSize/2.0f, min(x, imageWidth - maxSize/2.0f));
                y = max(maxSize/2.0f, min(y, imageHeight - maxSize/2.0f));
                
                if (x - w/2 < 0) x = w/2;
                if (x + w/2 > imageWidth) x = imageWidth - w/2;
                if (y - h/2 < 0) y = h/2;
                if (y + h/2 > imageHeight) y = imageHeight - h/2;
                
                boxes.emplace_back(x, y, w, h, confidence);
            }
        }
        
        return boxes;
    }
};

// ==================== NMS算法 ====================

class NMS {
public:
    // 计算IoU（交并比）
    static float computeIoU(const BoundingBox& box1, const BoundingBox& box2) {
        float box1_x1, box1_y1, box1_x2, box1_y2;
        float box2_x1, box2_y1, box2_x2, box2_y2;
        
        box1.getCoords(box1_x1, box1_y1, box1_x2, box1_y2);
        box2.getCoords(box2_x1, box2_y1, box2_x2, box2_y2);
        
        // 计算交集坐标
        float x1 = max(box1_x1, box2_x1);
        float y1 = max(box1_y1, box2_y1);
        float x2 = min(box1_x2, box2_x2);
        float y2 = min(box1_y2, box2_y2);
        
        // 计算交集面积
        float intersection = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
        
        // 计算并集面积
        float unionArea = box1.area() + box2.area() - intersection;
        
        // 计算IoU
        if (unionArea == 0) return 0.0f;
        return intersection / unionArea;
    }
    
    // 应用NMS算法
    static vector<BoundingBox> applyNMS(const vector<BoundingBox>& boxes, float iouThreshold = 0.5f) {
        if (boxes.empty()) return {};
        
        // 复制并排序（按置信度从高到低）
        vector<BoundingBox> sortedBoxes = boxes;
        sort(sortedBoxes.begin(), sortedBoxes.end(),
             [](const BoundingBox& a, const BoundingBox& b) {
                 return a.confidence > b.confidence;
             });
        
        vector<BoundingBox> selectedBoxes;
        vector<bool> keep(sortedBoxes.size(), true);
        
        for (size_t i = 0; i < sortedBoxes.size(); i++) {
            if (!keep[i]) continue;
            
            selectedBoxes.push_back(sortedBoxes[i]);
            
            for (size_t j = i + 1; j < sortedBoxes.size(); j++) {
                if (!keep[j]) continue;
                
                float iou = computeIoU(sortedBoxes[i], sortedBoxes[j]);
                if (iou > iouThreshold) {
                    keep[j] = false;
                }
            }
        }
        
        return selectedBoxes;
    }
};

// ==================== 性能测试器 ====================

class PerformanceTester {
public:
    struct TestResult {
        string algorithmName;
        vector<int> numBoxes;
        vector<double> randomTimes;
        vector<double> clusteredTimes;
    };
    
    static TestResult testSortingAlgorithm(const SortAlgorithm& algorithm, 
                                         const vector<int>& numBoxesList,
                                         int runs = 5) {
        TestResult result;
        result.algorithmName = removeNewlines(algorithm.getName());
        result.numBoxes = numBoxesList;
        
        cout << "Testing algorithm: " << removeNewlines(algorithm.getName()) << endl;

        // test random distribution
        cout << "Random distribution:" << endl;
        for (int numBoxes : numBoxesList) {
            // 生成置信度数据
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            vector<float> confidences;
            for (int i = 0; i < numBoxes; i++) {
                confidences.push_back(dist(gen));
            }
            
            // measure runtime
            double totalTime = 0;
            for (int r = 0; r < runs; r++) {
                vector<float> testData = confidences;
                auto start = chrono::high_resolution_clock::now();
                algorithm.sort(testData);
                auto end = chrono::high_resolution_clock::now();
                totalTime += chrono::duration<double>(end - start).count();
            }
            
              double avgTime = totalTime / runs;
              result.randomTimes.push_back(avgTime);
              cout << "  " << numBoxes << " boxes: " << fixed << setprecision(6)
                  << avgTime << " s" << endl;
        }
        
        // test clustered distribution
        cout << "Clustered distribution:" << endl;
        for (int numBoxes : numBoxesList) {
            // 生成边界框数据
            auto boxes = DataGenerator::generateBoxes(numBoxes, DataGenerator::CLUSTERED);
            
            // 提取置信度
            vector<float> confidences;
            for (const auto& box : boxes) {
                confidences.push_back(box.confidence);
            }
            
            // 测量运行时间
            double totalTime = 0;
            for (int r = 0; r < runs; r++) {
                vector<float> testData = confidences;
                auto start = chrono::high_resolution_clock::now();
                algorithm.sort(testData);
                auto end = chrono::high_resolution_clock::now();
                totalTime += chrono::duration<double>(end - start).count();
            }
            
              double avgTime = totalTime / runs;
              result.clusteredTimes.push_back(avgTime);
              cout << "  " << numBoxes << " boxes: " << fixed << setprecision(6)
                  << avgTime << " s" << endl;
        }
        
        cout << endl;
        return result;
    }
    
    static TestResult testNMSAlgorithm(const vector<int>& numBoxesList,
                                     int runs = 5) {
        TestResult result;
        result.algorithmName = "NMS";
        result.numBoxes = numBoxesList;
        cout << "Testing algorithm: NMS" << endl;

        // test random distribution
        cout << "Random distribution:" << endl;
        for (int numBoxes : numBoxesList) {
            double totalTime = 0;
            
            for (int r = 0; r < runs; r++) {
                auto boxes = DataGenerator::generateBoxes(numBoxes, DataGenerator::RANDOM);
                
                auto start = chrono::high_resolution_clock::now();
                auto selected = NMS::applyNMS(boxes, 0.5f);
                auto end = chrono::high_resolution_clock::now();
                
                totalTime += chrono::duration<double>(end - start).count();
            }
            
              double avgTime = totalTime / runs;
              result.randomTimes.push_back(avgTime);
              cout << "  " << numBoxes << " boxes: " << fixed << setprecision(6)
                  << avgTime << " s" << endl;
        }
        
        // test clustered distribution
        cout << "Clustered distribution:" << endl;
        for (int numBoxes : numBoxesList) {
            double totalTime = 0;
            
            for (int r = 0; r < runs; r++) {
                auto boxes = DataGenerator::generateBoxes(numBoxes, DataGenerator::CLUSTERED);
                
                auto start = chrono::high_resolution_clock::now();
                auto selected = NMS::applyNMS(boxes, 0.5f);
                auto end = chrono::high_resolution_clock::now();
                
                totalTime += chrono::duration<double>(end - start).count();
            }
            
              double avgTime = totalTime / runs;
              result.clusteredTimes.push_back(avgTime);
              cout << "  " << numBoxes << " boxes: " << fixed << setprecision(6)
                  << avgTime << " s" << endl;
        }
        
        cout << endl;
        return result;
    }
    
    static void printResults(const vector<TestResult>& results) {
        cout << "\n" << string(80, '=') << endl;
        cout << "Performance test results summary" << endl;
        cout << string(80, '=') << endl;
        
        for (const auto& result : results) {
            cout << "\nAlgorithm: " << result.algorithmName << endl;
            cout << string(40, '-') << endl;
            cout << "NumBoxes\tRandom(s)\tClustered(s)" << endl;
            
            for (size_t i = 0; i < result.numBoxes.size(); i++) {
                 cout << result.numBoxes[i] << "\t\t"
                     << fixed << setprecision(6) << result.randomTimes[i] << "\t\t"
                     << result.clusteredTimes[i] << endl;
            }
        }
    }
    
    static void runCompleteTest() {
        cout << string(80, '=') << endl;
        cout << "Sorting and NMS performance test system" << endl;
        cout << string(80, '=') << endl;
        
        // 定义测试的边界框数量（缩小以便快速验证）
        vector<int> numBoxesList = {100, 500, 1000};
        
        // 创建排序算法实例
        vector<unique_ptr<SortAlgorithm>> algorithms;
        algorithms.push_back(make_unique<QuickSort>());
        algorithms.push_back(make_unique<MergeSort>());
        algorithms.push_back(make_unique<HeapSort>());
        algorithms.push_back(make_unique<InsertionSort>());
        
        vector<TestResult> allResults;
        
        // 测试排序算法
        for (const auto& algorithm : algorithms) {
            string name = removeNewlines(algorithm->getName());
            // 跳过插入排序在大规模数据上的测试
            if (name == "插入排序" || name == "InsertionSort") {
                vector<int> smallNumBoxesList = {100, 500, 1000};
                allResults.push_back(testSortingAlgorithm(*algorithm, smallNumBoxesList, 3));
            } else {
                allResults.push_back(testSortingAlgorithm(*algorithm, numBoxesList, 3));
            }
        }
        
        // 测试NMS算法
        allResults.push_back(testNMSAlgorithm(numBoxesList, 3));
        
        // 打印结果汇总
        printResults(allResults);
        
        // 生成CSV文件
        generateCSV(allResults, "performance_results.csv");
        
        // 验证排序算法正确性
        validateAlgorithms();
        
        // 展示NMS示例
        demonstrateNMS();
    }
    
    static void generateCSV(const vector<TestResult>& results, const string& filename) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Unable to create CSV file: " << filename << endl;
            return;
        }
        
        // 写入表头
        file << "Algorithm,NumBoxes,RandomDistribution,ClusteredDistribution\n";
        
        // 写入数据
        for (const auto& result : results) {
            for (size_t i = 0; i < result.numBoxes.size(); i++) {
                file << result.algorithmName << ","
                     << result.numBoxes[i] << ","
                     << result.randomTimes[i] << ","
                     << result.clusteredTimes[i] << "\n";
            }
        }
        
        file.close();
        cout << "\nResults saved to: " << filename << endl;
    }
    
    static void validateAlgorithms() {
        cout << "\n" << string(80, '=') << endl;
        cout << "Sorting algorithm correctness validation" << endl;
        cout << string(80, '=') << endl;
        
        // 生成测试数据
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        vector<float> testData;
        for (int i = 0; i < 100; i++) {
            testData.push_back(dist(gen));
        }
        
        // 使用STL排序作为参考
        vector<float> reference = testData;
        sort(reference.begin(), reference.end());
        
        // 测试每种算法
        vector<unique_ptr<SortAlgorithm>> algorithms;
        algorithms.push_back(make_unique<QuickSort>());
        algorithms.push_back(make_unique<MergeSort>());
        algorithms.push_back(make_unique<HeapSort>());
        algorithms.push_back(make_unique<InsertionSort>());
        
        for (const auto& algorithm : algorithms) {
            vector<float> testCopy = testData;
            algorithm->sort(testCopy);

            bool isCorrect = true;
            for (size_t i = 0; i < testCopy.size(); i++) {
                if (abs(testCopy[i] - reference[i]) > 1e-6) {
                    isCorrect = false;
                    break;
                }
            }

            cout << removeNewlines(algorithm->getName()) << ": "
                 << (isCorrect ? "✓ 正确" : "✗ 错误") << endl;
        }
    }
    
    static void demonstrateNMS() {
        cout << "\n" << string(80, '=') << endl;
        cout << "NMS算法示例" << endl;
        cout << string(80, '=') << endl;

        // 生成示例边界框
        auto boxes = DataGenerator::generateBoxes(20, DataGenerator::CLUSTERED,
                                                 1000, 1000, 10, 100, 3);

        cout << "生成 " << boxes.size() << " 个边界框" << endl;
        cout << "前5个边界框:" << endl;
        for (int i = 0; i < min(5, (int)boxes.size()); i++) {
            cout << "  " << (i+1) << ". " << boxes[i].toString() << endl;
        }

        // 应用NMS
        float iouThreshold = 0.5f;
        auto selectedBoxes = NMS::applyNMS(boxes, iouThreshold);

        cout << "\n应用NMS (IoU阈值=" << iouThreshold << ") 后，选择 " 
             << selectedBoxes.size() << " 个边界框" << endl;
        cout << "选择的边界框:" << endl;
        for (int i = 0; i < min(5, (int)selectedBoxes.size()); i++) {
            cout << "  " << (i+1) << ". " << selectedBoxes[i].toString() << endl;
        }
    }
};

// ==================== 主函数 ====================

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    #endif
    setlocale(LC_ALL, "");
    // 运行完整的性能测试
    PerformanceTester::runCompleteTest();
    
    return 0;
}

 