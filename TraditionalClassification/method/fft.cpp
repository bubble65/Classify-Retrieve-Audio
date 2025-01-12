#include <complex>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
using namespace std;
using namespace std::chrono;

using cd = complex<double>;
using vcd = vector<cd>;

const double PI = acos(-1);

vcd string_to_coefficient(const string& num) {
    vcd cof;
    for (int i = num.length() - 1; i >= 0; i--) {
        cof.push_back(cd(num[i] - '0', 0));
    }
    return cof;
}


// 将两个系数数组补齐到2的幂长度
void pad_to_power_of_two(vcd& a, vcd& b) {
    int n = 1;
    while (n < a.size() + b.size()) {
        n <<= 1;
    }
    a.resize(n);
    b.resize(n);
}

// 检查是否为2的幂
bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// 递归版FFT实现
vcd fft(const vcd& a, int mode = 1) {
    int n = a.size();
    
    // 输入检验
    if (!is_power_of_two(n)) {
        throw invalid_argument("输入长度必须是2的幂");
    }
    if (n == 1) {
        return a;
    }
    cd omega = polar(1.0, mode * 2 * PI / n);
    vcd a_even(n/2), a_odd(n/2);
    for (int i = 0; i < n/2; i++) {
        a_even[i] = a[2*i];
        a_odd[i] = a[2*i + 1];
    }
    vcd y_even = fft(a_even, mode);
    vcd y_odd = fft(a_odd, mode);
    vcd y(n);
    cd omega_k = 1;
    for (int k = 0; k < n/2; k++) {
        y[k] = y_even[k] + omega_k * y_odd[k];
        y[k + n/2] = y_even[k] - omega_k * y_odd[k];
        omega_k *= omega;
    }
    return y;
}

vcd ifft(const vcd& a) {
    int n = a.size();

    // 输入检验
    if (!is_power_of_two(n)) {
        throw invalid_argument("输入长度必须是2的幂");
    }

    vcd res = fft(a, -1);
    for (auto& val : res) {
        val /= n;
    }
    return res;
}

int reverse_bits(int num, int bit_count) {
    int result = 0;
    for (int i = 0; i < bit_count; i++) {
        result = (result << 1) | (num & 1);
        num >>= 1;
    }
    return result;
}

// 迭代版FFT
vcd iterative_fft(const vcd& a) {
    int n = a.size();
    
    // 输入检验
    if (!is_power_of_two(n)) {
        throw invalid_argument("输入长度必须是2的幂");
    }
    
    vcd y(n);
    int bit_count = log2(n);
    for (int i = 0; i < n; i++) {
        y[reverse_bits(i, bit_count)] = a[i];
    }
    for (int len = 2; len <= n; len <<= 1) {
        double angle = 2 * PI / len;
        cd omega_len = polar(1.0, angle);
        
        for (int i = 0; i < n; i += len) {
            cd omega = 1;
            for (int j = 0; j < len/2; j++) {
                cd u = y[i + j];
                cd v = omega * y[i + j + len/2];
                y[i + j] = u + v;
                y[i + j + len/2] = u - v;
                omega *= omega_len;
            }
        }
    }
    
    return y;
}

// 测试函数
void run_fft_test() {
    // 准备测试数据：[1, 2, 3, 4]
    vcd test_data = {cd(1,0), cd(2,0), cd(3,0), cd(4,0)};
    
    cout << "原始数据: ";
    for (const auto& val : test_data) {
        cout << val << " ";
    }
    cout << "\n\n";
    
    // 测试递归FFT
    cout << "递归FFT结果:\n";
    vcd fft_result = fft(test_data);
    for (const auto& val : fft_result) {
        cout << val << "\n";
    }
    cout << "\n";
    
    // 测试IFFT
    cout << "IFFT结果:\n";
    vcd ifft_result1 = ifft(fft_result);
    for (const auto& val : ifft_result1) {
        cout << val << "\n";
    }
    cout << "\n";

    // 测试迭代FFT
    cout << "迭代FFT结果:\n";
    vcd iter_fft_result = iterative_fft(test_data);
    for (const auto& val : iter_fft_result) {
        cout << val << "\n";
    }
    cout << "\n";
}


// 使用FFT实现大整数乘法
string multiply_big_integers(const string& num1, const string& num2) {
    if (num1 == "0" || num2 == "0") return "0";
    vcd a = string_to_coefficient(num1);
    vcd b = string_to_coefficient(num2);
    pad_to_power_of_two(a, b);
    vcd fa = fft(a);
    vcd fb = fft(b);
    vcd fc(fa.size());
    for (size_t i = 0; i < fa.size(); i++) {
        fc[i] = fa[i] * fb[i];
    }
    vcd c = ifft(fc);
    vector<long long> result;
    long long carry = 0;
    for (const auto& val : c) {
        long long digit = round(val.real()) + carry;
        result.push_back(digit % 10);
        carry = digit / 10;
    }
    while (carry) {
        result.push_back(carry % 10);
        carry /= 10;
    }
    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }
    string answer;
    for (auto it = result.rbegin(); it != result.rend(); ++it) {
        answer += to_string(*it);
    }
    
    return answer.empty() ? "0" : answer;
}

string multiply_optimized(const string& num1, const string& num2) {
    if (num1 == "0" || num2 == "0") return "0";
    vcd a = string_to_coefficient(num1);
    vcd b = string_to_coefficient(num2);
    pad_to_power_of_two(a, b);
    int n = a.size();
    vcd p(n);
    for (int i = 0; i < n; i++) {
        p[i] = a[i] + cd(0, 1) * b[i];
    }
    vcd P = fft(p);
    vcd processed(n);
    for (int i = 0; i < n; i++)
    {
        processed[i] = P[i] * P[i];
    }
    vcd result = ifft(processed);
    vector<long long> digits;
    long long carry = 0;
    int valid_digits = num1.length() + num2.length();
    for (int i = 0; i < valid_digits; i++) {
        long long current = round(result[i].imag()/2) + carry;
        digits.push_back(current % 10);
        carry = current / 10;
    }
    while (carry > 0) {
        digits.push_back(carry % 10);
        carry /= 10;
    }
    while (digits.size() > 1 && digits.back() == 0) {
        digits.pop_back();
    }
    string answer;
    for (int i = digits.size() - 1; i >= 0; i--) {
        answer += to_string(digits[i]);
    }
    return answer;
}

void test_big_integer_multiplication() {
    vector<pair<string, string>> test_cases = {
        {"123", "456"},
        {"999", "999"},
        {"1234", "5678"},
        {"0", "1234"},
        {"99999", "99999"}
    };
    
    for (const auto& test : test_cases) {
        // string result = multiply_big_integers(test.first, test.second);
        string result = multiply_optimized(test.first, test.second);
        cout << test.first << " * " << test.second << " = " << result << "\n";
        // 验证结果
        long long expected = stoll(test.first) * stoll(test.second);
        cout << "Expected: " << expected << "\n";
        cout << (to_string(expected) == result ? "Correct!" : "Wrong!") << "\n\n";
    }
}



int main(int argc, char* argv[]) {
    // 首先检查参数数量是否正确
    // argc 总是至少为1，因为argv[0]是程序名称
    if (argc == 1){
        test_big_integer_multiplication();
        return 0;
    }
    if (argc != 3) {
        cout << "使用方法: " << argv[0] << " <第一个数> <第二个数>" << endl;
        cout << "示例: " << argv[0] << " 123 456" << endl;
        return 1;
    }
        // 获取命令行参数
    string num1 = argv[1];
    string num2 = argv[2];
    
    // 验证输入是否都是数字
    for (char c : num1) {
        if (!isdigit(c)) {
            cout << "错误：第一个参数必须是数字" << endl;
            return 1;
        }
    }
    
    for (char c : num2) {
        if (!isdigit(c)) {
            cout << "错误：第二个参数必须是数字" << endl;
            return 1;
        }
    }
    auto start = high_resolution_clock::now();
    string result1 = multiply_big_integers(num1, num2);
    auto end = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(end - start);
    
    start = high_resolution_clock::now();
    string result2 = multiply_optimized(num1, num2);
    end = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(end - start);
    
    cout << "Basic FFT implementation:\n";
    cout << num1 << " * " << num2 << " = " << result1 << "\n";
    cout << "Time taken(Basic): " << duration1.count() << " microseconds\n\n";
    
    cout << "Optimized implementation:\n";
    cout << num1 << " * " << num2 << " = " << result2 << "\n";
    cout << "Time taken(Optimized): " << duration2.count() << " microseconds\n";
    
    return 0;
}