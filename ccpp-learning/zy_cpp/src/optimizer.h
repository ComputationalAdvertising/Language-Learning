/*
 * File Name: optimizer.h
 * Author: zhouyong03@meituan.com
 * Created Time: 2016-11-01 16:54:51
 */
 
#include <iostream>
using namespace std;

class Optimizer {
  public:
    Optimizer() {  }
    ~Optimizer() {  }

    void Init(int value) {  }
    float Predict(float row) { return row }
    void Update(int row, float pred) {  }

};  // class Optimizer
