package org.caml.java.spring.junit4;

/**
 * Created by zhouyong on 16/5/29.
 */
public class Calculator {

    public int add(int a, int b) {
        return a + b;
    }

    public int minus(int a, int b) {
        return a - b;
    }

    public int square(int n) {
        return n*n;
    }

    // bug: 死循环
    public void squareRoot(int n) {
        while (true) {
        }
    }

    public int multiply(int a, int b) {
        return a * b;
    }

    public int divide(int a, int b) throws Exception {
        if (b == 0) {
            throw new Exception("除数不能为零!");
        }
        return a / b;
    }
}
