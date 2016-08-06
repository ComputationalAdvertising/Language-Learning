package org.caml.java.junit.samples;

/**
 * Created by zhouyong on 16/5/29.
 */

import org.caml.java.spring.junit4.CalculatorTest;
import org.caml.java.spring.junit4.CalculatorTest2;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/**
 * 打包测试类： Suite, Suite.SuiteClasses
 */
@RunWith(Suite.class)
@Suite.SuiteClasses({CalculatorTest.class, CalculatorTest2.class})
public class CalculatorTestSuite {
}
