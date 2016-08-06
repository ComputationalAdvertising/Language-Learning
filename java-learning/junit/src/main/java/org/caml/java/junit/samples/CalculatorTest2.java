package org.caml.java.junit.samples;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.junit.Assert;

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by zhouyong on 16/5/29.
 */

/**
 * 如果想给square方法（或其它method）多弄几个测试用例，按照上面的方法，我应该写好几个@Test方法来测试，
 * 或者每次测完再改一下输入的值和期望得到的值，好麻烦。JUnit提供如下的测试
 */

@RunWith(Parameterized.class)
public class CalculatorTest2 {

    private Calculator cal = new Calculator();
    private int param;
    private int result;

    /**
     * 构造函数，对变量进行初始化.
     * 定义一个待测试的类，并且定义两个变量，一个用于存放参数，一个用于存放期待的结果。
     *
     */
    public CalculatorTest2(int param, int result) {
        this.param = param;
        this.result = result;
    }

    @Parameters
    public static Collection data() {
        return Arrays.asList(new Object[][] {
                {2, 4},
                {0, 0},
                {-3, 9},
        });
    }

    @Test
    public void squareTest() {
        int tmp = cal.square(param);
        Assert.assertEquals(result, tmp);
    }
}
