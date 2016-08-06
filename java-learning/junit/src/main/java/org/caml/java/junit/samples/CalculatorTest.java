package org.caml.java.junit.samples;

import org.junit.*;
import org.junit.Assert;

/**
 * Created by zhouyong on 16/5/29.
 */
public class CalculatorTest {

    private Calculator cal = new Calculator();

    @BeforeClass
    public static void before() {
        System.out.println("[INFO] global beginning ...");
    }

    @AfterClass
    public static void after() {
        System.out.println("[INFO] global destroy ...");
    }

    @Before
    public void setUp() throws Exception {
        System.out.println("[INFO] a test beginning ...");
    }

    @After
    public void tearDown() throws Exception {
        System.out.println("[INFO] a test ending ...");
    }

    @Test
    @Ignore
    public void testAdd() {
        System.out.println("[INFO] testAdd ...");
        int result = cal.add(1, 2);
        Assert.assertEquals(3, result);
    }

    @Test (timeout = 1000)     // 单位：毫秒
    public void testSquareRoot() {
        System.out.println("[INFO] testSquareRoot ...");
        cal.squareRoot(4);
    }

    @Test (expected = Exception.class)
    public void testDivide() throws Exception {
        System.out.println("[INFO] test divide");
        cal.divide(4, 0);
    }
}
