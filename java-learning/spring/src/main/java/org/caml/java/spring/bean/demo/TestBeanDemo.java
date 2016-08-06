package org.caml.java.spring.bean.demo;

import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.FileSystemXmlApplicationContext;

/**
 * Created by zhouyong on 16/6/5.
 */
public class TestBeanDemo {

    @Test
    public void test() throws Exception {
        //ApplicationContext ac = new ClassPathXmlApplicationContext("appcontext-demo.xml", getClass());
        ApplicationContext ac = new FileSystemXmlApplicationContext("file:///Users/zhouyong/myhome/2016-Planning/Language-Learning/java-learning/spring/src/main/resources/config/spring/appcontext-demo.xml");

        User user1 = (User) ac.getBean("user1");
        User user2 = (User) ac.getBean("user2");
        User user3 = (User) ac.getBean("user3");
        System.out.println(user1);
        System.out.println(user2);
        System.out.println(user3);
        System.out.println(user3.getAddressSet().toString());

        // failure 20160605
        Student student = (Student) ac.getBean("student");
        System.out.println(student);
    }


}
