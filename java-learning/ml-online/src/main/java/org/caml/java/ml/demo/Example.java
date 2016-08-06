package org.caml.java.ml.demo;

import org.caml.java.ml.io.FileUtils;
import org.caml.java.ml.util.ModelReader;

import java.io.InputStream;
import java.util.StringTokenizer;

/**
 * Created by zhouyong on 16/6/5.
 */
public class Example {

    public static final String modelPath = "dump.nice.txt";

    public static void main(String[] args) {
        InputStream in = Example.class.getClassLoader().getResourceAsStream(modelPath);
        StringBuffer sb = new StringBuffer();
        FileUtils.readToBuffer(sb, in);
        System.out.println(sb.toString());


    }
}
