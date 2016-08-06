package org.caml.java.ml;

import java.io.*;

/**
 * Created by zhouyong on 16/6/5.
 */
public class IO {

    private static final String fileName = "/Users/zhouyong/myhome/2016-Planning/Language-Learning/java-learning/data/binary.model";

    public static void writeBinaryFile() {

        String str1 = "zhouyong";
        String str2 = "1";
        double d = 0.9999;

        try {
            DataOutputStream out = new DataOutputStream(new FileOutputStream(fileName));
            out.writeChars(str1);
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readBinaryFile() {
        try {
            DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        writeBinaryFile();
        readBinaryFile();
    }
}
