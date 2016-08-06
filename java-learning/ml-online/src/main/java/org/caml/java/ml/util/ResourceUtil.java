package org.caml.java.ml.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * Created by zhouyong on 16/6/5.
 */
public class ResourceUtil {

    //private static final String path = "xx.txt";
    private static final String path = "dump.nice.txt";

    public static String fetchDataFromResource(String path) {
        try {
            InputStream inputStream = ResourceUtil.class.getClassLoader().getResourceAsStream(path);
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            line = reader.readLine();
            while (line != null) {
                System.out.println(line.trim());
                line = reader.readLine();
            }
        } catch (IOException e) {
            System.out.println("IOException");
        }
        return "";
    }

    public static void main(String[] args) {
        String result = fetchDataFromResource(path);
        System.out.println(result);
    }
}
