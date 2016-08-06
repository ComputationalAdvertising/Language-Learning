package org.caml.java.ml.io;

import java.io.*;

/**
 * Created by zhouyong on 16/6/5.
 */
public class FileUtils {

    public static void readToBuffer(StringBuffer buffer, InputStream in) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            String line;                // save content of each line.
            line = reader.readLine();   // 读取第一行
            while (line != null) {
                buffer.append(line).append("\n");
                line = reader.readLine();   // 读取下一行
            }
            reader.close();
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readToBuffer(StringBuffer buffer, String filePath) {
        try {
            InputStream in = new FileInputStream(filePath);
            String line;                // save content of each line.
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            line = reader.readLine();   // 读取第一行
            while (line != null) {
                buffer.append(line).append("\n");
                line = reader.readLine();   // 读取下一行
            }
            reader.close();
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String readFileToString(String filePath) {
        StringBuffer sb = new StringBuffer();
        FileUtils.readToBuffer(sb, filePath);
        return sb.toString();
    }
}
