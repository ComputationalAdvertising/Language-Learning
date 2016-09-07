package org.caml.cross_lang;

import java.io.File;

/**
 * Created by zhouyong on 16/9/7.
 */
public class VisitCpp {
    public static void main(String[] args) {
        System.out.println("[INFO] normal all!");

        String userDir = System.getProperty("user.dir");
        System.out.println("[INFO] userDir: " + userDir);

        try {
            String curPath = new File(VisitCpp.class.getProtectionDomain().getCodeSource().getLocation().toURI().getPath()).getPath();
            System.out.println("[INFO] curPath: " + curPath);

            String soFilePath = curPath + File.separator + "test/test.txt";
            File file = new File(soFilePath);
            if (file.exists()) {
                System.out.println("[INFO] file: " + soFilePath + " exists!");
            } else {
                System.err.println("[INFO] file not in path: " + soFilePath);
            }

        } catch (Exception e) {
            System.out.println("[INFO] error!");
            e.printStackTrace();
        }
    }
}
