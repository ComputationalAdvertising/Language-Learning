package org.caml.java.ml;

import org.caml.java.ml.util.ModelReader;

import java.io.InputStream;

public class Predictor {


    public Predictor(InputStream in) {
        ModelReader reader = new ModelReader(in);
        if (reader == null) {
            System.exit(-1);
        }
    }

    public static void main(String[] args) {

        System.out.println("zhouyong");
    }
}
