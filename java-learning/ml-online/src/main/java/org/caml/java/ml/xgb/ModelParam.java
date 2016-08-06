package org.caml.java.ml.xgb;

import org.caml.java.ml.util.ModelReader;

/**
 * Created by zhouyong on 16/6/5.
 */
public class ModelParam {
    private float baseScore;
    private int numFeature;
    private int numClass;
    private int savedWithpBuffer;
    private int[] reserved;

    public ModelParam(ModelReader reader) {
    }

    public float getBaseScore() {
        return baseScore;
    }

    public void setBaseScore(float baseScore) {
        this.baseScore = baseScore;
    }

    public int getNumFeature() {
        return numFeature;
    }

    public void setNumFeature(int numFeature) {
        this.numFeature = numFeature;
    }

    public int getNumClass() {
        return numClass;
    }

    public void setNumClass(int numClass) {
        this.numClass = numClass;
    }

    public int getSavedWithpBuffer() {
        return savedWithpBuffer;
    }

    public void setSavedWithpBuffer(int savedWithpBuffer) {
        this.savedWithpBuffer = savedWithpBuffer;
    }

    public int[] getReserved() {
        return reserved;
    }

    public void setReserved(int[] reserved) {
        this.reserved = reserved;
    }
}
