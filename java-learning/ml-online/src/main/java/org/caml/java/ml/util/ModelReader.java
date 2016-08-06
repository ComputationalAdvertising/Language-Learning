package org.caml.java.ml.util;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Created by zhouyong on 16/6/5.
 */
public class ModelReader implements Closeable {

    private InputStream stream;
    private byte[] buffer;

    public ModelReader(InputStream in) {
        try {
            this.stream = in;
        } catch (Exception e) {
            this.stream = null;
        }
    }

    private int fillBuffer(int numBytes) {
        if (buffer == null || buffer.length < numBytes) {
            buffer = new byte[numBytes];
        }
        int numBayesRead = 0;
        while (numBayesRead < numBytes) {
            int count;
            try {
                count = stream.read(buffer, numBayesRead, numBytes - numBayesRead);
            } catch (IOException e) {
                count = -1;
            }
            if (count < 0) {
                return numBayesRead;
            }
            numBayesRead += count;
        }
        return numBayesRead;
    }

    public byte[] readByteArray(int numBytes) {
        int numBytesRead = fillBuffer(numBytes);
        if (numBytesRead < numBytes) {
            try {
                throw new EOFException(String.format("Cannot read byte array: expected = %d, actual = %d", numBytes, numBytesRead));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        byte[] result = new byte[numBytes];
        System.arraycopy(buffer, 0, result, 0, numBytes);
        return result;
    }

    public int readInt() {
        int numBytesRead = fillBuffer(4);
        if (numBytesRead < 4) {
            try {
                throw new EOFException(String.format("Cannot read int value: " + numBytesRead));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getInt();
    }

    public int[] readIntArray(int numValues) {
        int numBytesRead = fillBuffer(numValues * 4);
        if (numBytesRead < numValues * 4) {
            try {
                throw new EOFException(String.format("Cannot read int value: " + numBytesRead));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

        int[] result = new int[numValues];
        for (int i = 0; i < numValues; ++i) {
            result[i] = byteBuffer.getInt();
        }
        return null;
    }

    public int readunsignedInt() {
        int result = readInt();
        if (result < 0) {
            try {
                throw new IOException("[ERROR] Cannot read unsigned int (overflow): " + result);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return result;
    }

    public long readLong() {
        int numBytesRead = fillBuffer(8);
        if (numBytesRead < 8) {
            try {
                throw new IOException("[ERROR] Cannot read long value: " + numBytesRead);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getLong();
    }

    public float asFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getFloat();
    }

    public float readFloat() {
        int numBytesRead = fillBuffer(4);
        if (numBytesRead < 4) {
            try {
                throw new IOException("[ERROR] Cannot read float value (shortage): " + numBytesRead);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getFloat();
    }

    public float[] readFloatArray(int numValues) {
        int numBytesRead = fillBuffer(numValues * 4);
        if (numBytesRead < numValues * 4) {
            try {
                throw new EOFException(
                        String.format("Cannot read float array (shortage): expected = %d, actual = %d",
                                numValues * 4, numBytesRead));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

        float[] result = new float[numValues];
        for (int i = 0; i < numValues; ++i) {
            result[i] = byteBuffer.getFloat();
        }
        return result;
    }



    public void close() {
        try {
            stream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public InputStream getStream() {
        return stream;
    }

    public void setStream(InputStream stream) {
        this.stream = stream;
    }
}
