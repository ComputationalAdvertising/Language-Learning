package utils;

import java.util.*;

/**
 * Created by zhouyong on 16/6/25.
 */
public class ContainerUtils {

    /**
     * 将容器转化为字符串，适用于不同容器（Iterator）以及不同类型的方法（<T>）
     * @param iterator
     * @param sep
     * @return
     */
    public static String display(Iterator iterator, String sep) {
        StringBuilder sb = new StringBuilder();
        while (iterator.hasNext()) {
            //T t = (T) iterator.next();
            sb.append(iterator.next().toString()).append(iterator.hasNext() ? sep : "");
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        List<String> stringList = new ArrayList<String>();
        int n = 300000;
        for (int i = 0; i < n; i++) {
            stringList.add(String.valueOf(i));
        }
        System.out.println("xx");
        long startTime = System.currentTimeMillis();
        display(stringList.iterator(), ",");
        long endTime = System.currentTimeMillis();

        Calendar calendar = Calendar.getInstance();
        calendar.setTimeInMillis(endTime-startTime);

        System.out.println(calendar.get(Calendar.MINUTE) + " m " + calendar.get(Calendar.SECOND) + " s " + calendar.get(Calendar.MILLISECOND) + " 微秒");

        System.out.println();

        Set<Integer> integerSet = new HashSet<Integer>(Arrays.asList(1,2,3,4));
        System.out.println(display(integerSet.iterator(), "^"));


    }
}
