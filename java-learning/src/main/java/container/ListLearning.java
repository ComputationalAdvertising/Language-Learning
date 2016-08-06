package container;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by zhouyong on 16/6/25.
 */
public class ListLearning {
    public static void main(String[] args) {
        List<Integer> integerList = Arrays.asList(1,3,2,4,5,6,7,8,10,9);
        List<Integer> subList = integerList.subList(0, 4);
        System.out.println(subList.toString());
        Collections.sort(subList);
        System.out.println(integerList.toString());
        Random random = new Random(47);
        Collections.shuffle(integerList, random);
        System.out.println(integerList.toString());
    }
}
