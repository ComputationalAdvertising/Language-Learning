import com.alibaba.fastjson.JSON;

/**
 * Created by zhouyong on 16/8/19.
 */

public class Test {


    public static void main(String[] args ) {
        C1 c1 = new C1("zhou", 20, "山东");
        String jsonStr = JSON.toJSONString(c1);
        System.out.println("jsonStr:\n" + jsonStr);
    }

}
