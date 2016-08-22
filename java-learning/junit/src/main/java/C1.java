/**
 * Created by zhouyong on 16/8/19.
 */
public class C1 {

    private static String name;
    private static int age;
    private static String addr;

    public C1(String name, int age, String addr) {
        setName(name);
        setAge(age);
        setAddr(addr);
    }

    public static String getName() {
        return name;
    }

    public static void setName(String name) {
        C1.name = name;
    }

    public static int getAge() {
        return age;
    }

    public static void setAge(int age) {
        C1.age = age;
    }

    public static String getAddr() {
        return addr;
    }

    public static void setAddr(String addr) {
        C1.addr = addr;
    }
}

class C2 {
    private static String name;
    private static int age;

    public static String getName() {
        return name;
    }

    public static void setName(String name) {
        C2.name = name;
    }

    public static int getAge() {
        return age;
    }

    public static void setAge(int age) {
        C2.age = age;
    }
}
