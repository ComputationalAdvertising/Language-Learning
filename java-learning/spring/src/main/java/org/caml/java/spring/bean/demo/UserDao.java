package org.caml.java.spring.bean.demo;

/**
 * Created by zhouyong on 16/6/5.
 */
public class UserDao {

    private String dataSource;

    public String getDataSource() {
        return dataSource;
    }

    public void setDataSource(String dataSource) {
        this.dataSource = dataSource;
    }

    public void saveUser() {
        System.out.println("RoleDao.saveUser()");
    }
}
