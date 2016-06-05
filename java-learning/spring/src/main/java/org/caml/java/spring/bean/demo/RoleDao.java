package org.caml.java.spring.bean.demo;

/**
 * Created by zhouyong on 16/6/5.
 */
public class RoleDao {

    private String dataSource;

    public String getDataSource() {
        return dataSource;
    }

    public void setDataSource(String dataSource) {
        this.dataSource = dataSource;
    }

    public void saveRole() {
        System.out.println("RoleDao.saveRole()");
    }
}
