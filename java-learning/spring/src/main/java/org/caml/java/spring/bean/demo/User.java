package org.caml.java.spring.bean.demo;

import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.List;

/**
 * Created by zhouyong on 16/6/5.
 */
public class User {

    private Long id;
    private String name = "ZY";
    private boolean gender;

    private Set<String> addressSet;
    private Set numberSet;

    private List<String> addressList;
    private String[] addressArray;
    private Map<String, String> addressMap;
    private Properties properties;

    public User() {
    }

    public User(Long id, String name) {
        this.id = id;
        this.name = name;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isGender() {
        return gender;
    }

    public void setGender(boolean gender) {
        this.gender = gender;
    }

    public Set<String> getAddressSet() {
        return addressSet;
    }

    public void setAddressSet(Set<String> addressSet) {
        this.addressSet = addressSet;
    }

    public Set getNumberSet() {
        return numberSet;
    }

    public void setNumberSet(Set numberSet) {
        this.numberSet = numberSet;
    }

    public List<String> getAddressList() {
        return addressList;
    }

    public void setAddressList(List<String> addressList) {
        this.addressList = addressList;
    }

    public String[] getAddressArray() {
        return addressArray;
    }

    public void setAddressArray(String[] addressArray) {
        this.addressArray = addressArray;
    }

    public Map<String, String> getAddressMap() {
        return addressMap;
    }

    public void setAddressMap(Map<String, String> addressMap) {
        this.addressMap = addressMap;
    }

    public Properties getProperties() {
        return properties;
    }

    public void setProperties(Properties properties) {
        this.properties = properties;
    }

    @Override
    public String toString() {
        return "[User: id=" + id + ", name=" + name + ",gender=" + gender + "]";
    }
}
