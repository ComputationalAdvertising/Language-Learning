package org.caml.java.spring.bean.demo;

/**
 * Created by zhouyong on 16/6/5.
 */
public class Student {

    private int score;
    private User user;

    public Student(int score, User user) {
        this.score = score;
        this.user = user;
    }

    public User getUser() {
        return user;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    @Override
    public String toString() {
        return "[Student: id=" + user.getId() + ", name=" + user.getName() + ", score=" + score;
    }
}
