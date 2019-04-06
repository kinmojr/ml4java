package ch01;

public class Man {
    private String name;

    public Man(String name) {
        this.name = name;
        System.out.println("Initialized!");
    }

    private void hello() {
        System.out.println("Hello " + name + "!");
    }

    private void goodbye() {
        System.out.println("Good-bye " + name + "!");
    }

    public static void main(String... args) {
        Man m = new Man("David");
        m.hello();
        m.goodbye();
    }
}
