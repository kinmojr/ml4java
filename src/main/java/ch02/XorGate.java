package ch02;

import static ch02.AndGate.and;
import static ch02.NandGate.nand;
import static ch02.OrGate.or;

public class XorGate {
    public static int xor(int x1, int x2) {
        int s1 = nand(x1, x2);
        int s2 = or(x1, x2);
        int y = and(s1, s2);
        return y;
    }

    public static void main(String... args) {
        int[][] values = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
        for (int i = 0; i < values.length; i++) {
            int[] xs = values[i];
            int y = xor(xs[0], xs[1]);
            System.out.println("(" + xs[0] + ", " + xs[1] + ") -> " + y);
        }
    }

}
