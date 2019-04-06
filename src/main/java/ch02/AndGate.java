package ch02;

import static common.Compat.dot;

public class AndGate {
    public static int and(int x1, int x2) {
        double[] x = {x1, x2};
        double[] w = {0.5, 0.5};
        double b = -0.7;
        double tmp = dot(x, w) + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    public static void main(String... args) {
        int[][] values = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
        for (int i = 0; i < values.length; i++) {
            int[] xs = values[i];
            int y = and(xs[0], xs[1]);
            System.out.println("(" + xs[0] + ", " + xs[1] + ") -> " + y);
        }
    }

}
