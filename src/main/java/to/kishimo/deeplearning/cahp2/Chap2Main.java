package to.kishimo.deeplearning.cahp2;

public class Chap2Main {
    public static void main(String... args) {
        // # 2.3 パーセプトロンの実装
        System.out.println(dot(new double[]{0.0, 1.0}, new double[]{0.5, 0.5}));
        double b = -0.7;
        System.out.println(dot(new double[]{0.0, 1.0}, new double[]{0.5, 0.5}) + b);
        System.out.println();

        System.out.println(and(0.0, 0.0));
        System.out.println(and(1.0, 0.0));
        System.out.println(and(0.0, 1.0));
        System.out.println(and(1.0, 1.0));
        System.out.println();

        System.out.println(nand(0.0, 0.0));
        System.out.println(nand(1.0, 0.0));
        System.out.println(nand(0.0, 1.0));
        System.out.println(nand(1.0, 1.0));
        System.out.println();

        System.out.println(or(0.0, 0.0));
        System.out.println(or(1.0, 0.0));
        System.out.println(or(0.0, 1.0));
        System.out.println(or(1.0, 1.0));
        System.out.println();

        // # 2.5 多層パーセプトロン
        System.out.println(xor(0.0, 0.0));
        System.out.println(xor(1.0, 0.0));
        System.out.println(xor(0.0, 1.0));
        System.out.println(xor(1.0, 1.0));
        System.out.println();
    }

    private static double and(double x1, double x2) {
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

    private static double nand(double x1, double x2) {
        double[] x = {x1, x2};
        double[] w = {-0.5, -0.5};
        double b = 0.7;
        double tmp = dot(x, w) + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    private static double or(double x1, double x2) {
        double[] x = {x1, x2};
        double[] w = {0.5, 0.5};
        double b = -0.2;
        double tmp = dot(x, w) + b;
        if (tmp <= 0) {
            return 0;
        } else {
            return 1;
        }
    }

    private static double xor(double x1, double x2) {
        double s1 = nand(x1, x2);
        double s2 = or(x1, x2);
        double y = and(s1, s2);
        return y;
    }

    private static double dot(double[] x, double[] w) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += w[i] * x[i];
        }
        return sum;
    }
}

