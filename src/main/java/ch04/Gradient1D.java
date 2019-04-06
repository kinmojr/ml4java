package ch04;

import static common.Compat.arrange;

public class Gradient1D {
    private static double numericalDiff(Function1 f, double x) {
        double h = Math.pow(10, -4);
        return (f.function(x + h) - f.function(x - h)) / (2 * h);
    }

    private static double function1(double x) {
        return 0.01 * Math.pow(x, 2.0) + 0.1 * x;
    }

    private static double tangentLine(Function1 f, double x, double t) {
        double d = numericalDiff(f, x);
        double y = f.function(x) - d * x;
        return d * t + y;
    }

    public static void main(String... args) {
        double[] x = arrange(0.0, 20.0, 0.1);
        double[] y = new double[x.length];
        Function1 function1 = new Function1() {
            @Override
            public double function(double x) {
                return function1(x);
            }
        };
        for (int i = 0; i < x.length; i++) {
            y[i] = function1.function(x[i]);
        }

        Function12 tf = new Function12() {
            @Override
            public double function(Function1 f, double x, double t) {
                return tangentLine(f, x, t);
            }
        };
        double[] y2 = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y2[i] = tf.function(function1, 5, x[i]);
        }

        for (int i = 0; i < x.length; i++) {
            System.out.println("x=" + x[i] + ", y=" + y[i] + ", y2=" + y2[i]);
        }
    }

    private interface Function1 {
        public double function(double x);
    }

    private interface Function12 {
        public double function(Function1 f1, double x, double t);
    }
}

