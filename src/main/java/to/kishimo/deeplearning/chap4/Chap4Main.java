package to.kishimo.deeplearning.chap4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

public class Chap4Main {
    public static void main(String... args) throws Exception {
        // # 4.2 損失関数
        double[] t = {0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        double[] y1 = {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
        System.out.println(meanSquaredError(y1, t));

        double[] y = new double[]{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
        System.out.println(meanSquaredError(y, t));

        y = new double[]{0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0};
        System.out.println(crossEntropyError(y, t));

        y = new double[]{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
        System.out.println(crossEntropyError(y, t));

        double[] y2 = new double[]{0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0};
        System.out.println(crossEntropyErrorMean(new double[][]{y1, y2}, new double[][]{t, t}));
        System.out.println(crossEntropyErrorMean(new double[][]{y1, y2}, new int[]{2, 2}));

        // # 4.3 数値微分
        // 1変数関数 y=0.01x^2+0.1xの微分
        double value = numericalDiff(new Function() {
            public double function(double[] values) {
                return 0.01 * Math.pow(values[0], 2.0) + 0.1 * values[0];
            }
        }, new double[]{5.0});
        System.out.println(value);
        value = numericalDiff(new Function() {
            public double function(double[] values) {
                return 0.01 * Math.pow(values[0], 2.0) + 0.1 * values[0];
            }
        }, new double[]{10.0});
        System.out.println(value);

        // 2変数関数 f(x0, x1)=x0^2+x1^2、x0=3、x1=4、x0に対する偏微分
        value = numericalDiff(new Function() {
            public double function(double[] values) {
                return Math.pow(values[0], 2.0) + Math.pow(4, 2.0);
            }
        }, new double[]{3, 4});
        System.out.println(value);

        // 2変数関数 f(x0, x1)=x0^2+x1^2、x0=3、x1=4、x1に対する偏微分
        value = numericalDiff(new Function() {
            public double function(double[] values) {
                return Math.pow(3, 2.0) + Math.pow(values[1], 2.0);
            }
        }, new double[]{3, 4});
        System.out.println(value);

        // # 4.4 勾配
        // 2変数関数 f(x0, x1)=x0^2+x1^2、(x0=3, x1=4)に対する勾配
        double[] values = numericalGradient(new Function() {
            public double function(double[] values) {
                return Math.pow(values[0], 2.0) + Math.pow(values[1], 2.0);
            }
        }, new double[]{3, 4});
        for (double val : values) {
            System.out.println(val);
        }
        // 2変数関数 f(x0, x1)=x0^2+x1^2、(x0=0, x1=2)に対する勾配
        values = numericalGradient(new Function() {
            public double function(double[] values) {
                return Math.pow(values[0], 2.0) + Math.pow(values[1], 2.0);
            }
        }, new double[]{0, 2});
        for (double val : values) {
            System.out.println(val);
        }
        // 2変数関数 f(x0, x1)=x0^2+x1^2、(x0=3, x1=0)に対する勾配
        values = numericalGradient(new Function() {
            public double function(double[] values) {
                return Math.pow(values[0], 2.0) + Math.pow(values[1], 2.0);
            }
        }, new double[]{3, 0});
        for (double val : values) {
            System.out.println(val);
        }

        // 2変数関数 f(x0, x1)=x0^2+x1^2の最小値
        values = gradientDescent(new Function() {
            public double function(double[] values) {
                return Math.pow(values[0], 2.0) + Math.pow(values[1], 2.0);
            }
        }, new double[]{-3, 4}, 0.1, 100);
        for (double val : values) {
            System.out.println(val);
        }

        // # 4.4.2 ニューラルネットワークに対する勾配
        System.out.println("# 4.4.2 ニューラルネットワークに対する勾配");
        Chap4SimpleNet net = new Chap4SimpleNet();
        double[][] x = new double[][]{{0.6, 0.9}};
        double[][] p = net.predict(x);
        print(p);
        System.out.println(net.argMax(p[0]));
        t = new double[]{0, 0, 1};
        double loss = net.loss(x, t);
        System.out.println(loss);

        double[][] grad = numericalGradient(new Function() {
            public double function(double[] values) {
                return net.loss(x, new double[]{0, 0, 1});
            }
        }, net.w);
        print(grad);
    }

    private static void print(double[][] matrix) {
        for (double[] row : matrix) {
            for (double value : row) {
                System.out.print(value + ", ");
            }
            System.out.println();
        }
    }

    /**
     * 平均2乗和誤差.
     *
     * @param y
     * @param t
     * @return
     */
    public static double meanSquaredError(double[] y, double[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += Math.pow(y[i] - t[i], 2.0);
        }
        return 0.5 * sum;
    }

    /**
     * クロスエントロピー誤差.
     *
     * @param y
     * @param t
     * @return
     */
    public static double crossEntropyError(double[] y, double[] t) {
        for (int i = 0; i < y.length; i++) {
            if (t[i] == 1.0)
                return -Math.log(y[i] + Math.pow(10.0, -7.0));
        }
        return 0.0;
    }

    /**
     * クロスエントロピー誤差.
     *
     * @param y
     * @param t
     * @return
     */
    public static double crossEntropyError(double[] y, int[] t) {
        for (int i = 0; i < y.length; i++) {
            if (t[i] == 1)
                return -Math.log(y[i] + Math.pow(10.0, -7.0));
        }
        return 0.0;
    }

    /**
     * クロスエントロピー誤差.
     *
     * @param y
     * @param t
     * @return
     */
    public static double crossEntropyErrorMean(double[][] y, double[][] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += crossEntropyError(y[i], t[i]);
        }
        return sum / y.length;
    }

    /**
     * クロスエントロピー誤差.
     *
     * @param y
     * @param t
     * @return
     */
    public static double crossEntropyErrorMean(double[][] y, int[][] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += crossEntropyError(y[i], t[i]);
        }
        return sum / y.length;
    }

    public static double crossEntropyErrorMean(double[][] y, int[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += -Math.log(y[i][t[i]] + Math.pow(10.0, -7.0));
        }
        return sum / y.length;
    }

    /**
     * 数値微分.
     *
     * @param func
     * @param values
     * @return
     */
    public static double numericalDiff(Function func, double[] values) {
        double diff = Math.pow(10, -4);

        double[] values1 = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            values1[i] = values[i] + diff;
        }

        double[] values2 = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            values2[i] = values[i] - diff;
        }

        return (func.function(values1) - func.function(values2)) / (2 * diff);
    }

    /**
     * 数値微分による勾配.
     *
     * @param func
     * @param weights
     * @return
     */
    public static double[] numericalGradient(Function func, double[] weights) {
        double diff = Math.pow(10, -4);
        double[] ret = new double[weights.length];
        for (int i = 0; i < weights.length; i++) {
            double tmp = weights[i];

            weights[i] = tmp + diff;
            double value1 = func.function(weights);

            weights[i] = tmp - diff;
            double value2 = func.function(weights);

            ret[i] = (value1 - value2) / (2 * diff);
            weights[i] = tmp;
        }
        return ret;
    }

    /**
     * 数値微分による勾配.
     *
     * @param func
     * @param weights
     * @return
     */
    public static double[][] numericalGradient(Function func, double[][] weights) throws InterruptedException, TimeoutException, ExecutionException {
        double[][] ret = new double[weights.length][weights[0].length];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = numericalGradient(func, weights[i]);
        }
        return ret;

        //        double[][] ret = new double[weights.length][weights[0].length];
//        ExecutorService exec = Executors.newFixedThreadPool(8);
//        Future<double[]>[] futures = new Future[ret.length];
//        for (int i = 0; i < ret.length; i++) {
//            double[] value = weights[i];
//            futures[i] = exec.submit(new Callable<double[]>() {
//                @Override
//                public double[] call() throws Exception {
//                    double[] ds = numericalGradient(func, value);
//                    return ds;
//                }
//            });
//        }
//        for (int i = 0; i < ret.length; i++) {
//            ret[i] = futures[i].get(300, TimeUnit.SECONDS);
//        }
//        exec.shutdown();
//
//        return ret;
    }

    /**
     * 勾配降下法.
     *
     * @param func
     * @param initX
     * @param lr
     * @param stepNum
     * @return
     */
    public static double[] gradientDescent(Function func, double[] initX, double lr, int stepNum) {
//        Chap4TwoLayerNet.callCount.put("gradientDescent", Chap4TwoLayerNet.callCount.get("gradientDescent") + 1);
        double[] x = initX;
        for (int i = 0; i < stepNum; i++) {
            double[] grad = numericalGradient(func, x);
            for (int j = 0; j < x.length; j++) {
                x[j] -= lr * grad[j];
            }
        }
        return x;
    }
}
