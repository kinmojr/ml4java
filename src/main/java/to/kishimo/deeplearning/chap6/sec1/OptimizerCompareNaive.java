package to.kishimo.deeplearning.chap6.sec1;

import java.util.LinkedHashMap;
import java.util.Map;

public class OptimizerCompareNaive {
    public static void main(String... args) {
        double[] initPos = new double[]{-7.0, 2.0};

        Map<String, Optimizer> optimizers = new LinkedHashMap<>();
        optimizers.put("SGD", new SGD(0.95));
        optimizers.put("Momentum", new Momentum(0.1));
        optimizers.put("AdaGrad", new AdaGrad(1.5));
        optimizers.put("Adam", new Adam(0.3, 0.9, 0.999));

        for (Map.Entry<String, Optimizer> e : optimizers.entrySet()) {
            Optimizer optimizer = e.getValue();
            double[] params = new double[]{initPos[0], initPos[1]};
            double[] grads = new double[]{0.0, 0.0};

            for (int i = 0; i < 30; i++) {
                grads = df(params[0], params[1]);
                optimizer.update(params, grads);
            }
            System.out.println(e.getKey() + "\t" + params[0] + "\t" + params[1]);
        }
    }

    public static double f(double x, double y) {
        return Math.pow(x, 2) / 20 + Math.pow(y, 2);
    }

    public static double[] df(double x, double y) {
        return new double[]{x / 10, 2 * y};
    }
}

