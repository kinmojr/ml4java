package to.kishimo.deeplearning.common.optimizers;

import java.util.HashMap;
import java.util.Map;

public class AdaGrad implements Optimizer {
    private double lr;
    private Map<String, double[][]> hMap = new HashMap<>();

    public AdaGrad() {
        lr = 0.01;
    }

    public AdaGrad(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(Map<String, double[][]> paramsMap, Map<String, double[][]> gradsMap) {
        for (String key : paramsMap.keySet()) {
            double[][] params = paramsMap.get(key);
            double[][] grads = gradsMap.get(key);
            double[][] h;
            if (!hMap.containsKey(key)) {
                h = new double[params.length][params[0].length];
                hMap.put(key, h);
            } else {
                h = hMap.get(key);
            }

            for (int i = 0; i < params.length; i++) {
                for (int j = 0; j < params[0].length; j++) {
                    h[i][j] += grads[i][j] * grads[i][j];
                    params[i][j] -= lr * grads[i][j] / (Math.sqrt(h[i][j]) + 1e-7);
                }
            }
        }
    }
}

