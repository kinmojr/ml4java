package to.kishimo.deeplearning.common.optimizers;

import java.util.HashMap;
import java.util.Map;

public class Momentum implements Optimizer {
    private double lr;
    private double momentum;
    private Map<String, double[][]> vMap = new HashMap<>();

    public Momentum() {
        lr = 0.01;
        momentum = 0.9;
    }

    public Momentum(double lr) {
        this.lr = lr;
        this.momentum = 0.9;
    }

    public Momentum(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
    }

    @Override
    public void update(Map<String, double[][]> paramsMap, Map<String, double[][]> gradsMap) {
        for (String key : paramsMap.keySet()) {
            double[][] params = paramsMap.get(key);
            double[][] grads = gradsMap.get(key);
            double[][] v;
            if (!vMap.containsKey(key)) {
                v = new double[params.length][params[0].length];
                vMap.put(key, v);
            } else {
                v = vMap.get(key);
            }

            for (int i = 0; i < params.length; i++) {
                for (int j = 0; j < params[0].length; j++) {
                    v[i][j] = momentum * v[i][j] - lr * grads[i][j];
                    params[i][j] += v[i][j];
                }
            }
        }
    }
}
