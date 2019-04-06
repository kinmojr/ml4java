package to.kishimo.deeplearning.common.optimizers;

import java.util.Map;

public class SGD implements Optimizer {
    private double lr;

    public SGD() {
        lr = 0.01;
    }

    public SGD(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(Map<String, double[][]> paramsMap, Map<String, double[][]> gradsMap) {
        for (String key : paramsMap.keySet()) {
            double[][] params = paramsMap.get(key);
            double[][] grads = gradsMap.get(key);
            for (int i = 0; i < params.length; i++) {
                for (int j = 0; j < params[0].length; j++) {
                    params[i][j] -= lr * grads[i][j];
                }
            }
        }
    }
}
