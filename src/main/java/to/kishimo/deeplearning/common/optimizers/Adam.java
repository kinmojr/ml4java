package to.kishimo.deeplearning.common.optimizers;

import java.util.HashMap;
import java.util.Map;

public class Adam implements Optimizer {
    private double lr;
    private double beta1;
    private double beta2;
    private Map<String, Integer> iterMap = new HashMap<>();
    private Map<String, double[][]> mMap = new HashMap<String, double[][]>();
    private Map<String, double[][]> vMap = new HashMap<String, double[][]>();

    public Adam() {
        lr = 0.001;
        beta1 = 0.9;
        beta2 = 0.999;
    }

    public Adam(double lr, double beta1, double beta2) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    @Override
    public void update(Map<String, double[][]> paramsMap, Map<String, double[][]> gradsMap) {
        for (String key : paramsMap.keySet()) {
            double[][] params = paramsMap.get(key);
            double[][] grads = gradsMap.get(key);
            double[][] m;
            double[][] v;
            if (!mMap.containsKey(key)) {
                m = new double[params.length][params[0].length];
                mMap.put(key, m);
                v = new double[params.length][params[0].length];
                vMap.put(key, v);
            } else {
                m = mMap.get(key);
                v = vMap.get(key);
            }

            int iter = iterMap.get(key);
            iter++;
            iterMap.put(key, iter);
            double lrt = lr * Math.sqrt(1.0 - Math.pow(beta2, iter)) / (1.0 - Math.pow(beta1, iter));

            for (int i = 0; i < params.length; i++) {
                for (int j = 0; j < params[0].length; j++) {
                    m[i][j] += (1 - beta1) * (grads[i][j] - m[i][j]);
                    v[i][j] += (1 - beta2) * (Math.pow(grads[i][j], 2) - v[i][j]);
                    params[i][j] -= lrt * m[i][j] / (Math.sqrt(v[i][j]) + 1e-7);
                }
            }
        }
    }
}

