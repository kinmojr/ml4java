package to.kishimo.deeplearning.common.optimizers;

import java.util.Map;

public interface Optimizer {
    public void update(Map<String, double[][]> params, Map<String, double[][]> grads);
}

