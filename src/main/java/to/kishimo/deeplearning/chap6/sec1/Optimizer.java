package to.kishimo.deeplearning.chap6.sec1;

public interface Optimizer {
    public void update(double[] params, double[] grads);
}

