package to.kishimo.deeplearning.chap6.sec1;

public class SGD implements Optimizer {
    private double lr;

    public SGD() {
        lr = 0.01;
    }

    public SGD(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(double[] params, double[] grads) {
        for (int i = 0; i < params.length; i++) {
            params[i] -= lr * grads[i];
        }
    }
}

