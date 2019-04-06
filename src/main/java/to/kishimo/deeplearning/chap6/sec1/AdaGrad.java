package to.kishimo.deeplearning.chap6.sec1;

public class AdaGrad implements Optimizer {
    private double lr;
    private double[] h;

    public AdaGrad() {
        lr = 0.01;
    }

    public AdaGrad(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(double[] params, double[] grads) {
        if (h == null)
            h = new double[params.length];

        for (int i = 0; i < params.length; i++) {
            h[i] += grads[i] * grads[i];
            params[i] -= lr * grads[i] / (Math.sqrt(h[i]) + 1e-7);
        }
    }
}
