package to.kishimo.deeplearning.chap6.sec1;

public class Momentum implements Optimizer {
    private double lr;
    private double momentum;
    private double[] v;

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
    public void update(double[] params, double[] grads) {
        if (v == null)
            v = new double[params.length];

        for (int i = 0; i < params.length; i++) {
            v[i] = momentum * v[i] - lr * grads[i];
            params[i] += v[i];
        }
    }
}

