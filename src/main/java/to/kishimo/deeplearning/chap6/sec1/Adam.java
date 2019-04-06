package to.kishimo.deeplearning.chap6.sec1;

public class Adam implements Optimizer {
    private double lr;
    private double beta1;
    private double beta2;
    private int iter = 0;
    private double[] m;
    private double[] v;

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
    public void update(double[] params, double[] grads) {
        if (m == null) {
            m = new double[params.length];
            v = new double[params.length];
        }

        iter++;
        double lrt = lr * Math.sqrt(1.0 - Math.pow(beta2, iter)) / (1.0 - Math.pow(beta1, iter));

        for (int i = 0; i < params.length; i++) {
            m[i] += (1 - beta1) * (grads[i] - m[i]);
            v[i] += (1 - beta2) * (Math.pow(grads[i], 2) - v[i]);
            params[i] -= lrt * m[i] / (Math.sqrt(v[i]) + 1e-7);
        }
    }
}

