package to.kishimo.deeplearning.chap5;

public class SigmoidLayer {
    double out;

    double forward(double x) {
        double out = 1 / (1 + Math.exp(-x));
        this.out = out;
        return out;
    }

    double backward(double dout) {
        double dx = dout * (1.0 - out) * out;
        return dx;
    }
}

