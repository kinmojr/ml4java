package ch04;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;

import static common.Compat.*;
import static common.Compat.dot;
import static common.Functions.*;

public class TwoLayerNet {
    HashMap<String, RealMatrix> params = new HashMap<>();

    public TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd) {
        params.put("W1", randn(inputSize, hiddenSize, weightInitStd));
        params.put("b1", MatrixUtils.createRealMatrix(inputSize, hiddenSize));
        params.put("W2", randn(hiddenSize, outputSize, weightInitStd));
        params.put("b2", MatrixUtils.createRealMatrix(hiddenSize, outputSize));
    }

    public double[][] predict(RealMatrix x) {
        RealMatrix W1 = params.get("W1");
        RealMatrix W2 = params.get("W2");
        RealMatrix b1 = params.get("b1");
        RealMatrix b2 = params.get("b2");

        RealMatrix a1 = dot(x, W1).add(b1);
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = dot(z1, W2).add(b2);
        double[][] y = softmax(a2);

        return y;
    }

    public double loss(RealMatrix x, double[][] t) {
        double[][] y = predict(x);
        return crossEntropyError(y, t);
    }

    public double accuracy(RealMatrix x, double[][] t) {
        double[][] y = predict(x);
        int[] y2 = argmax(y);
        int[] t2 = argmax(t);

        int num = 0;
        for (int i = 0; i < y2.length; i++) {
            if (y2[i] == t2[i]) num++;
        }
        return (float) num / (float) y2.length;
    }

    public HashMap<String, RealMatrix> numericalGradient(RealMatrix x, double[][] t) {
        Function lossW = new Function() {
            @Override
            public double function() {
                return loss(x, t);
            }
        };
        HashMap<String, RealMatrix> grads = new HashMap<>();
//        grads.put("W1", numericalGradient(lossW, params.get("W1")));
//        grads.put("b1", numericalGradient(lossW, params.get("b1")));
//        grads.put("W2", numericalGradient(lossW, params.get("W2")));
//        grads.put("b2", numericalGradient(lossW, params.get("b2")));

        return grads;
    }

    public HashMap<String, RealMatrix> gradient(RealMatrix x, double[][] t) {
        RealMatrix W1 = params.get("W1");
        RealMatrix W2 = params.get("W2");
        RealMatrix b1 = params.get("b1");
        RealMatrix b2 = params.get("b2");
        HashMap<String, RealMatrix> grads = new HashMap<>();

        int batchNum = x.getRowDimension();

        RealMatrix a1 = dot(x, W1).add(b1);
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = dot(z1, W2).add(b2);
        double[][] y = softmax(a2);

        double[][] dy = new double[y.length][y[0].length];
        for (int i = 0; i < dy.length; i++) {
            for (int j = 0; j < dy[0].length; j++) {
                dy[i][j] = (y[i][j] - t[i][j]) / batchNum;
            }
        }
        grads.put("W2", dot(z1, MatrixUtils.createRealMatrix(dy)));
        grads.put("b2", MatrixUtils.createRealMatrix(sum(dy)));

        RealMatrix da1 = dot(MatrixUtils.createRealMatrix(dy), W2.transpose());
        RealMatrix dz1 = MatrixUtils.createRealMatrix(0, 0);// = sigmoidGrad(a1) * da1;
        grads.put("W1", dot(x.transpose(), dz1));
        grads.put("b1", MatrixUtils.createRealMatrix(sum(dz1.getData())));

        return grads;
    }

    interface Function {
        public double function();
    }
}
