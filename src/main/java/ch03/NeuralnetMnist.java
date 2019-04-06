package ch03;

import dataset.Mnist;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static common.Compat.*;
import static common.Functions.*;

import java.io.IOException;
import java.util.HashMap;

public class NeuralnetMnist {
    private RealMatrix W1, W2, W3, b1, b2, b3;

    private HashMap<String, double[][]> getData() throws IOException, ClassNotFoundException {
        return new Mnist().loadMinist(true, true, false);
    }

    private void initNetwork() throws IOException, ClassNotFoundException {
        W1 = readWeights("mnist/w1.tsv");
        W2 = readWeights("mnist/w2.tsv");
        W3 = readWeights("mnist/w3.tsv");
        b1 = readWeights("mnist/b1.tsv"); ;
        b2 = readWeights("mnist/b2.tsv"); ;
        b3 = readWeights("mnist/b3.tsv"); ;
    }

    private double[] predict(NeuralnetMnist network, RealMatrix x) {
        RealMatrix W1 = network.W1;
        RealMatrix W2 = network.W2;
        RealMatrix W3 = network.W3;
        RealMatrix b1 = network.b1;
        RealMatrix b2 = network.b2;
        RealMatrix b3 = network.b3;

        RealMatrix a1 = x.multiply(W1).add(b1);
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = z1.multiply(W2).add(b2);
        RealMatrix z2 = sigmoid(a2);
        RealMatrix a3 = z2.multiply(W3).add(b3);
        double[] y = softmax(a3.getRow(0));

        return y;
    }

    public static void main(String... args) throws IOException, ClassNotFoundException {
        NeuralnetMnist network = new NeuralnetMnist();
        network.initNetwork();
        HashMap<String, double[][]> dataset = network.getData();
        RealMatrix x = MatrixUtils.createRealMatrix(dataset.get("test_img"));
        double[][] t = dataset.get("test_label");
        int accuracyCnt = 0;
        for (int i = 0; i < x.getRowDimension(); i++) {
            double[] y = network.predict(network, x.getRowMatrix(i));
            int p = argmax(y);
            if (p == t[i][0]) {
                accuracyCnt++;
            }
        }

        System.out.println("Accuracy:" + (float) accuracyCnt / (float) x.getRowDimension());
    }
}
