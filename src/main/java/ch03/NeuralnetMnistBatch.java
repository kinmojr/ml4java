package ch03;

import dataset.Mnist;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;
import java.util.HashMap;

import static common.Compat.*;
import static common.Functions.sigmoid;
import static common.Functions.softmax;

public class NeuralnetMnistBatch {
    private RealMatrix w1, w2, w3, b1, b2, b3;

    private HashMap<String, double[][]> getData() throws IOException, ClassNotFoundException {
        return new Mnist().loadMinist(true, true, false);
    }

    private void initNetwork(int batchSize) throws IOException, ClassNotFoundException {
        w1 = readWeights("mnist/w1.tsv");
        w2 = readWeights("mnist/w2.tsv");
        w3 = readWeights("mnist/w3.tsv");
        b1 = readBatchedBias("mnist/b1.tsv", batchSize);
        b2 = readBatchedBias("mnist/b2.tsv", batchSize);
        b3 = readBatchedBias("mnist/b3.tsv", batchSize);
    }

    private double[][] predict(NeuralnetMnistBatch network, RealMatrix x) {
        RealMatrix w1 = network.w1;
        RealMatrix w2 = network.w2;
        RealMatrix w3 = network.w3;
        RealMatrix b1 = network.b1;
        RealMatrix b2 = network.b2;
        RealMatrix b3 = network.b3;

        RealMatrix a1 = x.multiply(w1).add(b1);
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = z1.multiply(w2).add(b2);
        RealMatrix z2 = sigmoid(a2);
        RealMatrix a3 = z2.multiply(w3).add(b3);
        double[][] y = softmax(a3.getData());

        return y;
    }

    public static void main(String... args) throws IOException, ClassNotFoundException {
        int batchSize = 100;
        NeuralnetMnistBatch network = new NeuralnetMnistBatch();
        network.initNetwork(batchSize);
        HashMap<String, double[][]> dataset = network.getData();
        RealMatrix x = MatrixUtils.createRealMatrix(dataset.get("test_img"));
        double[][] t = dataset.get("test_label");
        int accuracyCnt = 0;
        for (int i = 0; i < x.getRowDimension(); i += batchSize) {
            RealMatrix xBatch = x.getSubMatrix(i, i + batchSize - 1, 0, x.getColumnDimension() - 1);
            double[][] yBatch = network.predict(network, xBatch);
            int[] p = argmax(yBatch);
            for (int j = 0; j < p.length; j++) {
                if (p[j] == t[i + j][0]) {
                    accuracyCnt++;
                }
            }
        }

        System.out.println("Accuracy:" + (float) accuracyCnt / (float) x.getRowDimension());
    }
}

