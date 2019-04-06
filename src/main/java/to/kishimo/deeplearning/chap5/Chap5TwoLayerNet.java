package to.kishimo.deeplearning.chap5;

import to.kishimo.deeplearning.chap3.Chap3MnistDataSet;

import java.util.*;

public class Chap5TwoLayerNet {
    private int inputSize, hiddenSize, outputSize;
    private double weightInitStd, learnRate;
    private double[][] paramW1, paramW2, gradW1, gradW2;
    private double[] paramB1, paramB2, gradB1, gradB2;
    private LinkedHashMap<String, HiddenLayerFunction> layers;
    private LastLayerFunction lastLayer;

    public Chap5TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd, double learnRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightInitStd = weightInitStd;
        this.learnRate = learnRate;

        Random rand = new Random();
        paramW1 = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                paramW1[i][j] = weightInitStd * rand.nextGaussian();
            }
        }

        paramB1 = new double[hiddenSize];

        paramW2 = new double[hiddenSize][outputSize];
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                paramW2[i][j] = weightInitStd * rand.nextGaussian();
            }
        }

        paramB2 = new double[outputSize];

        layers = new LinkedHashMap<>();
        layers.put("Affine1", new AffineLayer(paramW1, paramB1));
        layers.put("Relu1", new ReluLayer());
        layers.put("Affine2", new AffineLayer(paramW2, paramB2));

        lastLayer = new SoftmaxWithLossLayer();
    }

    /**
     * 予測.
     *
     * @param x
     * @return
     */
    private double[][] predict(double[][] x) {
        for (Map.Entry<String, HiddenLayerFunction> entry : layers.entrySet()) {
            HiddenLayerFunction layer = entry.getValue();
            x = layer.forward(x);
        }
        return x;
    }

    /**
     * 損失関数.
     *
     * @param x
     * @param t
     * @return
     */
    private double loss(double[][] x, int[][] t) {
        double[][] y = predict(x);
        return lastLayer.forward(y, t);
    }

    private void gradient(double[][] x, int[][] t) {
        loss(x, t);
        double[][] dout = new double[t.length][t[0].length];
        for (int i = 0; i < dout.length; i++) {
            for (int j = 0; j < dout[0].length; j++) {
                dout[i][j] = 1.0;
            }
        }
        dout = lastLayer.backward(dout);

        List<HiddenLayerFunction> list = new ArrayList<>();
        for (Map.Entry<String, HiddenLayerFunction> entry : layers.entrySet()) {
            list.add(entry.getValue());
        }
        Collections.reverse(list);
        for (HiddenLayerFunction layer : list) {
            dout = layer.backward(dout);
        }

        updateWeights(paramW1, layers.get("Affine1").getGradWeights());
        updateBias(paramB1, layers.get("Affine1").getGradBias());
        updateWeights(paramW2, layers.get("Affine2").getGradWeights());
        updateBias(paramB2, layers.get("Affine2").getGradBias());
    }

    private void updateWeights(double[][] paramW, double[][] gradW) {
        for (int i = 0; i < paramW.length; i++) {
            for (int j = 0; j < paramW[0].length; j++) {
                if (gradW[i][j] != 0.0) {
                    paramW[i][j] += -learnRate * gradW[i][j];
                }
            }
        }
    }

    private void updateBias(double[] paramB, double[] gradB) {
        for (int i = 0; i < paramB.length; i++) {
            if (gradB[i] != 0.0) {
                paramB[i] += -learnRate * gradB[i];
            }
        }
    }

    private double accuracy(Chap3MnistDataSet dataSet) {
        double[][] features = dataSet.getFeatures();
        int[] labels = dataSet.getLabels();
        double[][] y = predict(features);
        int count = 0;
        for (int i = 0; i < labels.length; i++) {
            if (getMaxIndex(y[i]) == labels[i]) {
                count++;
            }
        }
        return (double) count / (double) features.length;
    }

    private static int getMaxIndex(double[] values) {
        double maxValue = Double.NaN;
        int maxIndex = 0;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(maxValue)) {
                maxValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void main(String... args) throws Exception {
        Chap3MnistDataSet trainDataSet = Chap3MnistDataSet.createInstance("train");
        Chap3MnistDataSet testDataSet = Chap3MnistDataSet.createInstance("test");
        int itersNum = 10000;
        int batchSize = 100;
        int iterPerEpoch = trainDataSet.getNumImages() / batchSize;

        Chap5TwoLayerNet network = new Chap5TwoLayerNet(784, 50, 10, 0.01, 0.1);
        System.out.println("Index\tTime(sec)\tAccuracy(train)\tAccuracy(test)\tLoss(train)\tLoss(test)");
        long start = System.currentTimeMillis();
        for (int i = 0; i < itersNum; i++) {
            int[] indexes = trainDataSet.getRandomIndexes(batchSize);
            double[][] batchX = trainDataSet.getRandomFeatures(indexes);
            int[][] batchT = trainDataSet.getRandomHotEncodinLabels(indexes);
            network.gradient(batchX, batchT);
            if (i % iterPerEpoch == 0) {
                System.out.print(i + "\t" + (System.currentTimeMillis() - start) / 1000 + "\t");
                System.out.print(network.accuracy(trainDataSet) + "\t");
                System.out.print(network.accuracy(testDataSet) + "\t");
                System.out.print(network.loss(testDataSet.getFeatures(), testDataSet.getHotEncodingLabels()) + "\t");
                System.out.println(network.loss(batchX, batchT));
            }
        }
    }
}
