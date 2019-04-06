package to.kishimo.deeplearning.chap6.sec2;

import to.kishimo.deeplearning.chap3.Chap3MnistDataSet;
import to.kishimo.deeplearning.common.MultiLayerNet;
import to.kishimo.deeplearning.common.Network;
import to.kishimo.deeplearning.common.optimizers.Optimizer;
import to.kishimo.deeplearning.common.optimizers.SGD;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class WeightInitCompare {
    public static void main(String... args) throws IOException, ClassNotFoundException {
        Chap3MnistDataSet trainDataSet = Chap3MnistDataSet.createInstance("train");
        double[][] xTrain = trainDataSet.getFeatures();
        int[][] tTrain = trainDataSet.getHotEncodingLabels();

        Chap3MnistDataSet testDataSet = Chap3MnistDataSet.createInstance("test");
        double[][] xTest = testDataSet.getFeatures();
        int[][] tTest = testDataSet.getHotEncodingLabels();

        int trainSize = xTrain.length;
        int batchSize = 128;
        int maxIteration = 2000;

        Map<String, String> weightInitTypes = new LinkedHashMap<>();
        weightInitTypes.put("std=0.01", "0.01");
        weightInitTypes.put("Xavier", "sigmoid");
        weightInitTypes.put("He", "relu");
        Optimizer optimizer = new SGD(0.01);

        Map<String, Network> networks = new LinkedHashMap<>();
        Map<String, Double> trainLoss = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : weightInitTypes.entrySet()) {
            String key = entry.getKey();
            String weightType = entry.getValue();
            Network network = new MultiLayerNet(784, new int[]{100, 100, 100, 100}, 10, "relu", weightType, 0.0);
            networks.put(key, network);
        }

        for (int i = 0; i < maxIteration; i++) {
            int[] batchMask = trainDataSet.getRandomIndexes(batchSize);
            double[][] xBatch = trainDataSet.getRandomFeatures(batchMask);
            int[][] tBatch = trainDataSet.getRandomHotEncodinLabels(batchMask);

            for (String key : weightInitTypes.keySet()) {
                LinkedHashMap<String, double[][]> grads = networks.get(key).gradient(xBatch, tBatch);
                optimizer.update(networks.get(key).params(), grads);

                double loss = networks.get(key).loss(xBatch, tBatch);
                trainLoss.put(key, loss);
            }

            if (i % 100 == 0) {
                System.out.println("===========" + "iteration:" + i + "===========");
                for (String key : weightInitTypes.keySet()) {
                    double loss = networks.get(key).loss(xBatch, tBatch);
                    System.out.println(key + ":" + loss);
                }
            }
        }
    }
}

