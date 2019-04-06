package to.kishimo.deeplearning.common;

import to.kishimo.deeplearning.common.layers.*;

import java.util.*;

public class MultiLayerNet implements Network {
    int inputSize;
    int[] hiddenSizeList;
    int outputSize;
    String activation;
    double weightDecayLambda;
    int hiddenLayerNum;
    Map<String, double[][]> params = new LinkedHashMap<>();
    Map<String, Layer> layers = new LinkedHashMap<>();
    LastLayer lastLayer;

    public MultiLayerNet(int inputSize, int[] hiddenSizeList, int outputSize, String activation, String weightInit, double weightDecayLambda) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenSizeList = hiddenSizeList;
        this.hiddenLayerNum = hiddenSizeList.length;
        this.weightDecayLambda = weightDecayLambda;

        initWeight(weightInit);

        int idx = 1;
        while (idx < hiddenLayerNum + 1) {
            layers.put("Affine" + idx, new Affine(params.get("W" + idx), params.get("b" + idx)));
            if ("sigmoid".equals(activation.toLowerCase())) {
                layers.put("Activation_function" + idx, new Sigmoid());
            } else if ("relu".equals(activation.toLowerCase())) {
                layers.put("Activation_function" + idx, new Relu());
            }
            idx++;
        }
        idx = hiddenLayerNum + 1;
        layers.put("Affine" + idx, new Affine(params.get("W" + idx), params.get("b" + idx)));

        lastLayer = new SoftmaxWithLoss();
    }

    private void initWeight(String weightInit) {
        int[] allSizeList = new int[hiddenLayerNum + 2];
        allSizeList[0] = inputSize;
        for (int i = 0; i < hiddenLayerNum; i++) {
            allSizeList[i + 1] = hiddenSizeList[i];
        }
        allSizeList[hiddenLayerNum + 1] = outputSize;
        for (int idx = 1; idx < allSizeList.length; idx++) {
            double scale;
            if ("relu".equals(weightInit.toLowerCase()) || "he".equals(weightInit.toLowerCase())) {
                scale = Math.sqrt(2.0 / allSizeList[idx - 1]);
            } else if ("sigmoid".equals(weightInit.toLowerCase()) || "xavier".equals(weightInit.toLowerCase())) {
                scale = Math.sqrt(1.0 / allSizeList[idx - 1]);
            } else {
                scale = Double.valueOf(weightInit);
            }
            params.put("W" + idx, Functions.randomMatrix(allSizeList[idx - 1], allSizeList[idx], scale));
            params.put("b" + idx, new double[1][allSizeList[idx]]);
        }
    }

    public double[][] predict(double[][] x) {
        for (Map.Entry<String, Layer> entry : layers.entrySet()) {
            x = entry.getValue().forward(x);
        }
        return x;
    }

    public double loss(double[][] x, int[][] t) {
        double[][] y = predict(x);

        double weightDecay = 0.0;
        for (int idx = 1; idx < hiddenLayerNum + 2; idx++) {
            double[][] W = params.get("W" + idx);
            weightDecay += 0.5 * weightDecayLambda * Functions.l2norm(W);
        }

        return lastLayer.forward(y, t) + weightDecay;
    }

    public double accuracy(double[][] x, int[][] t) {
        double[][] y = predict(x);
        y = Functions.argMax(y);
        int num = 0;
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < y[0].length; j++) {
                if (y[i][j] == 1 && t[i][j] == 1) {
                    num++;
                    break;
                } else if (y[i][j] == 1 || t[i][j] == 1) {
                    break;
                }
            }
        }
        return (double) num / (double) x.length;
    }

    public LinkedHashMap<String, double[][]> gradient(double[][] x, int[][] t) {
        loss(x, t);

        double[][] dout = Functions.makeMatrix(x.length, x[0].length, 1.0);
        dout = lastLayer.backward(dout);

        List<Layer> list = new ArrayList<>();
        for (Map.Entry<String, Layer> entry : layers.entrySet()) {
            list.add(entry.getValue());
        }
        Collections.reverse(list);
        for (Layer layer : list) {
            dout = layer.backward(dout);
        }

        LinkedHashMap<String, double[][]> grads = new LinkedHashMap<>();
        for (int idx = 1; idx < hiddenLayerNum + 2; idx++) {
            grads.put("W" + idx, Functions.calcWeights(layers.get("Affine" + idx).dW(), layers.get("Affine" + idx).W(), weightDecayLambda));
            grads.put("b" + idx, layers.get("Affine" + idx).db());
        }

        return grads;
    }

    public Map<String, double[][]> params() {return params;}
}
