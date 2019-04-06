package dataset;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

public class Mnist {
    private static final String URL_BASE = "http://yann.lecun.com/exdb/mnist/";
    private static final HashMap<String, String> KEY_FILE = new HashMap<>();
    private static final String DATASET_DIR = "./dataset/mnist";
    private static final String SAVE_FILE = DATASET_DIR + "/mnist.ser";
    private static HashMap<String, double[][]> dataset;

    static {
        KEY_FILE.put("train_img", "train-images-idx3-ubyte.gz");
        KEY_FILE.put("train_label", "train-labels-idx1-ubyte.gz");
        KEY_FILE.put("test_img", "t10k-images-idx3-ubyte.gz");
        KEY_FILE.put("test_label", "t10k-labels-idx1-ubyte.gz");
    }

    private void _download(String fileName) throws IOException {
        String filePath = DATASET_DIR + "/" + fileName;

        if (new File(filePath).exists()) return;

        System.out.println("Downloading " + fileName + " ... ");

        URL url = new URL(URL_BASE + fileName);
        URLConnection conn = url.openConnection();
        File file = new File(filePath);
        try (InputStream in = conn.getInputStream();
             FileOutputStream out = new FileOutputStream(file, false);) {
            byte[] data = new byte[1024];
            while (true) {
                int ret = in.read(data);
                if (ret == -1) {
                    break;
                }
                out.write(data, 0, ret);
            }
        }
        System.out.println("Done");
    }

    public void downloadMnist() throws IOException {
        for (String v : KEY_FILE.values()) {
            _download(v);
        }
    }

    private double[][] _loadLabel(String fileName) throws IOException {
        double[][] data;
        String filePath = DATASET_DIR + "/" + fileName;
        try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath)));) {
            is.readInt();
            int numRow = is.readInt();
            data = new double[numRow][1];
            for (int i = 0; i < numRow; i++) {
                data[i][0] = (double) is.readUnsignedByte();
            }
        }
        return data;
    }

    private double[][] _loadImg(String fileName) throws IOException {
        double[][] data;
        String filePath = DATASET_DIR + "/" + fileName;
        try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath)));) {
            is.readInt();
            int numRow = is.readInt();
            int numCol = is.readInt() * is.readInt();
            data = new double[numRow][numCol];
            for (int i = 0; i < numRow; i++) {
                for (int j = 0; j < numCol; j++) {
                    data[i][j] = (double) is.readUnsignedByte();
                }
            }
        }
        return data;
    }

    private HashMap<String, double[][]> _convertFeatures() throws IOException {
        HashMap<String, double[][]> dataset = new HashMap<>();
        dataset.put("train_img", _loadImg(KEY_FILE.get("train_img")));
        dataset.put("train_label", _loadLabel(KEY_FILE.get("train_label")));
        dataset.put("test_img", _loadImg(KEY_FILE.get("test_img")));
        dataset.put("test_label", _loadLabel(KEY_FILE.get("test_label")));
        return dataset;
    }

    public void initMnist() throws IOException {
        downloadMnist();
        dataset = _convertFeatures();
        System.out.println("Creating serialized file ...");
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(SAVE_FILE));) {
            oos.writeObject(dataset);
        }
        System.out.println("Done");
    }

    private double[][] _changeOneHotLabel(double[][] x) {
        double[][] ret = new double[x.length][10];
        for (int i = 0; i < x.length; i++) {
            ret[i][(int) x[i][0]] = 1.0;
        }
        return ret;
    }

    private void _normalize(double[][] x) {
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] /= 255;
            }
        }
    }

    public HashMap<String, double[][]> loadMinist() throws IOException, ClassNotFoundException {
        return loadMinist(true, true, false);
    }

    public HashMap<String, double[][]> loadMinist(boolean normalize, boolean flatten, boolean oneHotLabel) throws
            IOException, ClassNotFoundException {
        if (!new File(SAVE_FILE).exists()) {
            initMnist();
        }

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(SAVE_FILE));) {
            dataset = (HashMap<String, double[][]>) ois.readObject();
        }

        if (normalize) {
            _normalize(dataset.get("train_img"));
            _normalize(dataset.get("test_img"));
        }

        if (oneHotLabel) {
            dataset.put("train_label", _changeOneHotLabel(dataset.get("train_label")));
            dataset.put("test_label", _changeOneHotLabel(dataset.get("test_label")));
        }

        return dataset;
    }

    public static void main(String... args) throws IOException {
        new Mnist().initMnist();
    }

//    public static void main(String... args) throws IOException, ClassNotFoundException {
//        HashMap<String, double[][]> ds = new Mnist().loadMinist(true, true, true);
//    }
}
