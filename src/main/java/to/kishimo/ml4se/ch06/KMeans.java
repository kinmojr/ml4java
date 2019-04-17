package to.kishimo.ml4se.ch06;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class KMeans {
    private static int[] colors = new int[]{2, 3, 5, 16};

    public static void main(String[] args) throws IOException {
        String readFilePath = "photo.jpg";
        KMeans km = new KMeans();
        int[][] data = getData(readFilePath);
        for (int k : colors) {
            System.out.println();
            System.out.println("========================");
            System.out.println("Number of clusters: K=" + k);
            int[][] pixcels = km.runKMeans(data, k);
            String writeFilePaht = readFilePath.substring(0, readFilePath.lastIndexOf(".")) + k + ".png";
            save(readFilePath, writeFilePaht, pixcels);
        }
    }

    private static int[][] getData(String path) throws IOException {
        BufferedImage img = ImageIO.read(new File(path));
        int w = img.getWidth();
        int h = img.getHeight();
        int[][] data = new int[w * h][3];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int rgb = img.getRGB(i, j);
                data[i + j * i][0] = rgb >> 16 & 0xff;
                data[i + j * i][1] = rgb >> 8 & 0xff;
                data[i + j * i][2] = rgb & 0xff;
            }
        }
        return data;
    }

    private static void save(String readFilePath, String writeFilePath
            , int[][] data) throws IOException {
        BufferedImage readImg = ImageIO.read(new File(readFilePath));
        int w = readImg.getWidth();
        int h = readImg.getHeight();
        BufferedImage writeImg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                writeImg.setRGB(i, j, data[i + i * j][0] << 16 + data[i + i * j][1] << 8 + data[i + i * j][2]);
            }
        }
        ImageIO.write(writeImg, "bmp", new File(writeFilePath));
    }

    private int[][] runKMeans(int[][] data, int k) {
        int[] cls = new int[data.length];
        int[][] center = new int[k][3];
        Random rand = new Random();
        for (int i = 0; i < k; i++) {
            center[i] = new int[]{rand.nextInt(256), rand.nextInt(256), rand.nextInt(256)};
        }
        System.out.print("Initial centers:");
        for (int i = 0; i < k; i++) {
            System.out.print(center[i][0] + " " + center[i][1] + " " + center[i][2] + ", ");
        }
        System.out.println("\n========================");
        double distortion = 0.0;
        for (int iterNum = 0; iterNum < 50; iterNum++) {
            int[][] centerNew = new int[k][3];
            int[] numPoints = new int[k];
            double distortionNew = 0.0;
            for (int pix = 0; pix < data.length; pix++) {
                double minDist = 256.0 * 256.0 * 3.0;
                int[] point = new int[3];
                point = data[pix];
                for (int i = 0; i < k; i++) {
                    double d = Math.pow((double) (point[0] - center[i][0]), 2.0) + Math.pow((double) (point[1] - center[i][1]), 2.0) + Math.pow((double) (point[2] - center[i][2]), 2.0);
                    if (d < minDist) {
                        minDist = d;
                        cls[pix] = i;
                    }
                }
                centerNew[cls[pix]][0] += point[0];
                centerNew[cls[pix]][1] += point[1];
                centerNew[cls[pix]][2] += point[2];
                numPoints[cls[pix]] += 1;
                distortionNew += minDist;
            }
            for (int i = 0; i < k; i++) {
                if (numPoints[i] != 0) {
                    centerNew[i][0] = centerNew[i][0] / numPoints[i];
                    centerNew[i][1] = centerNew[i][1] / numPoints[i];
                    centerNew[i][2] = centerNew[i][2] / numPoints[i];
                }
            }
            center = centerNew;
            for (int i = 0; i < k; i++) {
                System.out.print("[" + center[i][0] + ", " + center[i][1] + ", " + center[i][2] + "], ");
            }
            System.out.println("\nDistortion: J=" + (int) distortionNew);
            if (iterNum > 0 && distortion - distortionNew < distortion * 0.001) break;
            distortion = distortionNew;
        }
        int[][] pixels = new int[data.length][3];
        for (int i = 0; i < data.length; i++) {
            pixels[i][0] = center[cls[i]][0];
            pixels[i][1] = center[cls[i]][1];
            pixels[i][2] = center[cls[i]][2];
        }
        return pixels;
    }
}
