import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.Scanner;

public class App {

    public static int[][] convertToGrayscaleArray(File imageFile) throws Exception {
        BufferedImage image = ImageIO.read(imageFile);
        int width = image.getWidth();
        int height = image.getHeight();

        int[][] grayPixels = new int[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);

                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);
                grayPixels[y][x] = gray;
            }
        }

        return grayPixels;
    }

    public static int predict(int method, int A, int B, int C, int previous, int previous2) {
        if (method == 1) {
            if (B <= Math.min(A, C)) return Math.max(A, C);
            else if (B >= Math.max(A, C)) return Math.min(A, C);
            else return A + C - B;
        } else if (method == 2) {
            return previous;
        } else if (method == 3) {
            return Math.min(255, Math.max(0, 2 * previous - previous2));
        } else {
            throw new IllegalArgumentException("Unknown prediction method: " + method);
        }
    }

    public static double calc_error(int actual, int predicted) {
        return actual - predicted;
    }

    public static int[][] quantizeGrayscale(int[][] grayPixels, int levels, int minValue, int maxValue) {
        int height = grayPixels.length;
        int width = grayPixels[0].length;
        int[][] quantizedPixels = new int[height][width];
        
        double range = maxValue - minValue;
        double step = range / (levels - 1);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = grayPixels[y][x];
                double normalized = (pixel - minValue) / step;
                int quantizedValue = (int) (Math.round(normalized) * step + minValue);
                quantizedPixels[y][x] = Math.min(maxValue, Math.max(minValue, quantizedValue));
            }
        }
        
        return quantizedPixels;
    }

    public static int[][] dequantizeGrayscale(int[][] quantizedPixels, int levels, int minValue, int maxValue) {
        int height = quantizedPixels.length;
        int width = quantizedPixels[0].length;
        int[][] dequantizedPixels = new int[height][width];
    
        double range = maxValue - minValue;
        double step = range / (levels - 1);
    
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int quantizedValue = quantizedPixels[y][x];
                double normalized = (quantizedValue - minValue) / step;
                int binIndex = (int) Math.round(normalized);
                double dequantizedValue = minValue + binIndex * step; 
    
                int reconstructedValue = (int) Math.round(dequantizedValue);
                dequantizedPixels[y][x] = Math.min(maxValue, Math.max(minValue, reconstructedValue));
            }
        }
    
        return dequantizedPixels;
    }

    public static int[][] compress(int[][] grayPixels, int method, int quantLevels) {
        int height = grayPixels.length;
        int width = grayPixels[0].length;
        int[][] quantizedErrors = new int[height][width];
        int[][] decodedImage = new int[height][width];

        decodedImage[0][0] = grayPixels[0][0];
        quantizedErrors[0][0] = 0;

        for (int x = 1; x < width; x++) {
            int predicted;
            if (method == 1) {
                predicted = decodedImage[0][x - 1];
            } else if (method == 3 && x >= 2) {
                predicted = predict(method, 0, 0, 0, decodedImage[0][x - 1], decodedImage[0][x - 2]);
            } else {
                predicted = predict(method, 0, 0, 0, decodedImage[0][x - 1], 0);
            }

            int actual = grayPixels[0][x];
            int error = (int) calc_error(actual, predicted);

            int[][] singlePixelError = new int[1][1];
            singlePixelError[0][0] = error;
            int[][] quantizedPixelError = quantizeGrayscale(singlePixelError, quantLevels, -255, 255);
            int quantizedError = quantizedPixelError[0][0];
            quantizedErrors[0][x] = quantizedError;

            int[][] singleQuantizedError = new int[1][1];
            singleQuantizedError[0][0] = quantizedError;
            int[][] dequantizedPixelError = dequantizeGrayscale(singleQuantizedError, quantLevels, -255, 255);
            int dequantizedError = dequantizedPixelError[0][0];

            decodedImage[0][x] = predicted + dequantizedError;
        }

        for (int y = 1; y < height; y++) {
            int predicted;
            if (method == 1) {
                predicted = decodedImage[y - 1][0];
            } else if (method == 3 && y >= 2) {
                predicted = predict(method, 0, 0, 0, decodedImage[y - 1][0], decodedImage[y - 2][0]);
            } else {
                predicted = predict(method, 0, 0, 0, decodedImage[y - 1][0], 0);
            }

            int actual = grayPixels[y][0];
            int error = (int) calc_error(actual, predicted);

            int[][] singlePixelError = new int[1][1];
            singlePixelError[0][0] = error;
            int[][] quantizedPixelError = quantizeGrayscale(singlePixelError, quantLevels, -255, 255);
            int quantizedError = quantizedPixelError[0][0];
            quantizedErrors[y][0] = quantizedError;

            int[][] singleQuantizedError = new int[1][1];
            singleQuantizedError[0][0] = quantizedError;
            int[][] dequantizedPixelError = dequantizeGrayscale(singleQuantizedError, quantLevels, -255, 255);
            int dequantizedError = dequantizedPixelError[0][0];

            decodedImage[y][0] = predicted + dequantizedError;
        }

        for (int y = 1; y < height; y++) {
            for (int x = 1; x < width; x++) {
                int predicted;
                if (method == 1) {
                    int A = decodedImage[y][x - 1];
                    int B = decodedImage[y - 1][x - 1];
                    int C = decodedImage[y - 1][x];
                    predicted = predict(method, A, B, C, 0, 0);
                } else if (method == 3 && x >= 2) {
                    predicted = predict(method, 0, 0, 0, decodedImage[y][x - 1], decodedImage[y][x - 2]);
                } else {
                    predicted = predict(method, 0, 0, 0, decodedImage[y][x - 1], 0);
                }

                int actual = grayPixels[y][x];
                int error = (int) calc_error(actual, predicted);

                int[][] singlePixelError = new int[1][1];
                singlePixelError[0][0] = error;
                int[][] quantizedPixelError = quantizeGrayscale(singlePixelError, quantLevels, -255, 255);
                int quantizedError = quantizedPixelError[0][0];
                quantizedErrors[y][x] = quantizedError;

                int[][] singleQuantizedError = new int[1][1];
                singleQuantizedError[0][0] = quantizedError;
                int[][] dequantizedPixelError = dequantizeGrayscale(singleQuantizedError, quantLevels, -255, 255);
                int dequantizedError = dequantizedPixelError[0][0];

                decodedImage[y][x] = predicted + dequantizedError;
            }
        }

        return quantizedErrors;
    }

    public static int[][] decompress(File compressedFile, int method, int quantLevels) throws Exception {
        DataInputStream dis = new DataInputStream(new FileInputStream(compressedFile));

        int height = dis.readInt();
        int width = dis.readInt();
        int[][] decodedImage = new int[height][width];
        int[][] quantizedErrors = new int[height][width];

        for (int x = 0; x < width; x++) {
            decodedImage[0][x] = dis.readInt();
            quantizedErrors[0][x] = 0;
        }

        for (int y = 0; y < height; y++) {
            decodedImage[y][0] = dis.readInt();
            quantizedErrors[y][0] = 0;
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                quantizedErrors[y][x] = dis.readInt();
            }
        }

        dis.close();

        for (int y = 1; y < height; y++) {
            for (int x = 1; x < width; x++) {
                int predicted;
                if (method == 1) {
                    int A = decodedImage[y][x - 1];
                    int B = decodedImage[y - 1][x - 1];
                    int C = decodedImage[y - 1][x];
                    predicted = predict(method, A, B, C, 0, 0);
                } else if (method == 3 && x >= 2) {
                    predicted = predict(method, 0, 0, 0, decodedImage[y][x - 1], decodedImage[y][x - 2]);
                } else {
                    predicted = predict(method, 0, 0, 0, decodedImage[y][x - 1], 0);
                }

                int[][] singleQuantizedError = new int[1][1];
                singleQuantizedError[0][0] = quantizedErrors[y][x];
                int[][] dequantizedPixelError = dequantizeGrayscale(singleQuantizedError, quantLevels, -255, 255);
                int dequantizedError = dequantizedPixelError[0][0];

                decodedImage[y][x] = predicted + dequantizedError;
            }
        }

        return decodedImage;
    }

    public static double calculateMSE(int[][] original, int[][] decompressed) {
        int height = original.length;
        int width = original[0].length;
        double sumSquaredError = 0.0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double error = original[y][x] - decompressed[y][x];
                sumSquaredError += error * error;
            }
        }

        return sumSquaredError / (height * width);
    }

    public static void saveCompressedData(int[][] grayPixels, int[][] quantizedErrors, String outputPath) throws Exception {
        File outputFile = new File(outputPath);
        DataOutputStream dos = new DataOutputStream(new FileOutputStream(outputFile));
        int height = grayPixels.length;
        int width = grayPixels[0].length;

        dos.writeInt(height);
        dos.writeInt(width);

        for (int x = 0; x < width; x++) {
            dos.writeInt(grayPixels[0][x]);
        }

        for (int y = 0; y < height; y++) {
            dos.writeInt(grayPixels[y][0]);
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dos.writeInt(quantizedErrors[y][x]);
            }
        }

        dos.close();
    }

    public static void saveImage(int[][] image, String outputPath) throws Exception {
        int height = image.length;
        int width = image[0].length;

        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = Math.min(255, Math.max(0, image[y][x]));
                int rgb = (pixel << 16) | (pixel << 8) | pixel;
                bufferedImage.setRGB(x, y, rgb);
            }
        }

        File outputFile = new File(outputPath);
        ImageIO.write(bufferedImage, "bmp", outputFile);
    }

    public static double calculateCompressionRatio(int[][] grayPixels, int[][] quantizedErrors, int quantizerLevels) {
        int originalHeight = grayPixels.length;
        int originalWidth = grayPixels[0].length;
        int totalOriginalPixels = originalHeight * originalWidth;
        int originalSizeBits = totalOriginalPixels * 8; 
    
        int compressedHeight = quantizedErrors.length;
        int compressedWidth = quantizedErrors[0].length;
        int totalCompressedPixels = compressedHeight * compressedWidth;
    
        int bitsPerError = (int) Math.ceil(Math.log(quantizerLevels) / Math.log(2));
        int compressedSizeBits = totalCompressedPixels * bitsPerError;
    
        return (double) originalSizeBits / compressedSizeBits;
    }
    

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        int[][] grayPixels = null;
        File inputFile = null;

        while (true) {
            System.out.println("\nImage Compression/Decompression Menu:");
            System.out.println("1. Compress");
            System.out.println("2. Decompress");
            System.out.println("3. Exit");
            System.out.print("Choose an option (1-3): ");

            int choice = scanner.nextInt();

            if (choice == 3) {
                System.out.println("Exiting program.");
                break;
            }

            if (choice == 1) {
                System.out.print("Enter the path of the image to compress (e.g., assets\\girlgray.bmp): ");
                scanner.nextLine(); 
                String imagePath = scanner.nextLine();
                inputFile = new File(imagePath);

                if (!inputFile.exists()) {
                    System.out.println("Image file does not exist. Please try again.");
                    continue;
                }

                grayPixels = convertToGrayscaleArray(inputFile);

                System.out.println("\nSelect prediction method:");
                System.out.println("1. Adaptive 2-D Predictor)");
                System.out.println("2. First-Order Prediction");
                System.out.println("3. Second-Order Prediction");
                System.out.print("Choose a method (1-3): ");
                int method = scanner.nextInt();

                if (method < 1 || method > 3) {
                    System.out.println("Invalid method. Please try again.");
                    continue;
                }

                System.out.print("Enter number of quantization levels (e.g., 8, 16, 32): ");
                int quantLevels = scanner.nextInt();

                if (quantLevels < 2 || quantLevels > 256) {
                    System.out.println("Invalid number of quantization levels. Please try again.");
                    continue;
                }

                int[][] quantizedErrors = compress(grayPixels, method, quantLevels);
                String outputPath = "compressed_method_" + method + "_levels_" + quantLevels + ".bin";
                saveCompressedData(grayPixels, quantizedErrors, outputPath);
                System.out.println("Image compressed and saved to " + outputPath);


                double compressionRatio = calculateCompressionRatio(grayPixels, quantizedErrors, quantLevels);
                System.out.printf("Compression Ratio: (%.2f%% of original size)%n",  (1.0 / compressionRatio) * 100);

                int[][] decompressedImage = decompress(new File(outputPath), method, quantLevels);
                double mse = calculateMSE(grayPixels, decompressedImage);
                System.out.printf("Mean Squared Error (MSE): %.4f%n", mse);

            } else if (choice == 2) {
                System.out.print("Enter the path of the compressed file (e.g., compressed_method_1_levels_8.bin): ");
                scanner.nextLine();
                String compressedPath = scanner.nextLine();
                File compressedFile = new File(compressedPath);

                if (!compressedFile.exists()) {
                    System.out.println("Compressed file does not exist. Please try again.");
                    continue;
                }

                System.out.println("\nSelect prediction method used for compression:");
                System.out.println("1. Adaptive 2-D Predictor)");
                System.out.println("2. First-Order Prediction");
                System.out.println("3. Second-Order Prediction");
                System.out.print("Choose a method (1-3): ");
                int method = scanner.nextInt();

                if (method < 1 || method > 3) {
                    System.out.println("Invalid method. Please try again.");
                    continue;
                }

                System.out.print("Enter number of quantization levels used (e.g., 8, 16, 32): ");
                int quantLevels = scanner.nextInt();

                if (quantLevels < 2 || quantLevels > 256) {
                    System.out.println("Invalid number of quantization levels. Please try again.");
                    continue;
                }

                System.out.print("Enter the path of the original image for MSE calculation (e.g., assets\\girlgray.bmp, or press Enter to skip MSE): ");
                scanner.nextLine(); 
                String originalImagePath = scanner.nextLine();

                if (!originalImagePath.isEmpty()) {
                    inputFile = new File(originalImagePath);
                    if (!inputFile.exists()) {
                        System.out.println("Original image file does not exist. Skipping MSE calculation.");
                        grayPixels = null;
                    } else {
                        grayPixels = convertToGrayscaleArray(inputFile);
                    }
                } else {
                    grayPixels = null;
                }

                int[][] decompressedImage = decompress(compressedFile, method, quantLevels);
                String outputImagePath = "decompressed_method_" + method + "_levels_" + quantLevels + ".bmp";
                saveImage(decompressedImage, outputImagePath);
                System.out.println("Image decompressed and saved to " + outputImagePath);

                if (grayPixels != null) {
                    double mse = calculateMSE(grayPixels, decompressedImage);
                    System.out.printf("Mean Squared Error (MSE): %.4f%n", mse);
                } else {
                    System.out.println("MSE calculation skipped (original image not provided).");
                }

            } else {
                System.out.println("Invalid option. Please try again.");
            }
        }

        scanner.close();
    }
}