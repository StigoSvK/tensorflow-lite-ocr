package sk.stigo.tensorflowliteocr.utils;

import static org.opencv.android.Utils.bitmapToMat;
import static org.opencv.android.Utils.matToBitmap;
import static org.opencv.dnn.Dnn.NMSBoxesRotated;
import static org.opencv.imgproc.Imgproc.boxPoints;
import static org.opencv.imgproc.Imgproc.getPerspectiveTransform;
import static org.opencv.imgproc.Imgproc.warpPerspective;
import static org.opencv.utils.Converters.vector_RotatedRect_to_Mat;
import static org.opencv.utils.Converters.vector_float_to_Mat;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.preference.PreferenceManager;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRotatedRect;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class OCRModelExecutor {
    private static final int MOBILE_NET_DETECTION_IMAGE_HEIGHT = 300;
    private static final int MOBILE_NET_DETECTION_IMAGE_WIDTH = 300;
    private static final int EAST_DETECTION_IMAGE_HEIGHT = 320;
    private static final int EAST_DETECTION_IMAGE_WIDTH = 320;
    private static final float[] MOBILE_NET_DETECTION_IMAGE_MEANS = new float[]{127.5f, 127.5f, 127.5f};
    private static final float[] MOBILE_NET_DETECTION_IMAGE_STDS = new float[]{127.5f, 127.5f, 127.5f};
    private static final float[] EAST_DETECTION_IMAGE_MEANS = new float[]{103.94f, 116.78f, 123.68f};
    private static final float[] EAST_DETECTION_IMAGE_STDS = new float[]{1f, 1f, 1f};
    private static final String MOBILE_NET_DETECTION_MODEL = "final_model.tflite";
    private static final String EAST_DETECTION_MODEL = "lite-model_east-text-detector_fp16_1.tflite";
    private static final String TXT_RECOGNITION_MODEL = "lite-model_keras-ocr_float16_2.tflite";
    private static final float DETECTION_CONFIDENCE_THRESHOLD = 0.5f;
    private static final float DETECTION_NMS_THRESHOLD = 0.4f;
    private static final boolean USE_GPU = false;
    private static final int RECOGNITION_IMAGE_HEIGHT = 31;
    private static final int RECOGNITION_IMAGE_WIDTH = 200;
    private static final float RECOGNITION_IMAGE_MEAN = 0f;
    private static final float RECOGNITION_IMAGE_STD = 255f;
    private static final int RECOGNITION_MODEL_OUTPUT_SIZE = 48;
    private static final String ALPHABETS = "0123456789abcdefghijklmnopqrstuvwxyz";


    private Context context;
    private Interpreter mobileNetDetectionInterpreter;
    private Interpreter eastDetectionInterpreter;
    private Interpreter recognitionInterpreter;
    private GpuDelegate gpuDelegate;
    private MatOfRotatedRect boundingBoxesMat;
    private MatOfInt indicesMat;
    private ByteBuffer recognitionResult;
    private List<String> ocrResults;
    float ratioHeight;
    float ratioWidth;

    public OCRModelExecutor(Context context) throws IOException {
        this.context = context;
        this.indicesMat = new MatOfInt();
        this.ocrResults = new ArrayList<>();
        this.ratioWidth = 0f;
        this.ratioHeight = 0f;
        init();
    }

    private void init() throws IOException {
        this.mobileNetDetectionInterpreter = getInterpreter(this.context, MOBILE_NET_DETECTION_MODEL, USE_GPU);
        this.eastDetectionInterpreter = getInterpreter(this.context, EAST_DETECTION_MODEL, USE_GPU);
        this.recognitionInterpreter = getInterpreter(this.context, TXT_RECOGNITION_MODEL, USE_GPU);
        this.recognitionResult = ByteBuffer.allocateDirect(RECOGNITION_MODEL_OUTPUT_SIZE * 8);
        this.recognitionResult.order(ByteOrder.nativeOrder());
    }

    public Bitmap run(Bitmap image) {
        SharedPreferences preferences = PreferenceManager.getDefaultSharedPreferences(context);

        ocrResults.clear();

        if (preferences.getBoolean("useMobileNetDetectionModel", true)) {
            ratioHeight = (float)image.getHeight() / MOBILE_NET_DETECTION_IMAGE_HEIGHT;
            ratioWidth = (float)image.getWidth() / MOBILE_NET_DETECTION_IMAGE_WIDTH;
            detectTextsUsingMobileNet(image);
        } else {
            ratioHeight = (float)image.getHeight() / EAST_DETECTION_IMAGE_HEIGHT;
            ratioWidth = (float)image.getWidth() / EAST_DETECTION_IMAGE_WIDTH;
            detectTextsUsingEast(image);
        }


        return recognizeTexts(image);
    }

    private void detectTextsUsingMobileNet(Bitmap image) {
        TensorImage detectionTensorImage = ImageUtils.bitmapToTensorImageForDetection(
                image,
                MOBILE_NET_DETECTION_IMAGE_WIDTH,
                MOBILE_NET_DETECTION_IMAGE_HEIGHT,
                MOBILE_NET_DETECTION_IMAGE_MEANS,
                MOBILE_NET_DETECTION_IMAGE_STDS
        );


        Buffer[] detectionInputs = new Buffer[]{detectionTensorImage.getBuffer().rewind()};
        HashMap<Integer, Object> detectionOutputs = new HashMap();


        float[][][] detectionBoxes = new float[1][10][4];
        float[][] detectionClasses = new float[1][10];
        float[][] detectionScores = new float[1][10];
        float[] detectionMasks = new float[1];

        detectionOutputs.put(0, detectionBoxes);
        detectionOutputs.put(1, detectionClasses);
        detectionOutputs.put(2, detectionScores);
        detectionOutputs.put(3, detectionMasks);


        mobileNetDetectionInterpreter.runForMultipleInputsOutputs(detectionInputs, detectionOutputs);

        List detectedRotatedRects = new ArrayList<RotatedRect>();
        List detectedConfidences = new ArrayList<Float>();

        for (int x = 0; x < detectionBoxes[0][0].length; x++) {
            float detectionScoreData = detectionScores[0][x];
            float detectionGeometryYMin = detectionBoxes[0][x][0];
            float detectionGeometryXMin = detectionBoxes[0][x][1];
            float detectionGeometryYMax = detectionBoxes[0][x][2];
            float detectionGeometryXMax = detectionBoxes[0][x][3];

            if (detectionScoreData < 0.5) {
                continue;
            }

            double w = (detectionGeometryXMax - detectionGeometryXMin) * MOBILE_NET_DETECTION_IMAGE_WIDTH;
            double h = (detectionGeometryYMax - detectionGeometryYMin) * MOBILE_NET_DETECTION_IMAGE_HEIGHT;
            double centerW = detectionGeometryXMin * MOBILE_NET_DETECTION_IMAGE_WIDTH  + w/2;
            double centerH = detectionGeometryYMin * MOBILE_NET_DETECTION_IMAGE_HEIGHT + h/2;

            float angle = 0;
            Point center = new Point(centerW,centerH);

            RotatedRect textDetection =
                    new RotatedRect(
                            center,
                            new Size(w, h),
                            (-1 * angle * 180.0 / Math.PI)
                    );
            detectedRotatedRects.add(textDetection);
            detectedConfidences.add(detectionScoreData);
        }

        if (detectedConfidences.size() == 0) {
            return;
        }

        MatOfFloat detectedConfidencesMat = new MatOfFloat(vector_float_to_Mat(detectedConfidences));
        boundingBoxesMat = new MatOfRotatedRect(vector_RotatedRect_to_Mat(detectedRotatedRects));
        NMSBoxesRotated(
                boundingBoxesMat,
                detectedConfidencesMat,
                DETECTION_CONFIDENCE_THRESHOLD,
                DETECTION_NMS_THRESHOLD,
                indicesMat
        );
    }

        public void detectTextsUsingEast(Bitmap image) {
            TensorImage detectionTensorImage = ImageUtils.bitmapToTensorImageForDetection(
                    image,
                    EAST_DETECTION_IMAGE_WIDTH,
                    EAST_DETECTION_IMAGE_HEIGHT,
                    EAST_DETECTION_IMAGE_MEANS,
                    EAST_DETECTION_IMAGE_STDS
            );


            Buffer[] detectionInputs = new Buffer[]{detectionTensorImage.getBuffer().rewind()};
            HashMap<Integer, Object> detectionOutputs = new HashMap();

            float[][][][] detectionScores = new float[1][80][80][1];
            float[][][][] detectionGeometries = new float[1][80][80][5];

            detectionOutputs.put(0, detectionScores);
            detectionOutputs.put(1, detectionGeometries);

            eastDetectionInterpreter.runForMultipleInputsOutputs(detectionInputs, detectionOutputs);

            float[][][][] transposeddetectionScores = new float[1][1][80][80];
            float[][][][] transposedDetectionGeometries = new float[1][5][80][80];

            for (int i = 0; i < transposeddetectionScores[0][0].length ; i++){
                for (int j = 0; j < transposeddetectionScores[0][0][0].length ; j++){
                    for (int k = 0; k < 1 ; k++){
                        transposeddetectionScores[0][k][i][j] = detectionScores[0][i][j][k];
                    }
                    for (int k = 0; k < 5 ; k++){
                        transposedDetectionGeometries[0][k][i][j] = detectionGeometries[0][i][j][k];
                    }
                }
            }

            List detectedRotatedRects = new ArrayList<RotatedRect>();
            List detectedConfidences = new ArrayList<Float>();

            for (int y = 0; y < transposeddetectionScores[0][0].length; y++) {
                float[] detectionScoreData = transposeddetectionScores[0][0][y];
                float[] detectionGeometryX0Data = transposedDetectionGeometries[0][0][y];
                float[] detectionGeometryX1Data = transposedDetectionGeometries[0][1][y];
                float[] detectionGeometryX2Data = transposedDetectionGeometries[0][2][y];
                float[] detectionGeometryX3Data = transposedDetectionGeometries[0][3][y];
                float[] detectionRotationAngleData = transposedDetectionGeometries[0][4][y];

                for (int x = 0; x < transposeddetectionScores[0][0][0].length; x++) {
                    if (detectionScoreData[x] < 0.5) {
                        continue;
                    }

                    double offsetX = x * 4.0;
                    double offsetY = y * 4.0;

                    double h = detectionGeometryX0Data[x] + detectionGeometryX2Data[x];
                    double w = detectionGeometryX1Data[x] + detectionGeometryX3Data[x];

                    float angle = detectionRotationAngleData[x];
                    double cos = Math.cos(angle);
                    double sin = Math.sin(angle);

                    Point offset = new Point(
                            offsetX + cos * detectionGeometryX1Data[x] + sin * detectionGeometryX2Data[x],
                            offsetY - sin * detectionGeometryX1Data[x] + cos * detectionGeometryX2Data[x]
                    );

                    Point p1 = new Point(-sin * h + offset.x, -cos * h + offset.y);
                    Point p3 = new Point(-cos * w + offset.x, sin * w + offset.y);
                    Point center = new Point(0.5 * (p1.x + p3.x), 0.5 * (p1.y + p3.y));

                    RotatedRect textDetection =
                            new RotatedRect(
                                    center,
                                    new Size(w, h),
                                    (-1 * angle * 180.0 / Math.PI)
                            );
                    detectedRotatedRects.add(textDetection);
                    detectedConfidences.add(detectionScoreData[x]);
                }
            }

            if (detectedConfidences.size() == 0) {
                return;
            }

        MatOfFloat detectedConfidencesMat = new MatOfFloat(vector_float_to_Mat(detectedConfidences));
        boundingBoxesMat = new MatOfRotatedRect(vector_RotatedRect_to_Mat(detectedRotatedRects));
        NMSBoxesRotated(
                boundingBoxesMat,
                detectedConfidencesMat,
                DETECTION_CONFIDENCE_THRESHOLD,
                DETECTION_NMS_THRESHOLD,
                indicesMat
        );
    }

    public Bitmap recognizeTexts(Bitmap image) {
        if (boundingBoxesMat == null) {
            return image;
        }

        Bitmap bitmapWithBoundingBoxes = image.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(bitmapWithBoundingBoxes);
        Paint paint = new Paint();

        paint.setStyle( Paint.Style.STROKE);
        paint.setStrokeWidth(10f);
        paint.setColor(Color.GREEN);

        for(int i:indicesMat.toArray()) {
            RotatedRect boundingBox = boundingBoxesMat.toArray()[i];
            List<Point> targetVertices = new ArrayList<>();
            targetVertices.add(new Point(0, (RECOGNITION_IMAGE_HEIGHT - 1)));
            targetVertices.add(new Point(0, 0));
            targetVertices.add(new Point((RECOGNITION_IMAGE_WIDTH - 1), 0));
            targetVertices.add(new Point((RECOGNITION_IMAGE_WIDTH - 1), (RECOGNITION_IMAGE_HEIGHT - 1)));

            List<Point> srcVertices = new ArrayList<>();

            Mat boundingBoxPointsMat = new Mat();
            boxPoints(boundingBox, boundingBoxPointsMat);

            for(int j = 0; j < 4; j++) {
                srcVertices.add(
                        new Point(
                                boundingBoxPointsMat.get(j, 0)[0] * ratioWidth,
                                boundingBoxPointsMat.get(j, 1)[0] * ratioHeight
                        )
                );

                if (j != 0) {
                    canvas.drawLine(
                            (float)boundingBoxPointsMat.get(j, 0)[0] *  ratioWidth,
                            (float)boundingBoxPointsMat.get(j, 1)[0] * ratioHeight,
                            (float)boundingBoxPointsMat.get(j - 1, 0)[0] *  ratioWidth,
                            (float)boundingBoxPointsMat.get(j - 1, 1)[0] * ratioHeight,
                            paint
                    );
                }
            }
            canvas.drawLine(
                    (float)boundingBoxPointsMat.get(0, 0)[0] * ratioWidth,
                    (float)boundingBoxPointsMat.get(0, 1)[0] * ratioHeight,
                    (float)boundingBoxPointsMat.get(3, 0)[0] * ratioWidth,
                    (float)boundingBoxPointsMat.get(3, 1)[0] * ratioHeight,
                    paint
            );

            MatOfPoint2f srcVerticesMat = new MatOfPoint2f(
                    srcVertices.get(0),
                    srcVertices.get(1),
                    srcVertices.get(2),
                    srcVertices.get(3)
            );
            MatOfPoint2f targetVerticesMat = new MatOfPoint2f(
                    targetVertices.get(0),
                    targetVertices.get(1),
                    targetVertices.get(2),
                    targetVertices.get(3)
            );
            Mat rotationMatrix = getPerspectiveTransform(srcVerticesMat, targetVerticesMat);
            Mat recognitionBitmapMat = new Mat();
            Mat srcBitmapMat = new Mat();

            bitmapToMat(image, srcBitmapMat);
            warpPerspective(
                    srcBitmapMat,
                    recognitionBitmapMat,
                    rotationMatrix,
                    new Size(RECOGNITION_IMAGE_WIDTH, RECOGNITION_IMAGE_HEIGHT)
            );

            Bitmap recognitionBitmap = ImageUtils.createEmptyBitmap(
                    RECOGNITION_IMAGE_WIDTH,
                    RECOGNITION_IMAGE_HEIGHT,
                    0,
                    Bitmap.Config.ARGB_8888
            );
            matToBitmap(recognitionBitmapMat, recognitionBitmap);

            TensorImage recognitionTensorImage = ImageUtils.bitmapToTensorImageForRecognition(
                    recognitionBitmap,
                    RECOGNITION_IMAGE_WIDTH,
                    RECOGNITION_IMAGE_HEIGHT,
                    RECOGNITION_IMAGE_MEAN,
                    RECOGNITION_IMAGE_STD
            );

            recognitionResult.rewind();
            recognitionInterpreter.run(recognitionTensorImage.getBuffer(), recognitionResult);

            String recognizedText = "";
            for (int k = 0; k < RECOGNITION_MODEL_OUTPUT_SIZE ; k++) {
                int alphabetIndex = recognitionResult.getInt(k * 8);

                if (0 <= alphabetIndex && 35 >= alphabetIndex) {
                    recognizedText = recognizedText + ALPHABETS.charAt(alphabetIndex);
                }
            }
            if (!recognizedText.isEmpty()) {
                ocrResults.add(recognizedText);
            }
        }

        return bitmapWithBoundingBoxes;
    }

    private Interpreter getInterpreter(Context context, String modelName, boolean useGpu) throws IOException {
        Interpreter.Options tfliteOptions = new Interpreter.Options();
        tfliteOptions.setNumThreads(4);
        this.gpuDelegate = null;
        if (useGpu) {
            this.gpuDelegate = new GpuDelegate();
            tfliteOptions.addDelegate(this.gpuDelegate);
        }

        return new Interpreter(this.loadModelFile(context, modelName), tfliteOptions);
    }

    private final MappedByteBuffer loadModelFile(Context context, String modelFile) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        fileDescriptor.close();
        return retFile;
    }

    public List<String> getOcrResults() {
        return ocrResults;
    }
}
