package sk.stigo.tensorflowliteocr.utils;

import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;

public class ImageUtils {

    public static TensorImage bitmapToTensorImageForDetection(Bitmap bitmapIn, int width, int height, float[] means, float[] stds) {
        ResizeOp resizeOp = new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR);
        NormalizeOp normalizeOp = new NormalizeOp(means, stds);
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        ImageProcessor.Builder imageProcessorBuilder = new ImageProcessor.Builder();
        imageProcessorBuilder.add(resizeOp);
        imageProcessorBuilder.add(normalizeOp);

        ImageProcessor imageProcessor = imageProcessorBuilder.build();

        tensorImage.load(bitmapIn);
        tensorImage = imageProcessor.process(tensorImage);
        return tensorImage;
    }

    public static TensorImage bitmapToTensorImageForRecognition(Bitmap bitmapIn, int width, int height, float means, float stds) {
        ResizeOp resizeOp = new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR);
        NormalizeOp normalizeOp = new NormalizeOp(means, stds);
        TransformToGrayscaleOp transformToGrayscaleOp = new TransformToGrayscaleOp();
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        ImageProcessor.Builder imageProcessorBuilder = new ImageProcessor.Builder();
        imageProcessorBuilder.add(resizeOp);
        imageProcessorBuilder.add(transformToGrayscaleOp);
        imageProcessorBuilder.add(normalizeOp);

        ImageProcessor imageProcessor = imageProcessorBuilder.build();

        tensorImage.load(bitmapIn);
        tensorImage = imageProcessor.process(tensorImage);
        return tensorImage;
    }


    public static Bitmap createEmptyBitmap(
            int imageWidth,
            int imageHeigth,
            int color,
            Bitmap.Config imageConfig
    ){
        Bitmap ret = Bitmap.createBitmap(imageWidth, imageHeigth, imageConfig);
        if (color != 0) {
            ret.eraseColor(color);
        }
        return ret;
    }
}
