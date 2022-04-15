package sk.stigo.tensorflowliteocr.utils;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.ExifInterface;

import java.io.File;
import java.io.IOException;

public class PhotoUtils {
    public static Bitmap getRotatedImg(Bitmap source, String currentPhotoPath) throws IOException {
        ExifInterface ei = new ExifInterface(currentPhotoPath);
        int orientation = ei.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED);

        switch(orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                return rotateImage(source, 90);
            case ExifInterface.ORIENTATION_ROTATE_180:
                return rotateImage(source, 180);
            case ExifInterface.ORIENTATION_ROTATE_270:
                return rotateImage(source, 270);
            case ExifInterface.ORIENTATION_NORMAL:
            default:
                return source;
        }
    }

    private static Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true);
    }

    public static void deletePhoto(String currentPhotoPath) {
        if (currentPhotoPath != null) {
            File f = new File(currentPhotoPath);
            f.delete();
        }
    }
}
