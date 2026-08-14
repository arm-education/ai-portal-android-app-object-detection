package org.arm.learningpath.objectdetection;

import android.graphics.Bitmap;

import java.io.Closeable;

interface DetectionRunner extends Closeable {
    DetectionResult detect(Bitmap bitmap, float confidenceThreshold) throws Exception;

    @Override
    void close();
}
