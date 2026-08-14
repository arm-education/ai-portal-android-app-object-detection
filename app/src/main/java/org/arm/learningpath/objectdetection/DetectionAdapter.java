package org.arm.learningpath.objectdetection;

import android.content.Context;

import java.io.File;

interface DetectionAdapter {
    AdapterDefinition definition();

    DetectionRunner createRunner(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception;

    default void validateModel(Context context, File modelFile, ModelDescriptor descriptor)
            throws Exception {
        try (DetectionRunner ignored = createRunner(context, modelFile, descriptor)) {
        }
    }
}
