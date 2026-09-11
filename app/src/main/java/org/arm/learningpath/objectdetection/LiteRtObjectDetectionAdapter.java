package org.arm.learningpath.objectdetection;

import android.content.Context;

import java.io.File;

final class LiteRtObjectDetectionAdapter implements DetectionAdapter {
    static final String ID = "litert-object-detection";
    static final String CONFIG_YOLO_26_FP16 = "yolo-26-fp16";
    static final String CONFIG_YOLO_26_INT8_WEIGHT_ONLY = "yolo-26-int8-weight-only";
    static final String CONFIG_YOLO_11_INT8 = "yolo-11-int8";

    private static final AdapterDefinition DEFINITION = new AdapterDefinition(
            ID,
            R.string.litert_mode_title,
            R.string.litert_mode_description,
            R.string.import_litert_model,
            R.string.litert_model_missing,
            R.string.litert_model_ready,
            R.string.running_litert
    );

    @Override
    public AdapterDefinition definition() {
        return DEFINITION;
    }

    @Override
    public DetectionRunner createRunner(Context context, File modelFile,
                                        ModelDescriptor descriptor) throws Exception {
        if (!ID.equals(descriptor.adapterId())) {
            throw new IllegalArgumentException(
                    "The LiteRT detection adapter received another model type."
            );
        }
        return switch (descriptor.configurationId()) {
            case CONFIG_YOLO_26_FP16 -> new LiteRtYoloDetector(
                    context,
                    modelFile,
                    LiteRtYoloDetector.Profile.YOLO_26_FP16
            );
            case CONFIG_YOLO_26_INT8_WEIGHT_ONLY -> new LiteRtYoloDetector(
                    context,
                    modelFile,
                    LiteRtYoloDetector.Profile.YOLO_26_INT8_WEIGHT_ONLY
            );
            case CONFIG_YOLO_11_INT8 -> new LiteRtYoloDetector(
                    context,
                    modelFile,
                    LiteRtYoloDetector.Profile.YOLO_11_INT8
            );
            default -> throw new IllegalArgumentException(
                    "No LiteRT detector configuration is registered for "
                            + descriptor.configurationId()
            );
        };
    }
}
