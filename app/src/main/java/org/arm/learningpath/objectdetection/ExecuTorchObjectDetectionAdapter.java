package org.arm.learningpath.objectdetection;

import android.content.Context;

import java.io.File;

final class ExecuTorchObjectDetectionAdapter implements DetectionAdapter {
    static final String ID = "executorch-object-detection";
    static final String CONFIG_YOLO_V5 = "yolo-v5";
    static final String CONFIG_YOLO_V8 = "yolo-v8";
    static final String CONFIG_YOLO_V9 = "yolo-v9";
    static final String CONFIG_RT_DETR = "rt-detr";
    static final String CONFIG_DEFORMABLE_DETR = "deformable-detr";
    static final String CONFIG_SSD_RESNET50 = "ssd-resnet50";

    private static final AdapterDefinition DEFINITION = new AdapterDefinition(
            ID,
            R.string.executorch_mode_title,
            R.string.executorch_mode_description,
            R.string.import_executorch_model,
            R.string.executorch_model_missing,
            R.string.executorch_model_ready,
            R.string.running_executorch
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
                    "The ExecuTorch detection adapter received another model type."
            );
        }
        return switch (descriptor.configurationId()) {
            case CONFIG_YOLO_V5 -> new ExecuTorchYoloDetector(
                    context,
                    modelFile,
                    ExecuTorchYoloDetector.Profile.YOLO_V5
            );
            case CONFIG_YOLO_V8 -> new ExecuTorchYoloDetector(
                    context,
                    modelFile,
                    ExecuTorchYoloDetector.Profile.YOLO_V8
            );
            case CONFIG_YOLO_V9 -> new ExecuTorchYoloDetector(
                    context,
                    modelFile,
                    ExecuTorchYoloDetector.Profile.YOLO_V9
            );
            case CONFIG_RT_DETR -> new ExecuTorchDetrDetector(
                    context,
                    modelFile,
                    ExecuTorchDetrDetector.Profile.RT_DETR
            );
            case CONFIG_DEFORMABLE_DETR -> new ExecuTorchDetrDetector(
                    context,
                    modelFile,
                    ExecuTorchDetrDetector.Profile.DEFORMABLE_DETR
            );
            case CONFIG_SSD_RESNET50 -> new ExecuTorchSsdDetector(
                    context,
                    modelFile
            );
            default -> throw new IllegalArgumentException(
                    "No ExecuTorch detector configuration is registered for "
                            + descriptor.configurationId()
            );
        };
    }
}
