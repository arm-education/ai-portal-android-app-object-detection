package org.arm.learningpath.objectdetection;

import java.util.ArrayList;
import java.util.List;

final class ModelRegistry {
    private static final int DEFAULT_CONFIDENCE_PERCENT = 75;
    private static final int LITERT_CONFIDENCE_PERCENT = 25;
    private static final List<ModelDescriptor> BUILT_IN_MODELS = List.of(
            new ModelDescriptor(
                    "yolov5s-executorch",
                    "YOLOv5s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov5s_raspberry_executorch_optimized.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V5,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolov8s-executorch",
                    "YOLOv8s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov8s_raspberry_executorch_optimized.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V8,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolov9s-executorch",
                    "YOLOv9s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov9s_raspberry_executorch_optimized.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V9,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolo26n-fp16-litert",
                    "YOLO26n FP16",
                    LiteRtObjectDetectionAdapter.ID,
                    "LiteRT",
                    "yolo26n_conv2d_f16_weights.tflite",
                    LiteRtObjectDetectionAdapter.CONFIG_YOLO_26_FP16,
                    LITERT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolo26n-int8-weight-only-litert",
                    "YOLO26n INT8 weight-only",
                    LiteRtObjectDetectionAdapter.ID,
                    "LiteRT",
                    "yolo26n_conv_fc_f16_int8w.tflite",
                    LiteRtObjectDetectionAdapter.CONFIG_YOLO_26_INT8_WEIGHT_ONLY,
                    LITERT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolo11n-int8-litert",
                    "YOLO11n INT8",
                    LiteRtObjectDetectionAdapter.ID,
                    "LiteRT",
                    "yolo11n_android_litert_optimized.tflite",
                    LiteRtObjectDetectionAdapter.CONFIG_YOLO_11_INT8,
                    LITERT_CONFIDENCE_PERCENT
            )
    );
    private static final List<ModelDescriptor> MODELS = createModels();

    private ModelRegistry() {
    }

    private static List<ModelDescriptor> createModels() {
        List<ModelDescriptor> models = new ArrayList<>(BUILT_IN_MODELS);
        models.addAll(CompatibleModelRegistry.models());
        models.addAll(GeneratedAdapterRegistry.models());
        return List.copyOf(models);
    }

    static ModelDescriptor forFileName(String fileName) {
        for (ModelDescriptor descriptor : MODELS) {
            if (descriptor.fileName().equalsIgnoreCase(fileName)) {
                return descriptor;
            }
        }
        return null;
    }

    static ModelDescriptor forId(String id) {
        for (ModelDescriptor descriptor : MODELS) {
            if (descriptor.id().equals(id)) {
                return descriptor;
            }
        }
        return null;
    }

    static String supportedFiles() {
        StringBuilder files = new StringBuilder();
        for (ModelDescriptor descriptor : MODELS) {
            if (files.length() > 0) {
                files.append(", ");
            }
            files.append(descriptor.fileName());
        }
        return files.toString();
    }
}
