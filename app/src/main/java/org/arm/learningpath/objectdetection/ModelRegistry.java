package org.arm.learningpath.objectdetection;

import java.util.ArrayList;
import java.util.List;

final class ModelRegistry {
    private static final int DEFAULT_CONFIDENCE_PERCENT = 75;
    private static final List<ModelDescriptor> BUILT_IN_MODELS = List.of(
            new ModelDescriptor(
                    "yolov5s-executorch",
                    "YOLOv5s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov5s-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V5,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolov8s-executorch",
                    "YOLOv8s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov8s-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V8,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "yolov9s-executorch",
                    "YOLOv9s INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "yolov9s-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_YOLO_V9,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "rtdetr-l-executorch",
                    "RT-DETR-L INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "rtdetr-l-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_RT_DETR,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "deformable-detr-executorch",
                    "Deformable DETR INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "deformable-detr-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_DEFORMABLE_DETR,
                    DEFAULT_CONFIDENCE_PERCENT
            ),
            new ModelDescriptor(
                    "ssd-resnet50-executorch",
                    "SSD ResNet50 INT8",
                    ExecuTorchObjectDetectionAdapter.ID,
                    "ExecuTorch",
                    "ssd-resnet50-int8-executorch.pte",
                    ExecuTorchObjectDetectionAdapter.CONFIG_SSD_RESNET50,
                    DEFAULT_CONFIDENCE_PERCENT
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
