package org.arm.learningpath.objectdetection;

import java.util.ArrayList;
import java.util.List;

final class AdapterRegistry {
    private static final List<DetectionAdapter> ADAPTERS = createAdapters();

    private AdapterRegistry() {
    }

    static List<DetectionAdapter> all() {
        return ADAPTERS;
    }

    static DetectionAdapter forId(String id) {
        for (DetectionAdapter adapter : ADAPTERS) {
            if (adapter.definition().id().equals(id)) {
                return adapter;
            }
        }
        return null;
    }

    private static List<DetectionAdapter> createAdapters() {
        List<DetectionAdapter> adapters = new ArrayList<>();
        adapters.add(new ExecuTorchObjectDetectionAdapter());
        adapters.addAll(GeneratedAdapterRegistry.adapters());
        return List.copyOf(adapters);
    }
}
