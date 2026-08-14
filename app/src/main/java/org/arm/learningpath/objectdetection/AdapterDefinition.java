package org.arm.learningpath.objectdetection;

record AdapterDefinition(
        String id,
        int titleResource,
        int descriptionResource,
        int importButtonResource,
        int missingModelResource,
        int readyModelResource,
        int runningResource
) {
}
