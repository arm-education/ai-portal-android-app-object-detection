plugins {
    id("com.android.application")
}

android {
    namespace = "org.arm.learningpath.objectdetection"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.arm.learningpath.objectdetection"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        disable += "ChromeOsAbiSupport"
    }
}

dependencies {
    implementation("androidx.activity:activity:1.10.1")
    implementation("androidx.camera:camera-camera2:1.4.2")
    implementation("androidx.camera:camera-lifecycle:1.4.2")
    implementation("androidx.camera:camera-view:1.4.2")
    implementation("org.pytorch:executorch-android:1.3.1")
    implementation("com.google.ai.edge.litert:litert:2.2.0")
}

apply(from = "generated-runtime-dependencies.gradle.kts")
