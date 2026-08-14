package org.arm.learningpath.objectdetection;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

final class ModelImporter {
    private static final long MAXIMUM_MODEL_BYTES = 700L * 1024L * 1024L;

    private ModelImporter() {
    }

    static ImportResult importModel(Context context, Uri source, File modelsDirectory)
            throws Exception {
        ContentResolver resolver = context.getContentResolver();
        String fileName = displayName(resolver, source);
        if (fileName == null) {
            throw new IOException("Android could not determine the selected file name.");
        }

        ModelDescriptor descriptor = ModelRegistry.forFileName(fileName);
        if (descriptor == null) {
            throw new IOException(
                    "This application does not recognize " + fileName
                            + ". Keep the downloaded filename unchanged and select one of: "
                            + ModelRegistry.supportedFiles()
            );
        }

        DetectionAdapter adapter = AdapterRegistry.forId(descriptor.adapterId());
        if (adapter == null) {
            throw new IOException("No adapter is registered for " + descriptor.adapterId());
        }

        File destination = modelFile(modelsDirectory, descriptor);
        File parent = destination.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("Unable to create app model storage.");
        }

        File temporaryFile = new File(parent, destination.getName() + ".importing");
        if (temporaryFile.exists() && !temporaryFile.delete()) {
            throw new IOException("Unable to replace the temporary model file.");
        }
        try (InputStream input = resolver.openInputStream(source)) {
            if (input == null) {
                throw new IOException("Android could not open the selected model file.");
            }
            copy(input, temporaryFile);
            adapter.validateModel(context, temporaryFile, descriptor);
        } catch (Exception exception) {
            temporaryFile.delete();
            throw exception;
        }

        if (destination.exists() && !destination.delete()) {
            temporaryFile.delete();
            throw new IOException("Unable to replace the installed model.");
        }
        if (!temporaryFile.renameTo(destination)) {
            temporaryFile.delete();
            throw new IOException("Unable to activate the imported model.");
        }

        State state = new State(descriptor);
        writeAdapterState(modelsDirectory, state);
        writeActiveState(modelsDirectory, state);
        return new ImportResult(state);
    }

    static State activeState(File modelsDirectory) {
        return readState(new File(modelsDirectory, "active-model.json"), modelsDirectory, null);
    }

    static State stateForAdapter(File modelsDirectory, String adapterId) {
        return readState(adapterStateFile(modelsDirectory, adapterId), modelsDirectory, adapterId);
    }

    static void activate(File modelsDirectory, State state) throws Exception {
        if (!modelFile(modelsDirectory, state.descriptor()).isFile()) {
            throw new IOException("The selected detector is not installed.");
        }
        writeActiveState(modelsDirectory, state);
    }

    static File modelFile(File modelsDirectory, ModelDescriptor descriptor) {
        return new File(new File(modelsDirectory, descriptor.id()), descriptor.fileName());
    }

    private static State readState(File stateFile, File modelsDirectory, String adapterId) {
        if (!stateFile.isFile()) {
            return null;
        }
        try {
            JSONObject state = new JSONObject(readText(stateFile));
            ModelDescriptor descriptor = ModelRegistry.forId(state.getString("descriptorId"));
            if (descriptor == null
                    || (adapterId != null && !adapterId.equals(descriptor.adapterId()))
                    || !modelFile(modelsDirectory, descriptor).isFile()) {
                return null;
            }
            return new State(descriptor);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static void writeAdapterState(File modelsDirectory, State state) throws Exception {
        writeState(adapterStateFile(modelsDirectory, state.adapterId()), state);
    }

    private static void writeActiveState(File modelsDirectory, State state) throws Exception {
        writeState(new File(modelsDirectory, "active-model.json"), state);
    }

    private static File adapterStateFile(File modelsDirectory, String adapterId) {
        return new File(new File(modelsDirectory, "adapter-states"), adapterId + ".json");
    }

    private static void writeState(File stateFile, State state) throws Exception {
        File parent = stateFile.getParentFile();
        if (parent == null || (!parent.isDirectory() && !parent.mkdirs())) {
            throw new IOException("Unable to create app model storage.");
        }
        JSONObject value = new JSONObject();
        value.put("descriptorId", state.descriptor().id());
        try (FileOutputStream output = new FileOutputStream(stateFile)) {
            output.write(value.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private static void copy(InputStream source, File destination) throws IOException {
        byte[] buffer = new byte[1024 * 1024];
        long total = 0;
        try (BufferedInputStream input = new BufferedInputStream(source, buffer.length);
             BufferedOutputStream output = new BufferedOutputStream(
                     new FileOutputStream(destination), buffer.length)) {
            int count;
            while ((count = input.read(buffer)) != -1) {
                total += count;
                if (total > MAXIMUM_MODEL_BYTES) {
                    throw new IOException("The selected model is larger than 700 MB.");
                }
                output.write(buffer, 0, count);
            }
        }
    }

    private static String displayName(ContentResolver resolver, Uri source) {
        try (Cursor cursor = resolver.query(
                source,
                new String[]{OpenableColumns.DISPLAY_NAME},
                null,
                null,
                null
        )) {
            if (cursor != null && cursor.moveToFirst()) {
                int nameColumn = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameColumn >= 0) {
                    return cursor.getString(nameColumn);
                }
            }
        }
        return null;
    }

    private static String readText(File file) throws IOException {
        try (InputStream input = new FileInputStream(file)) {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toString(StandardCharsets.UTF_8.name());
        }
    }

    record State(ModelDescriptor descriptor) {
        String adapterId() {
            return descriptor.adapterId();
        }
    }

    record ImportResult(State state) {
    }
}
