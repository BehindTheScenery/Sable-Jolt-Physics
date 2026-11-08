package dev.behindthescenery.sablejolt;

import org.jetbrains.annotations.ApiStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Puts a native library on disk and loads it without ever touching a copy that is already there.
 * <p>
 * Windows refuses to delete or overwrite a DLL any process has loaded, and a game folder is shared: a second
 * game, a game that has not finished exiting, or a virus scanner can all hold the file. Loading the same file
 * twice is fine, though, so nothing here deletes or rewrites a good copy:
 * <ul>
 *     <li>the folder is named after the library's content, so different versions never meet at one path, and
 *     the JVM never mistakes a new library for an old one it loaded by the same path;</li>
 *     <li>an existing copy is compared byte by byte and then simply reused, which also spares a scanner from
 *     checking a freshly written unsigned DLL on every start;</li>
 *     <li>a new copy is written under a temporary name and renamed, so another process never sees half a file;</li>
 *     <li>when the game folder is unusable (permissions, a damaged copy that is locked), the same is tried in
 *     the player's home folder.</li>
 * </ul>
 */
@ApiStatus.Internal
public final class NativeLibraryInstaller {
    private static final Logger LOGGER = LoggerFactory.getLogger("SableJolt");
    /** Pauses before the second and third try: scanners usually let go of a new file within a second. */
    private static final long[] RETRY_DELAYS_MS = {200, 600};

    private NativeLibraryInstaller() {
    }

    /**
     * Loads the library the way its caller would: {@code System.load} binds a library to the class loader of the
     * class calling it, and JNI only looks for a class's native methods in libraries of that class's own loader.
     * So the caller has to pass a lambda of its own that calls {@code System.load}. A method reference or a
     * helper here could end up binding the library to the wrong loader.
     */
    @FunctionalInterface
    public interface Loader {
        void load(String absolutePath);
    }

    /**
     * Installs the library under {@code <baseDir>/<content hash>/<fileName>} and loads it, falling back to
     * {@code <user home>/.sable/<baseDir's name>}.
     *
     * @return the file that was loaded
     * @throws IOException when no location worked; the failure of each one is attached
     */
    public static Path install(final Path baseDir, final String fileName, final byte[] content, final Loader loader) throws IOException {
        final List<Path> bases = new ArrayList<>();
        bases.add(baseDir);
        final Path home = Paths.get(System.getProperty("user.home", System.getProperty("user.dir")), ".sable")
                .resolve(baseDir.getFileName());
        if (!home.toAbsolutePath().normalize().equals(baseDir.toAbsolutePath().normalize())) {
            bases.add(home);
        }
        return install(bases, fileName, content, loader);
    }

    static Path install(final List<Path> bases, final String fileName, final byte[] content, final Loader loader) throws IOException {
        final String version = shortHash(content);
        final IOException failure = new IOException("Could not install the native library " + fileName);
        for (final Path base : bases) {
            final Path dir = base.resolve(version).normalize();
            try {
                final Path target = withRetries(() -> place(dir, fileName, content));
                final String path = target.toAbsolutePath().toString();
                withRetries(() -> {
                    loader.load(path);
                    return null;
                });
                return target;
            } catch (final IOException | UnsatisfiedLinkError | SecurityException e) {
                LOGGER.warn("[SableJolt] Could not load {} from {}: {}", fileName, dir, e.toString());
                failure.addSuppressed(e);
            }
        }
        throw failure;
    }

    private interface Attempt<T> {
        T run() throws IOException;
    }

    private static <T> T withRetries(final Attempt<T> attempt) throws IOException {
        for (int i = 0; ; i++) {
            try {
                return attempt.run();
            } catch (final IOException | UnsatisfiedLinkError e) {
                if (i == RETRY_DELAYS_MS.length) {
                    throw e;
                }
            }
            try {
                Thread.sleep(RETRY_DELAYS_MS[i]);
            } catch (final InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Interrupted while installing a native library", e);
            }
        }
    }

    /** Makes sure {@code dir/fileName} holds content, leaving a good copy untouched. */
    private static Path place(final Path dir, final String fileName, final byte[] content) throws IOException {
        final Path target = dir.resolve(fileName);
        if (holds(target, content)) {
            return target;
        }
        Files.createDirectories(dir);
        // Not Files.createTempFile: its files are private to the user, and the library's permissions would change
        // with that from what every other file in the game folder has.
        final Path part = dir.resolve(fileName + "." + ProcessHandle.current().pid() + ".part");
        try {
            Files.write(part, content);
            try {
                Files.move(part, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (final AtomicMoveNotSupportedException e) {
                Files.move(part, target, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (final IOException e) {
            // Another game started at the same moment, finished first and already has the file loaded.
            if (!holds(target, content)) {
                throw e;
            }
        } finally {
            try {
                Files.deleteIfExists(part);
            } catch (final IOException ignored) {
                // A stray .part file harms nothing.
            }
        }
        return target;
    }

    private static boolean holds(final Path file, final byte[] content) {
        try {
            return Files.size(file) == content.length && Arrays.equals(Files.readAllBytes(file), content);
        } catch (final IOException | SecurityException e) {
            return false;
        }
    }

    /** First 8 uppercase hex chars of the SHA-256 of the content. */
    static String shortHash(final byte[] content) {
        final MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (final NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        final StringBuilder hex = new StringBuilder();
        for (final byte b : digest.digest(content)) {
            hex.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return hex.substring(0, 8).toUpperCase(Locale.ROOT);
    }
}
