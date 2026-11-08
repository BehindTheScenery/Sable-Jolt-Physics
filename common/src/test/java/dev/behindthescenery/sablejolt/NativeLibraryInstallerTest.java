package dev.behindthescenery.sablejolt;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class NativeLibraryInstallerTest {
    private static final byte[] LIBRARY = "the library".getBytes();
    private static final byte[] DAMAGED = "the libXXXX".getBytes();

    @TempDir
    Path temp;

    private final List<String> loaded = new ArrayList<>();
    private final NativeLibraryInstaller.Loader recording = loaded::add;

    private Path versioned(final Path base, final byte[] content) {
        return base.resolve(NativeLibraryInstaller.shortHash(content)).resolve("lib.dll");
    }

    @Test
    void writesTheLibraryAndLoadsIt() throws IOException {
        final Path base = temp.resolve("natives");

        final Path target = NativeLibraryInstaller.install(List.of(base), "lib.dll", LIBRARY, recording);

        assertEquals(versioned(base, LIBRARY), target);
        assertArrayEquals(LIBRARY, Files.readAllBytes(target));
        assertEquals(List.of(target.toAbsolutePath().toString()), loaded);
        try (var files = Files.list(target.getParent())) {
            assertEquals(1, files.count(), "no temporary file may stay behind");
        }
    }

    @Test
    void leavesAGoodCopyAlone() throws IOException {
        final Path base = temp.resolve("natives");
        final Path target = NativeLibraryInstaller.install(List.of(base), "lib.dll", LIBRARY, recording);
        final FileTime old = FileTime.fromMillis(1_000_000_000_000L);
        Files.setLastModifiedTime(target, old);

        NativeLibraryInstaller.install(List.of(base), "lib.dll", LIBRARY, recording);

        assertEquals(old, Files.getLastModifiedTime(target), "the file must not be rewritten");
    }

    @Test
    void replacesADamagedCopyOfTheSameSize() throws IOException {
        final Path base = temp.resolve("natives");
        final Path target = versioned(base, LIBRARY);
        Files.createDirectories(target.getParent());
        Files.write(target, DAMAGED);
        assertEquals(LIBRARY.length, DAMAGED.length, "same size is the point");

        NativeLibraryInstaller.install(List.of(base), "lib.dll", LIBRARY, recording);

        assertArrayEquals(LIBRARY, Files.readAllBytes(target));
    }

    @Test
    void differentVersionsDoNotMeet() throws IOException {
        final Path base = temp.resolve("natives");
        final Path one = NativeLibraryInstaller.install(List.of(base), "lib.dll", LIBRARY, recording);
        final Path two = NativeLibraryInstaller.install(List.of(base), "lib.dll", DAMAGED, recording);

        assertNotEquals(one, two);
        assertArrayEquals(LIBRARY, Files.readAllBytes(one));
        assertArrayEquals(DAMAGED, Files.readAllBytes(two));
    }

    @Test
    void movesOnWhenALocationCannotBeWritten() throws IOException {
        final Path blocked = temp.resolve("blocked");
        Files.writeString(blocked, "a file where the folder should be");
        final Path second = temp.resolve("second");

        final Path target = NativeLibraryInstaller.install(List.of(blocked, second), "lib.dll", LIBRARY, recording);

        assertEquals(versioned(second, LIBRARY), target);
        assertEquals(List.of(target.toAbsolutePath().toString()), loaded);
    }

    @Test
    void movesOnWhenALocationCannotBeLoadedFrom() throws IOException {
        final Path first = temp.resolve("first");
        final Path second = temp.resolve("second");

        final Path target = NativeLibraryInstaller.install(List.of(first, second), "lib.dll", LIBRARY, path -> {
            if (path.startsWith(first.toAbsolutePath().toString())) {
                throw new UnsatisfiedLinkError("Access is denied");
            }
            loaded.add(path);
        });

        assertEquals(versioned(second, LIBRARY), target);
    }

    @Test
    void reportsEveryLocationWhenNoneWorks() {
        final IOException e = assertThrows(IOException.class, () -> NativeLibraryInstaller.install(
                List.of(temp.resolve("a"), temp.resolve("b")), "lib.dll", LIBRARY, path -> {
                    throw new UnsatisfiedLinkError("no");
                }));

        assertEquals(2, e.getSuppressed().length);
    }

    /**
     * The player's crash: another process has the DLL loaded. Sable's delete fails there, so this first checks
     * that the situation really is the one it claims to be, then that installing and loading still work.
     */
    @Test
    @EnabledOnOs(OS.WINDOWS)
    void worksWhileAnotherProcessHasTheLibraryLoaded() throws Exception {
        final Path base = temp.resolve("natives");
        final byte[] dll = Files.readAllBytes(Path.of(System.getProperty("java.home"), "bin", "verify.dll"));
        final Path target = versioned(base, dll).resolveSibling("verify.dll");
        NativeLibraryInstaller.install(List.of(base), "verify.dll", dll, recording);

        final Process holder = holdLoaded(target);
        Process second = null;
        try {
            assertThrows(AccessDeniedException.class, () -> Files.delete(target), "the DLL should be locked");
            final FileTime before = Files.getLastModifiedTime(target);

            final Path again = NativeLibraryInstaller.install(List.of(base), "verify.dll", dll, recording);

            assertEquals(target, again);
            assertEquals(before, Files.getLastModifiedTime(target));
            // A real System.load of that path works at the same time. It runs in a JVM of its own: a library this
            // JVM loads stays locked until the JVM exits, and the test's temporary folder could not be removed.
            second = holdLoaded(again);
        } finally {
            if (second != null) {
                second.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
            }
            release(holder, target);
        }
    }

    /** A damaged copy that cannot be replaced because something holds it: the next location takes over. */
    @Test
    @EnabledOnOs(OS.WINDOWS)
    void movesOnWhenADamagedCopyIsLocked() throws Exception {
        final Path base = temp.resolve("natives");
        final Path second = temp.resolve("second");
        final byte[] held = Files.readAllBytes(Path.of(System.getProperty("java.home"), "bin", "verify.dll"));
        final Path target = versioned(base, LIBRARY);
        Files.createDirectories(target.getParent());
        Files.write(target, held);

        final Process holder = holdLoaded(target);
        try {
            final Path used = NativeLibraryInstaller.install(List.of(base, second), "lib.dll", LIBRARY, recording);

            assertEquals(versioned(second, LIBRARY), used);
        } finally {
            release(holder, target);
        }
    }

    /** Ends the holder and removes the file, which Windows can keep locked for a moment after the process ended. */
    private static void release(final Process holder, final Path dll) throws Exception {
        holder.destroyForcibly().waitFor(10, TimeUnit.SECONDS);
        final long ended = System.nanoTime();
        while (true) {
            try {
                Files.deleteIfExists(dll);
                System.out.printf("%s unlocked %d ms after its holder ended%n", dll.getFileName(),
                        TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - ended));
                return;
            } catch (final IOException e) {
                if (System.nanoTime() - ended > TimeUnit.SECONDS.toNanos(15)) {
                    throw e;
                }
                Thread.sleep(20);
            }
        }
    }

    private Process holdLoaded(final Path dll) throws Exception {
        final Path source = temp.resolve("Hold.java");
        Files.writeString(source, """
                public class Hold {
                    public static void main(String[] a) throws Exception {
                        System.load(a[0]);
                        System.out.println("READY");
                        System.out.flush();
                        Thread.sleep(120000);
                    }
                }
                """);
        final Process holder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                source.toString(), dll.toAbsolutePath().toString()).redirectErrorStream(true).start();
        final BufferedReader out = new BufferedReader(new InputStreamReader(holder.getInputStream()));
        final String line = out.readLine();
        assertEquals("READY", line, "the holding JVM did not start");
        return holder;
    }
}
