package dev.behindthescenery.sableJoltPhysics.mixin;

import dev.behindthescenery.sablejolt.NativeLibraryInstaller;
import dev.ryanhcode.sable.Sable;
import dev.ryanhcode.sable.physics.impl.rapier.Rapier3D;
import net.jpountz.lz4.LZ4FrameInputStream;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Sable deletes and rewrites {@code .sable/natives/<library>} on every start. On Windows that fails with
 * AccessDeniedException as soon as anything holds the file, most often another game process that has the DLL
 * loaded, and the failure takes the whole world down with a crash report: the player cannot enter singleplayer.
 * <p>
 * The library itself is fine, and loading it from two processes is too; only the delete is the problem. So this
 * loads the same bundled library through {@link NativeLibraryInstaller}. If anything about the bundle is not as
 * expected (a future Sable, say), Sable's own loader still runs.
 */
@Mixin(value = Rapier3D.class, remap = false)
public abstract class Rapier3DMixin {
    @Shadow
    @Final
    private static Path NATIVE_DIR;

    @Shadow
    @Final
    public static String NATIVE_NAME;

    @Inject(method = "loadLibrary", at = @At("HEAD"), cancellable = true, remap = false, require = 0)
    private static void sablejolt$loadWithoutDeleting(final CallbackInfo ci) {
        final Path loaded;
        try {
            final byte[] library = sablejolt$readBundledLibrary(NATIVE_NAME);
            // The lambda becomes part of Rapier3D, so the library binds to Rapier3D's own class loader.
            loaded = NativeLibraryInstaller.install(NATIVE_DIR, NATIVE_NAME, library, path -> System.load(path));
        } catch (final IOException | UnsatisfiedLinkError | RuntimeException e) {
            Sable.LOGGER.warn("[SableJolt] Could not load the Rapier natives safely; using Sable's own loader", e);
            return;
        }
        Sable.LOGGER.info("[SableJolt] Loaded the Rapier natives from {}", loaded);
        ci.cancel();
        // What older runs extracted is unused now. Still held by another game? Then a later start removes it.
        try {
            Files.deleteIfExists(NATIVE_DIR.resolve(NATIVE_NAME));
        } catch (final IOException | RuntimeException ignored) {
        }
    }

    private static byte[] sablejolt$readBundledLibrary(final String name) throws IOException {
        try (final InputStream bundle = Rapier3D.class.getResourceAsStream("/natives/sable_rapier/sable_rapier_binaries.zip.l4z")) {
            if (bundle == null) {
                throw new FileNotFoundException("sable_rapier_binaries.zip.l4z");
            }
            try (final ZipInputStream zip = new ZipInputStream(new LZ4FrameInputStream(bundle))) {
                ZipEntry entry;
                while ((entry = zip.getNextEntry()) != null) {
                    if (entry.getName().equals(name)) {
                        return zip.readAllBytes();
                    }
                }
            }
        }
        throw new FileNotFoundException(name);
    }
}
