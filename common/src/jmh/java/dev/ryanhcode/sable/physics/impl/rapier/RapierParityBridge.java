package dev.ryanhcode.sable.physics.impl.rapier;

import dev.ryanhcode.sable.api.physics.mass.MassData;
import org.jetbrains.annotations.Nullable;
import org.joml.Matrix3d;
import org.joml.Matrix3dc;
import org.joml.Vector3d;
import org.joml.Vector3dc;

/**
 * <b>HEADLESS PARITY SCAFFOLDING.</b> Public entry points into Sable's own Rapier natives (the
 * package-private statics of {@link Rapier3D}), so the parity harness can run the real Rapier
 * pipeline next to the Jolt scene on identical voxel data. Lives in Rapier3D's package for access;
 * compiled into the {@code jmh} source set only.
 */
public final class RapierParityBridge {
    private RapierParityBridge() {
    }

    public static long initialize(final double gx, final double gy, final double gz, final double universalDrag) {
        return Rapier3D.initialize(gx, gy, gz, universalDrag);
    }

    /**
     * The values {@code RapierPhysicsPipeline.updateConfigFrom} passes for Sable's defaults.
     */
    public static void configure(final double springFrequency, final double springDamping,
                                 final int solverIterations, final int pgsIterations, final int stabilizationIterations,
                                 final int minIslandSize) {
        Rapier3D.configFrequencyAndDamping(springFrequency, springDamping);
        Rapier3D.configSolverIterations(solverIterations, pgsIterations, stabilizationIterations);
        Rapier3D.configMinIslandSize(minIslandSize);
    }

    /**
     * Registers a voxel collider entry and returns its handle (the packed id is {@code handle + 1}).
     */
    public static int newCollider(final double friction, final double volume, final double restitution, final boolean fluid,
                                  final double[][] boxes) {
        final int handle = Rapier3D.createVoxelColliderEntry(friction, volume, restitution, fluid, null).handle();
        for (final double[] box : boxes) {
            Rapier3D.addVoxelColliderBox(handle, box);
        }
        return handle;
    }

    public static void createSubLevel(final long scene, final int id, final double[] pose) {
        Rapier3D.createSubLevel(scene, id, pose);
    }

    public static void removeSubLevel(final long scene, final int id) {
        Rapier3D.removeSubLevel(scene, id);
    }

    public static void setCenterOfMass(final long scene, final int id, final double x, final double y, final double z) {
        Rapier3D.setCenterOfMass(scene, id, x, y, z);
    }

    public static void setLocalBounds(final long scene, final int id, final int minX, final int minY, final int minZ,
                                      final int maxX, final int maxY, final int maxZ) {
        Rapier3D.setLocalBounds(scene, id, minX, minY, minZ, maxX, maxY, maxZ);
    }

    public static void setMassProperties(final long scene, final int id, final double mass, final Vector3dc com, final Matrix3dc inertia) {
        Rapier3D.setMassPropertiesFrom(scene, id, new FixedMass(mass, com, inertia));
    }

    public static void addChunk(final long scene, final int x, final int y, final int z, final int[] data, final boolean global, final int objectId) {
        Rapier3D.addChunk(scene, x, y, z, data, global, objectId);
    }

    public static void changeBlock(final long scene, final int x, final int y, final int z, final int packed) {
        Rapier3D.changeBlock(scene, x, y, z, packed);
    }

    public static void tick(final long scene, final double timeStep) {
        Rapier3D.tick(scene, timeStep);
    }

    public static void step(final long scene, final double timeStep) {
        Rapier3D.step(scene, timeStep);
    }

    public static void getPose(final long scene, final int id, final double[] store) {
        Rapier3D.getPose(scene, id, store);
    }

    public static void getLinearVelocity(final long scene, final int id, final double[] store) {
        Rapier3D.getLinearVelocity(scene, id, store);
    }

    public static void getAngularVelocity(final long scene, final int id, final double[] store) {
        Rapier3D.getAngularVelocity(scene, id, store);
    }

    public static void addVelocities(final long scene, final int id, final double lx, final double ly, final double lz,
                                     final double ax, final double ay, final double az) {
        Rapier3D.addLinearAngularVelocities(scene, id, lx, ly, lz, ax, ay, az, true);
    }

    public static void applyImpulse(final long scene, final int id, final double x, final double y, final double z,
                                    final double fx, final double fy, final double fz) {
        Rapier3D.applyForce(scene, id, x, y, z, fx, fy, fz, true);
    }

    public static double[] clearCollisions(final long scene) {
        return Rapier3D.clearCollisions(scene);
    }

    public static void dispose(final long scene) {
        Rapier3D.dispose(scene);
    }

    private record FixedMass(double mass, Vector3dc com, Matrix3dc inertia) implements MassData {
        @Override
        public double getMass() {
            return this.mass;
        }

        @Override
        public double getInverseMass() {
            return 1.0 / this.mass;
        }

        @Override
        public Matrix3dc getInertiaTensor() {
            return this.inertia;
        }

        @Override
        public Matrix3dc getInverseInertiaTensor() {
            return new Matrix3d(this.inertia).invert();
        }

        @Override
        public @Nullable Vector3dc getCenterOfMass() {
            return new Vector3d(this.com);
        }
    }
}
