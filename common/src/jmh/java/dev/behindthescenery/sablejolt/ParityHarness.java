package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.ryanhcode.sable.physics.impl.rapier.RapierParityBridge;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import org.joml.Matrix3d;
import org.joml.Quaterniond;
import org.joml.Vector3d;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Consumer;

/**
 * <b>HEADLESS PARITY HARNESS.</b> Runs the same voxel scenes through Sable's own Rapier natives and
 * through {@link JoltPhysicsScene}, with the inputs the two pipelines would upload in game: Sable's
 * neighborhood states, block masses and inertia, gravity, drag, substeps and solver settings.
 * Prints the behaviour metrics of both engines side by side and writes per-tick trajectories to
 * {@code build/parity-run/<scenario>.csv}.
 * <p>
 * Run with {@code gradle :common:parity} (all scenarios) or
 * {@code gradle :common:parity -PparityScenario=slide} (scenarios whose name contains the text).
 */
public final class ParityHarness {

    // Sable 2.0.5 defaults (DimensionPhysics, PhysicsConfigData, SableServerConfig).
    static final double GRAVITY = -11.0;
    static final double DRAG = 0.09;
    static final int SUBSTEPS = 2;
    static final int SOLVER_ITERATIONS = 18;
    static final int PGS_ITERATIONS = 2;
    static final int STABILIZATION_ITERATIONS = 2;
    static final double SPRING_FREQUENCY = 40.0;
    static final double SPRING_DAMPING = 5.0;
    static final int MIN_ISLAND = 128;

    /** Plot area of the test sub-levels: far from the terrain, as Sable's plots are. */
    static final int PLOT_X = 1 << 20;
    static final int PLOT_Y = 64;
    static final int PLOT_Z = 1 << 20;

    // Voxel neighborhood states (VoxelNeighborhoodState ordinals).
    static final int EMPTY = 0, FACE = 1, EDGE = 2, CORNER = 3, INTERIOR = 4;

    private ParityHarness() {
    }

    // region Scene description

    /** A block type: one physics data entry of the collider bakery. */
    record Material(String name, double friction, double volume, double restitution, boolean fluid,
                    double mass, boolean callback, double[]... boxes) {
        boolean full() {
            return !this.fluid && this.boxes.length == 1 && this.boxes[0][0] == 0 && this.boxes[0][1] == 0 && this.boxes[0][2] == 0
                    && this.boxes[0][3] == 1 && this.boxes[0][4] == 1 && this.boxes[0][5] == 1;
        }

        boolean solid() {
            return !this.fluid && this.boxes.length > 0;
        }
    }

    static final double[] CUBE = {0, 0, 0, 1, 1, 1};
    static final Material STONE = new Material("stone", 1.0, 1.0, 0.0, false, 1.0, false, CUBE);
    static final Material DIRT = new Material("dirt", 1.0, 1.0, 0.0, false, 1.0, false, CUBE);
    /** Planks are in Sable's {@code #sable:light} tag (mass 0.5). */
    static final Material PLANKS = new Material("planks", 1.0, 1.0, 0.0, false, 0.5, false, CUBE);
    static final Material ICE = new Material("slippery", 0.0, 1.0, 0.0, false, 1.0, false, CUBE);
    static final Material BOUNCY = new Material("bouncy", 1.0, 1.0, 0.5, false, 1.0, false, CUBE);
    static final Material WATER = new Material("water", 1.0, 1.0, 0.0, true, 0.0, false, CUBE);
    static final Material SLAB = new Material("slab", 1.0, 0.5, 0.0, false, 1.0, false, new double[]{0, 0, 0, 1, 0.5, 1});
    static final Material TOP_SLAB = new Material("top_slab", 1.0, 0.5, 0.0, false, 1.0, false, new double[]{0, 0.5, 0, 1, 1, 1});
    static final Material STAIRS = new Material("stairs", 1.0, 0.75, 0.0, false, 1.0, false,
            new double[]{0, 0, 0, 1, 0.5, 1}, new double[]{0, 0.5, 0.5, 1, 1, 1});

    /** A sparse block grid (terrain in world coordinates, a sub-level in plot coordinates). */
    static final class Grid {
        final Long2ObjectOpenHashMap<Material> blocks = new Long2ObjectOpenHashMap<>();

        static long key(final int x, final int y, final int z) {
            return ((long) (x & 0x3FFFFFF) << 38) | ((long) (z & 0x3FFFFFF) << 12) | (y & 0xFFF);
        }

        static int keyX(final long key) {
            return (int) (key >> 38);
        }

        static int keyY(final long key) {
            return (int) ((key << 52) >> 52);
        }

        static int keyZ(final long key) {
            return (int) ((key << 26) >> 38);
        }

        Material get(final int x, final int y, final int z) {
            return this.blocks.get(key(x, y, z));
        }

        Grid set(final int x, final int y, final int z, final Material material) {
            if (material == null) {
                this.blocks.remove(key(x, y, z));
            } else {
                this.blocks.put(key(x, y, z), material);
            }
            return this;
        }

        Grid fill(final int x0, final int y0, final int z0, final int x1, final int y1, final int z1, final Material material) {
            for (int x = x0; x <= x1; x++) {
                for (int y = y0; y <= y1; y++) {
                    for (int z = z0; z <= z1; z++) {
                        this.set(x, y, z, material);
                    }
                }
            }
            return this;
        }

        boolean fullSolid(final int x, final int y, final int z) {
            final Material m = this.get(x, y, z);
            return m != null && m.solid() && m.full();
        }

        /** {@code VoxelNeighborhoodState.getState} for this grid. */
        int state(final int x, final int y, final int z) {
            final Material m = this.get(x, y, z);
            if (m == null) {
                return EMPTY;
            }
            if (m.fluid() || m.callback()) {
                return CORNER;
            }
            if (!m.solid()) {
                return EMPTY;
            }
            if (!m.full()) {
                return CORNER;
            }
            boolean allSolid = true;
            boolean cornerSolid = true;
            int bothSides = 0;
            for (int axis = 0; axis < 3; axis++) {
                final boolean n = this.fullSolid(x - (axis == 0 ? 1 : 0), y - (axis == 1 ? 1 : 0), z - (axis == 2 ? 1 : 0));
                final boolean p = this.fullSolid(x + (axis == 0 ? 1 : 0), y + (axis == 1 ? 1 : 0), z + (axis == 2 ? 1 : 0));
                if (!n || !p) {
                    allSolid = false;
                }
                if (n && p) {
                    cornerSolid = false;
                    bothSides++;
                }
            }
            if (allSolid) {
                return INTERIOR;
            }
            if (bothSides == 1) {
                return EDGE;
            }
            if (cornerSolid) {
                return CORNER;
            }
            return FACE;
        }

        int minX, minY, minZ, maxX, maxY, maxZ;

        void computeBounds() {
            this.minX = this.minY = this.minZ = Integer.MAX_VALUE;
            this.maxX = this.maxY = this.maxZ = Integer.MIN_VALUE;
            for (final long key : this.blocks.keySet()) {
                final int x = keyX(key), y = keyY(key), z = keyZ(key);
                this.minX = Math.min(this.minX, x);
                this.minY = Math.min(this.minY, y);
                this.minZ = Math.min(this.minZ, z);
                this.maxX = Math.max(this.maxX, x);
                this.maxY = Math.max(this.maxY, y);
                this.maxZ = Math.max(this.maxZ, z);
            }
        }

        /**
         * Packs the grid into 16^3 sections the way the pipelines upload them.
         *
         * @param ids        material to collider id (handle + 1)
         * @param fluidLevel include the fluid level bits (Jolt pipeline only; Rapier reads them as a state)
         */
        Long2ObjectMap<int[]> sections(final Map<Material, Integer> ids, final boolean fluidLevel) {
            final Long2ObjectOpenHashMap<int[]> out = new Long2ObjectOpenHashMap<>();
            for (final Long2ObjectMap.Entry<Material> e : this.blocks.long2ObjectEntrySet()) {
                final int x = keyX(e.getLongKey()), y = keyY(e.getLongKey()), z = keyZ(e.getLongKey());
                final long section = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
                final int[] data = out.computeIfAbsent(section, k -> new int[ChunkSectionData.BLOCKS]);
                final Material m = e.getValue();
                final int level = fluidLevel && m.fluid() ? 8 : 0;
                data[ChunkSectionData.index(x & 15, y & 15, z & 15)] = (this.state(x, y, z) & 0xFF) | (level << 8) | (ids.get(m) << 16);
            }
            return out;
        }
    }

    /** One sub-level: blocks in plot coordinates plus its initial pose and velocity. */
    static final class BodySpec {
        final int id;
        final Grid blocks = new Grid();
        final Vector3d position = new Vector3d();
        final Quaterniond orientation = new Quaterniond();
        final Vector3d linearVelocity = new Vector3d();
        final Vector3d angularVelocity = new Vector3d();

        // MassTracker.build
        double mass;
        final Vector3d com = new Vector3d();
        final Matrix3d inertia = new Matrix3d();

        BodySpec(final int id) {
            this.id = id;
        }

        int plotX() {
            return PLOT_X + this.id * 256;
        }

        /** Adds blocks relative to the plot origin. */
        BodySpec fill(final int x0, final int y0, final int z0, final int x1, final int y1, final int z1, final Material m) {
            this.blocks.fill(this.plotX() + x0, PLOT_Y + y0, PLOT_Z + z0, this.plotX() + x1, PLOT_Y + y1, PLOT_Z + z1, m);
            return this;
        }

        BodySpec set(final int x, final int y, final int z, final Material m) {
            this.blocks.set(this.plotX() + x, PLOT_Y + y, PLOT_Z + z, m);
            return this;
        }

        void computeMass() {
            this.mass = 0.0;
            this.com.zero();
            for (final Long2ObjectMap.Entry<Material> e : this.blocks.blocks.long2ObjectEntrySet()) {
                if (!e.getValue().solid()) {
                    continue;
                }
                final double m = e.getValue().mass();
                this.mass += m;
                this.com.fma(m, new Vector3d(Grid.keyX(e.getLongKey()) + 0.5, Grid.keyY(e.getLongKey()) + 0.5, Grid.keyZ(e.getLongKey()) + 0.5));
            }
            this.com.div(this.mass);
            this.inertia.zero();
            for (final Long2ObjectMap.Entry<Material> e : this.blocks.blocks.long2ObjectEntrySet()) {
                if (!e.getValue().solid()) {
                    continue;
                }
                final double m = e.getValue().mass();
                final double rx = Grid.keyX(e.getLongKey()) + 0.5 - this.com.x;
                final double ry = Grid.keyY(e.getLongKey()) + 0.5 - this.com.y;
                final double rz = Grid.keyZ(e.getLongKey()) + 0.5 - this.com.z;
                this.inertia.m00 += m / 6.0 + (ry * ry + rz * rz) * m;
                this.inertia.m11 += m / 6.0 + (rz * rz + rx * rx) * m;
                this.inertia.m22 += m / 6.0 + (rx * rx + ry * ry) * m;
                this.inertia.m01 -= ry * rx * m;
                this.inertia.m02 -= rz * rx * m;
                this.inertia.m10 -= rx * ry * m;
                this.inertia.m12 -= rz * ry * m;
                this.inertia.m20 -= rx * rz * m;
                this.inertia.m21 -= ry * rz * m;
            }
            this.blocks.computeBounds();
        }

        /**
         * World COM height at which the body's lowest block face touches {@code groundTop} (level pose).
         */
        double restingHeight(final double groundTop) {
            return groundTop + (this.com.y - this.blocks.minY);
        }
    }

    static final class Scenario {
        final String name;
        final String description;
        final Grid terrain = new Grid();
        final List<BodySpec> bodies = new ArrayList<>();
        int ticks = 200;
        /** Runs before each tick with the tick index (impulses, scripted pushes). */
        TickAction action;
        /** Prints the scenario-specific metrics of one run. */
        Metrics metrics = (s, r) -> "";

        Scenario(final String name, final String description) {
            this.name = name;
            this.description = description;
        }

        BodySpec body(final int id) {
            final BodySpec spec = new BodySpec(id);
            this.bodies.add(spec);
            return spec;
        }
    }

    interface TickAction {
        void run(Sim sim, int tick);
    }

    interface Metrics {
        String describe(Scenario scenario, Run run);
    }

    // endregion

    // region Engines

    interface Sim extends AutoCloseable {
        String name();

        void tick();

        void pose(int id, double[] out7);

        void linearVelocity(int id, double[] out3);

        void angularVelocity(int id, double[] out3);

        void addVelocity(int id, double lx, double ly, double lz, double ax, double ay, double az);

        void applyImpulse(int id, double localX, double localY, double localZ, double fx, double fy, double fz);

        @Override
        void close();
    }

    /** Collider handles are global in the Rapier natives: registered once per JVM. */
    private static final Map<Material, Integer> RAPIER_IDS = new IdentityHashMap<>();

    private static int rapierId(final Material m) {
        return RAPIER_IDS.computeIfAbsent(m, k -> RapierParityBridge.newCollider(k.friction(), k.volume(), k.restitution(), k.fluid(), k.boxes()) + 1);
    }

    static final class RapierSim implements Sim {
        private final long scene;

        RapierSim(final Scenario scenario) {
            this.scene = RapierParityBridge.initialize(0.0, GRAVITY, 0.0, DRAG);
            RapierParityBridge.configure(SPRING_FREQUENCY, SPRING_DAMPING, SOLVER_ITERATIONS, PGS_ITERATIONS, STABILIZATION_ITERATIONS, MIN_ISLAND);
            final Map<Material, Integer> ids = new IdentityHashMap<>();
            for (final Material m : materials(scenario)) {
                ids.put(m, rapierId(m));
            }
            for (final Long2ObjectMap.Entry<int[]> e : scenario.terrain.sections(ids, false).long2ObjectEntrySet()) {
                RapierParityBridge.addChunk(this.scene, unpackX(e.getLongKey()), unpackY(e.getLongKey()), unpackZ(e.getLongKey()), e.getValue(), true, -1);
            }
            for (final BodySpec b : scenario.bodies) {
                RapierParityBridge.createSubLevel(this.scene, b.id, new double[]{b.position.x, b.position.y, b.position.z,
                        b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w});
                RapierParityBridge.setCenterOfMass(this.scene, b.id, b.com.x, b.com.y, b.com.z);
                RapierParityBridge.setMassProperties(this.scene, b.id, b.mass, b.com, b.inertia);
                for (final Long2ObjectMap.Entry<int[]> e : b.blocks.sections(ids, false).long2ObjectEntrySet()) {
                    RapierParityBridge.addChunk(this.scene, unpackX(e.getLongKey()), unpackY(e.getLongKey()), unpackZ(e.getLongKey()), e.getValue(), false, -1);
                }
                final Grid g = b.blocks;
                RapierParityBridge.setLocalBounds(this.scene, b.id, g.minX, g.minY, g.minZ, g.maxX, g.maxY, g.maxZ);
                if (b.linearVelocity.lengthSquared() > 0 || b.angularVelocity.lengthSquared() > 0) {
                    RapierParityBridge.addVelocities(this.scene, b.id, b.linearVelocity.x, b.linearVelocity.y, b.linearVelocity.z,
                            b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z);
                }
            }
        }

        @Override
        public String name() {
            return "rapier";
        }

        @Override
        public void tick() {
            RapierParityBridge.tick(this.scene, 1.0 / 20.0);
            for (int s = 0; s < SUBSTEPS; s++) {
                RapierParityBridge.step(this.scene, 1.0 / 20.0 / SUBSTEPS);
            }
            RapierParityBridge.clearCollisions(this.scene);
        }

        @Override
        public void pose(final int id, final double[] out7) {
            RapierParityBridge.getPose(this.scene, id, out7);
        }

        @Override
        public void linearVelocity(final int id, final double[] out3) {
            RapierParityBridge.getLinearVelocity(this.scene, id, out3);
        }

        @Override
        public void angularVelocity(final int id, final double[] out3) {
            RapierParityBridge.getAngularVelocity(this.scene, id, out3);
        }

        @Override
        public void addVelocity(final int id, final double lx, final double ly, final double lz, final double ax, final double ay, final double az) {
            RapierParityBridge.addVelocities(this.scene, id, lx, ly, lz, ax, ay, az);
        }

        @Override
        public void applyImpulse(final int id, final double localX, final double localY, final double localZ, final double fx, final double fy, final double fz) {
            RapierParityBridge.applyImpulse(this.scene, id, localX, localY, localZ, fx, fy, fz);
        }

        @Override
        public void close() {
            RapierParityBridge.dispose(this.scene);
        }
    }

    static final class JoltSim implements Sim {
        final JoltPhysicsScene scene;
        private final JoltPhysicsScene.PoseCache cache = new JoltPhysicsScene.PoseCache();

        JoltSim(final Scenario scenario) {
            BenchSupport.boot();
            this.scene = new JoltPhysicsScene(0.0, GRAVITY, 0.0, DRAG);
            this.scene.configSolverIterations(SOLVER_ITERATIONS, PGS_ITERATIONS);
            final Map<Material, Integer> ids = new IdentityHashMap<>();
            for (final Material m : materials(scenario)) {
                final JoltVoxelColliderData data = this.scene.colliderRegistry.create(m.friction(), m.volume(), m.restitution(), m.fluid(), null, null);
                for (final double[] box : m.boxes()) {
                    data.addBox(box[0], box[1], box[2], box[3], box[4], box[5]);
                }
                ids.put(m, data.colliderHandle + 1);
            }
            for (final Long2ObjectMap.Entry<int[]> e : scenario.terrain.sections(ids, true).long2ObjectEntrySet()) {
                this.scene.addChunk(unpackX(e.getLongKey()), unpackY(e.getLongKey()), unpackZ(e.getLongKey()), e.getValue(), true, -1);
            }
            for (final BodySpec b : scenario.bodies) {
                this.scene.createSubLevel(b.id, b.position, b.orientation);
                this.scene.setCenterOfMass(b.id, b.com.x, b.com.y, b.com.z);
                this.scene.setMassProperties(b.id, b.mass, b.inertia);
                for (final Long2ObjectMap.Entry<int[]> e : b.blocks.sections(ids, true).long2ObjectEntrySet()) {
                    this.scene.addChunk(unpackX(e.getLongKey()), unpackY(e.getLongKey()), unpackZ(e.getLongKey()), e.getValue(), false, b.id);
                }
                final Grid g = b.blocks;
                this.scene.setLocalBounds(b.id, g.minX, g.minY, g.minZ, g.maxX, g.maxY, g.maxZ);
                if (b.linearVelocity.lengthSquared() > 0 || b.angularVelocity.lengthSquared() > 0) {
                    this.scene.addLinearAngularVelocities(b.id, b.linearVelocity.x, b.linearVelocity.y, b.linearVelocity.z,
                            b.angularVelocity.x, b.angularVelocity.y, b.angularVelocity.z, true);
                }
            }
        }

        @Override
        public String name() {
            return "jolt";
        }

        @Override
        public void tick() {
            this.scene.prePhysicsTick();
            for (int s = 0; s < SUBSTEPS; s++) {
                this.scene.step(1.0 / 20.0 / SUBSTEPS);
                this.scene.processDeferredCallbacks();
            }
            this.scene.clearCollisions();
        }

        @Override
        public void pose(final int id, final double[] out7) {
            this.scene.getPose(id, this.cache);
            out7[0] = this.cache.p1;
            out7[1] = this.cache.p2;
            out7[2] = this.cache.p3;
            out7[3] = this.cache.p4;
            out7[4] = this.cache.p5;
            out7[5] = this.cache.p6;
            out7[6] = this.cache.p7;
        }

        @Override
        public void linearVelocity(final int id, final double[] out3) {
            this.scene.getLinearVelocity(id, this.cache);
            out3[0] = this.cache.p1;
            out3[1] = this.cache.p2;
            out3[2] = this.cache.p3;
        }

        @Override
        public void angularVelocity(final int id, final double[] out3) {
            this.scene.getAngularVelocity(id, this.cache);
            out3[0] = this.cache.p1;
            out3[1] = this.cache.p2;
            out3[2] = this.cache.p3;
        }

        @Override
        public void addVelocity(final int id, final double lx, final double ly, final double lz, final double ax, final double ay, final double az) {
            this.scene.addLinearAngularVelocities(id, lx, ly, lz, ax, ay, az, true);
        }

        @Override
        public void applyImpulse(final int id, final double localX, final double localY, final double localZ, final double fx, final double fy, final double fz) {
            this.scene.applyForce(id, localX, localY, localZ, fx, fy, fz, true);
        }

        @Override
        public void close() {
            this.scene.dispose();
        }
    }

    static List<Material> materials(final Scenario scenario) {
        final List<Material> out = new ArrayList<>();
        final Consumer<Material> add = m -> {
            if (!out.contains(m)) {
                out.add(m);
            }
        };
        scenario.terrain.blocks.values().forEach(add);
        for (final BodySpec b : scenario.bodies) {
            b.blocks.blocks.values().forEach(add);
        }
        return out;
    }

    static int unpackX(final long key) {
        return (int) (key >> 42);
    }

    static int unpackY(final long key) {
        return (int) ((key << 44) >> 44);
    }

    static int unpackZ(final long key) {
        return (int) ((key << 22) >> 42);
    }

    // endregion

    // region Running

    /** Per-tick samples of every body of one engine run. */
    static final class Run {
        final String engine;
        final int bodies;
        final int ticks;
        /** [tick][body][13]: position, orientation (x, y, z, w), linear velocity, angular velocity. */
        final double[][][] samples;
        long nanos;

        Run(final String engine, final int ticks, final int bodies) {
            this.engine = engine;
            this.ticks = ticks;
            this.bodies = bodies;
            this.samples = new double[ticks + 1][bodies][13];
        }

        double[] at(final int tick, final int body) {
            return this.samples[Math.min(tick, this.ticks)][body];
        }

        double[] last(final int body) {
            return this.samples[this.ticks][body];
        }

        double max(final int body, final java.util.function.ToDoubleFunction<double[]> f) {
            double m = Double.NEGATIVE_INFINITY;
            for (int t = 0; t <= this.ticks; t++) {
                m = Math.max(m, f.applyAsDouble(this.samples[t][body]));
            }
            return m;
        }

        double min(final int body, final java.util.function.ToDoubleFunction<double[]> f) {
            double m = Double.POSITIVE_INFINITY;
            for (int t = 0; t <= this.ticks; t++) {
                m = Math.min(m, f.applyAsDouble(this.samples[t][body]));
            }
            return m;
        }

        /** First tick from which the body stays below the given speed (or -1). */
        int settleTick(final int body, final double speed) {
            int settled = -1;
            for (int t = 0; t <= this.ticks; t++) {
                final double[] s = this.samples[t][body];
                final double v = Math.sqrt(s[7] * s[7] + s[8] * s[8] + s[9] * s[9]) + Math.sqrt(s[10] * s[10] + s[11] * s[11] + s[12] * s[12]);
                if (v < speed) {
                    if (settled < 0) {
                        settled = t;
                    }
                } else {
                    settled = -1;
                }
            }
            return settled;
        }
    }

    static Run run(final Scenario scenario, final Sim sim) {
        final Run run = new Run(sim.name(), scenario.ticks, scenario.bodies.size());
        final long start = System.nanoTime();
        sample(scenario, sim, run, 0);
        for (int t = 1; t <= scenario.ticks; t++) {
            if (scenario.action != null) {
                scenario.action.run(sim, t);
            }
            sim.tick();
            sample(scenario, sim, run, t);
        }
        run.nanos = System.nanoTime() - start;
        return run;
    }

    private static void sample(final Scenario scenario, final Sim sim, final Run run, final int tick) {
        final double[] pose = new double[7];
        final double[] v = new double[3];
        for (int b = 0; b < scenario.bodies.size(); b++) {
            final int id = scenario.bodies.get(b).id;
            final double[] out = run.samples[tick][b];
            sim.pose(id, pose);
            System.arraycopy(pose, 0, out, 0, 7);
            sim.linearVelocity(id, v);
            System.arraycopy(v, 0, out, 7, 3);
            sim.angularVelocity(id, v);
            System.arraycopy(v, 0, out, 10, 3);
        }
    }

    /** Rotation angle (degrees) between two orientations. */
    static double angleBetween(final double[] a, final double[] b) {
        final double dot = Math.abs(a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6]);
        return Math.toDegrees(2.0 * Math.acos(Math.min(1.0, dot)));
    }

    /** Tilt (degrees) of the body's up axis from world up. */
    static double tilt(final double[] s) {
        final Quaterniond q = new Quaterniond(s[3], s[4], s[5], s[6]);
        final Vector3d up = q.transform(new Vector3d(0, 1, 0));
        return Math.toDegrees(Math.acos(Math.max(-1.0, Math.min(1.0, up.y))));
    }

    static double speed(final double[] s) {
        return Math.sqrt(s[7] * s[7] + s[8] * s[8] + s[9] * s[9]);
    }

    static double spin(final double[] s) {
        return Math.sqrt(s[10] * s[10] + s[11] * s[11] + s[12] * s[12]);
    }

    static void writeCsv(final Path dir, final Scenario scenario, final Run rapier, final Run jolt) throws IOException {
        Files.createDirectories(dir);
        try (final PrintWriter out = new PrintWriter(Files.newBufferedWriter(dir.resolve(scenario.name + ".csv")))) {
            out.println("tick,body,engine,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz");
            for (int t = 0; t <= scenario.ticks; t++) {
                for (int b = 0; b < scenario.bodies.size(); b++) {
                    for (final Run r : new Run[]{rapier, jolt}) {
                        final double[] s = r.samples[t][b];
                        final StringBuilder line = new StringBuilder().append(t).append(',').append(b).append(',').append(r.engine);
                        for (final double d : s) {
                            line.append(',').append(String.format(Locale.ROOT, "%.5f", d));
                        }
                        out.println(line);
                    }
                }
            }
        }
    }

    public static void main(final String[] args) throws Exception {
        String filter = "";
        for (final String arg : args) {
            final int eq = arg.indexOf('=');
            if (eq > 0) {
                // key=value: tuning property for experiments
                System.setProperty(arg.substring(0, eq), arg.substring(eq + 1));
            } else {
                filter = arg;
            }
        }
        final Path out = Paths.get(System.getProperty("parity.out", "parity-results"));
        final List<Scenario> all = ParityScenarios.all();
        System.out.printf(Locale.ROOT, "Sable defaults: g=%.1f drag=%.2f substeps=%d solver=%d/%d/%d springs=%.0fHz/%.1f%n",
                GRAVITY, DRAG, SUBSTEPS, SOLVER_ITERATIONS, PGS_ITERATIONS, STABILIZATION_ITERATIONS, SPRING_FREQUENCY, SPRING_DAMPING);
        for (final Scenario scenario : all) {
            if (!scenario.name.contains(filter)) {
                continue;
            }
            for (final BodySpec b : scenario.bodies) {
                b.computeMass();
            }
            final Run rapier;
            try (final RapierSim sim = new RapierSim(scenario)) {
                rapier = run(scenario, sim);
            }
            final Run jolt;
            try (final JoltSim sim = new JoltSim(scenario)) {
                jolt = run(scenario, sim);
            }
            writeCsv(out, scenario, rapier, jolt);
            report(scenario, rapier, jolt);
        }
        System.exit(0);
    }

    static void report(final Scenario scenario, final Run rapier, final Run jolt) {
        System.out.println();
        System.out.printf(Locale.ROOT, "== %s: %s (%d ticks)%n", scenario.name, scenario.description, scenario.ticks);
        for (int b = 0; b < scenario.bodies.size(); b++) {
            final double[] r = rapier.last(b);
            final double[] j = jolt.last(b);
            double maxDev = 0.0;
            for (int t = 0; t <= scenario.ticks; t++) {
                final double[] a = rapier.samples[t][b], c = jolt.samples[t][b];
                maxDev = Math.max(maxDev, Math.sqrt(sq(a[0] - c[0]) + sq(a[1] - c[1]) + sq(a[2] - c[2])));
            }
            System.out.printf(Locale.ROOT, "  body %d  final pos   rapier (%.3f, %.3f, %.3f)  jolt (%.3f, %.3f, %.3f)  |d|=%.3f  max|d|=%.3f  rot diff %.1f deg%n",
                    scenario.bodies.get(b).id, r[0], r[1], r[2], j[0], j[1], j[2],
                    Math.sqrt(sq(r[0] - j[0]) + sq(r[1] - j[1]) + sq(r[2] - j[2])), maxDev, angleBetween(r, j));
        }
        final String rm = scenario.metrics.describe(scenario, rapier);
        final String jm = scenario.metrics.describe(scenario, jolt);
        if (!rm.isEmpty()) {
            System.out.println("  rapier: " + rm);
            System.out.println("  jolt:   " + jm);
        }
        System.out.printf(Locale.ROOT, "  time: rapier %.1f ms, jolt %.1f ms%n", rapier.nanos / 1e6, jolt.nanos / 1e6);
    }

    static double sq(final double v) {
        return v * v;
    }

    // endregion
}
