package dev.behindthescenery.sablejolt;

import com.github.stephengold.joltjni.*;
import com.github.stephengold.joltjni.enumerate.EActivation;
import com.github.stephengold.joltjni.enumerate.EMotionQuality;
import com.github.stephengold.joltjni.enumerate.EMotionType;
import com.github.stephengold.joltjni.enumerate.EOverrideMassProperties;
import com.github.stephengold.joltjni.enumerate.EPhysicsUpdateError;
import com.github.stephengold.joltjni.enumerate.ESpringMode;
import com.github.stephengold.joltjni.readonly.ConstBodyIdArray;
import com.github.stephengold.joltjni.readonly.ConstShape;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderBakery;
import dev.behindthescenery.sablejolt.collider.JoltVoxelColliderData;
import dev.ryanhcode.sable.Sable;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.longs.Long2IntOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import it.unimi.dsi.fastutil.objects.ReferenceOpenHashSet;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.*;

import java.lang.Math;
import java.lang.Runtime;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntConsumer;

/**
 * A Jolt physics scene: the direct equivalent of the native rapier {@code PhysicsScene}.
 * <p>
 * Every sub-level is a dynamic body whose compound shape is built from voxel
 * chunk data, one static compound per 16³ section under a small mutable parent
 * (sections are rebuilt individually when their blocks change); its full cubes
 * are merged into large boxes. The global world is one static body per terrain
 * region of 1024 x 1024 blocks, made of tiles of sections (see
 * {@link #TILE_SHIFT}): a surface mesh of the full cubes' exposed faces whose
 * triangles share every vertex with their neighbours (see
 * {@link #buildSurfaceMesh}), recessed boxes under it, and boxes for other
 * blocks; blocks with contact callbacks form a second body per region (see
 * {@link #LAYER_STATIC_SPECIAL}). Seam ("ghost") contacts are removed by Jolt's
 * enhanced internal edge removal: the Rapier pipeline gets the same from its
 * neighborhood states and interior collision test. Kinematic contraptions
 * mounted on a sub-level are part of the sub-level's compound, as in Rapier.
 * <p>
 * {@code gradle :common:parity} compares this scene with Sable's own Rapier
 * natives on a set of scenes (src/jmh, ParityHarness).
 */
@ApiStatus.Internal
public final class JoltPhysicsScene {
    public static final int LAYER_MOVING = 0;
    public static final int LAYER_STATIC = 1;
    public static final int LAYER_ROPE = 2;
    /**
     * Moving bodies whose contacts are forwarded to the Java contact listener:
     * bodies with special blocks or mounted contraptions, dragged bodies and
     * bodies fast enough for an impact report. Collides exactly like
     * {@link #LAYER_MOVING}; the split only saves Java upcalls, which cost
     * ~80% of a step when thousands of bodies move.
     */
    public static final int LAYER_MOVING_REPORT = 3;
    /**
     * Terrain blocks with a contact callback: the second static body of each
     * terrain region. Collides like {@link #LAYER_STATIC}, and every contact
     * with it reaches the contact listener whatever the moving body's layer.
     * That body keeps one contact manifold per block pair (no manifold
     * reduction), so every block gets its own callback, as Rapier's hooks do
     * per voxel pair (hooks.rs), and a callback can cancel or move its own
     * contact without touching the neighbouring blocks'.
     */
    public static final int LAYER_STATIC_SPECIAL = 4;
    private static final int NUM_OBJECT_LAYERS = 5;

    /**
     * Penetration Jolt leaves uncorrected: Rapier's {@code normalized_allowed_linear_error}
     * (lib.rs), so bodies rest as deep in the ground as with Rapier.
     */
    private static final float ALLOWED_PENETRATION = 0.0025f;
    /**
     * Distance at which contacts are created ahead of touching: Rapier's
     * {@code normalized_prediction_distance} (lib.rs) instead of Jolt's 2 cm.
     * Ahead-of-touch contacts with the edge of the next block tilt the contact
     * normal towards the sliding body (bumps at block seams), and catch only part
     * of a voxel face that lands at an angle: a rolling block wheel then lands
     * on the middle of its next face, which stops its fall but not its spin.
     */
    private static final float SPECULATIVE_CONTACT_DISTANCE = 0.005f;
    /**
     * Fraction of a body's inner radius it must move in a substep before a swept
     * body casts, and the fraction it may end up inside what it hit.
     */
    private static final float LINEAR_CAST_THRESHOLD = 0.2f;
    private static final float LINEAR_CAST_MAX_PENETRATION = 0.05f;

    /**
     * Terrain sections are grouped into tiles of 4x4x4 sections, and tiles into
     * regions of 64x64 sections (all heights): one static body per region. Jolt
     * removes the contacts with internal edges ("ghost" contacts at the seams
     * between blocks, which trip sliding bodies) only within one body pair, so
     * the seams between sections must not be body boundaries. Rapier uses one
     * collider for the whole world and drops such contacts itself (dispatcher.rs,
     * is_interior_collision). A region spans 1024 blocks around its origin, so
     * sub-shape offsets stay within 512 blocks (exact in float) and sub-shape
     * ids within 30 bits; region boundaries lie at 512 + 1024 * k, away from
     * spawn.
     */
    private static final int TILE_SHIFT = 2;
    private static final int REGION_SECTION_SHIFT = 6;
    private static final int REGION_HALF_SECTIONS = 1 << (REGION_SECTION_SHIFT - 1);

    /**
     * Terrain layers: every block without a contact callback (one region body
     * with surface meshes, recessed boxes and the boxes of partial blocks), and
     * blocks with a contact callback (a second region body, see
     * {@link #LAYER_STATIC_SPECIAL}).
     */
    static final int TERRAIN_PLAIN = 0;
    static final int TERRAIN_SPECIAL = 1;

    /**
     * Inset of the merged boxes under the surface mesh of plain terrain. The
     * mesh carries every contact of a body resting or sliding on the terrain;
     * the boxes only hold bodies pushed deeper than this (and stay out of
     * speculative contact range of resting bodies).
     */
    private static final float TERRAIN_RECESS = 0.02f;
    /**
     * Triangles per surface mesh: bigger sections get several meshes, which
     * keeps the sub-shape ids of a mesh within 17 bits.
     */
    private static final int MAX_MESH_TRIANGLES = 4096;

    /**
     * Terrain sections get their surface mesh only near awake bodies (a mesh
     * costs about ten times the build time of merged boxes, and depends on its
     * neighbours): within this many blocks of a body's bounds, stretched along
     * its velocity by the distance it covers in {@link #MESH_LOOKAHEAD}
     * seconds. At most {@link #MESH_UPGRADES_PER_FLUSH} sections are meshed per
     * step, nearest first; sections stay meshed for {@link #MESH_KEEP_STEPS}
     * substeps after the last body left.
     */
    private static final double MESH_MARGIN = 12.0;
    private static final double MESH_LOOKAHEAD = 0.75;
    private static final double MESH_MAX_LOOKAHEAD = 96.0;
    private static final int MESH_UPGRADES_PER_FLUSH = 32;
    private static final long MESH_KEEP_STEPS = 1200L;

    /**
     * Upper bound of the speed (m/s) below which a body takes no part in an
     * impact report: a report needs a relative normal speed of 25 * dt (the
     * pipeline's 25 * minMass force threshold, 0.625 m/s at 2 substeps), see
     * {@link #reportSpeed}.
     */
    private static final float REPORT_SPEED = 0.55f;

    /**
     * Bitmask of all DOFs, used when overriding mass properties.
     */
    private static final int ALLOWED_DOFS_ALL = 0x3F;

    /**
     * Density of fluids in per-block mass units for the Jolt buoyancy model.
     * Buoyancy is gravity-proportional (F = density * submergedVolume * |g|),
     * so a body floats when its average block mass is below this value and
     * sinks when it is above.
     */
    private static final double FLUID_DENSITY = 2.0;

    /**
     * Upward force per submerged block volume of the Rapier buoyancy model
     * (buoyancy.rs), independent of gravity.
     */
    private static final double RAPIER_FLOAT_FORCE = 10.5;

    /**
     * Fluid drag coefficient (both buoyancy models).
     */
    private static final double FLUID_DRAG = 1.7;

    /**
     * Friction of the Rapier colliders (lib.rs / contraptions.rs / boxes.rs /
     * rope.rs; the level collider uses Rapier's default 0.5).
     */
    private static final float FRICTION_SUB_LEVEL = 0.525f;
    private static final float FRICTION_CONTRAPTION = 0.45f;
    private static final float FRICTION_BOX = 0.45f;
    private static final float FRICTION_ROPE = 0.15f;
    private static final float FRICTION_TERRAIN_RAPIER = 0.5f;

    /**
     * Speed limits of sub-level bodies in Rapier-compatible mode (Rapier has
     * none; Jolt's defaults would cap fast propellers at 450 rpm).
     */
    private static final float RAPIER_MAX_LINEAR_VELOCITY = 2_000.0f;
    private static final float RAPIER_MAX_ANGULAR_VELOCITY = 400.0f;

    // region Shared native resources

    /**
     * One native job system for every dimension: per-scene pools multiplied
     * native threads by the dimension count and were never shut down.
     */
    private static JobSystem sharedJobSystem;
    private static ExecutorService sharedWorkers;
    private static int sharedWorkerCount = 1;

    private static synchronized JobSystem sharedJobSystem(final int threads) {
        if (sharedJobSystem == null) {
            // Extra barriers so dimensions ticked in parallel by other mods never starve.
            sharedJobSystem = new JobSystemThreadPool(Jolt.cMaxPhysicsJobs, Jolt.cMaxPhysicsBarriers * 2, threads);
        }
        return sharedJobSystem;
    }

    private static synchronized ExecutorService sharedWorkers() {
        if (sharedWorkers == null) {
            sharedWorkerCount = SableJoltConfig.get().workerThreads();
            final AtomicInteger counter = new AtomicInteger();
            sharedWorkers = Executors.newFixedThreadPool(sharedWorkerCount, r -> {
                final Thread t = new Thread(r, "sable-jolt-worker-" + counter.incrementAndGet());
                t.setDaemon(true);
                return t;
            });
        }
        return sharedWorkers;
    }

    // endregion

    private final SableJoltConfig config;

    /**
     * Rapier-compatible behavior (see {@link SableJoltConfig#rapierCompatiblePhysics}).
     */
    private final boolean rapierMode;

    private final PhysicsSystem system;
    private final BodyInterface bi;

    /**
     * Shared BoxShapes keyed by half extents. Terrain slabs repeat the same few
     * sizes across thousands of sections and rope points repeat one size per
     * rope; Jolt shapes are immutable and reference-counted, so one native
     * instance per size replaces one instance per slab (native memory saver).
     */
    private final Object2ObjectOpenHashMap<BoxKey, BoxShape> sharedBoxShapes = new Object2ObjectOpenHashMap<>();

    private static final class BoxKey {
        private float hx;
        private float hy;
        private float hz;

        private BoxKey() {
            this(0, 0, 0);
        }

        private BoxKey(final float hx, final float hy, final float hz) {
            this.hx = hx;
            this.hy = hy;
            this.hz = hz;
        }

        private BoxKey set(final float hx, final float hy, final float hz) {
            this.hx = hx;
            this.hy = hy;
            this.hz = hz;
            return this;
        }

        @Override
        public boolean equals(final Object obj) {
            if (obj == this) return true;
            if (!(obj instanceof final BoxKey that)) return false;
            return this.hx == that.hx &&
                    this.hy == that.hy &&
                    this.hz == that.hz;
        }

        @Override
        public int hashCode() {
            int result = Float.floatToRawIntBits(this.hx);
            result = 31 * result + Float.floatToRawIntBits(this.hy);
            result = 31 * result + Float.floatToRawIntBits(this.hz);
            return result;
        }

        @Override
        public String toString() {
            return "BoxKey[hx=" + this.hx + ", hy=" + this.hy + ", hz=" + this.hz + ']';
        }
    }

    private static final ThreadLocal<BoxKey> BOX_KEY = ThreadLocal.withInitial(BoxKey::new);

    private BoxShape sharedBoxShape(final float hx, final float hy, final float hz) {
        // Probe with a reusable ThreadLocal key (no allocation on hit; merged
        // boxes repeat a handful of sizes). The probe is never stored: on miss a
        // dedicated immutable copy becomes the map key, so later set() calls
        // cannot corrupt the stored hash. Sections are built on worker threads.
        final BoxKey probe = BOX_KEY.get().set(hx, hy, hz);
        synchronized (this.sharedBoxShapes) {
            final BoxShape existing = this.sharedBoxShapes.get(probe);
            if (existing != null) {
                return existing;
            }
            // Same rounding as the per-block boxes (JoltVoxelColliderData#shape).
            final BoxShape created = newBoxShape(hx, hy, hz, 0.025f);
            this.sharedBoxShapes.put(new BoxKey(hx, hy, hz), created);
            return created;
        }
    }

    /**
     * Creates a box shape whose convex radius never exceeds its smallest half
     * extent (Jolt only asserts this in debug builds; a larger radius turns
     * the support function inside out and produces garbage contacts).
     */
    public static BoxShape newBoxShape(final float hx, final float hy, final float hz, final float convexRadius) {
        final float ex = Math.max(hx, 1.0e-4f);
        final float ey = Math.max(hy, 1.0e-4f);
        final float ez = Math.max(hz, 1.0e-4f);
        final float radius = Math.min(convexRadius, 0.5f * Math.min(ex, Math.min(ey, ez)));
        return new BoxShape(ex, ey, ez, radius);
    }

    /**
     * Reused {@code MassProperties}/{@code Mat44} for {@link #setMassProperties}
     * (server thread only): avoids a native allocation pair per body per tick.
     */
    private final MassProperties massPropsScratch = new MassProperties();
    private final Mat44 inertiaScratch = new Mat44(
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 0f,
            0f, 0f, 0f, 1f);

    /**
     * Lock-free batch reads for the snapshot (nothing else touches bodies
     * between steps).
     */
    private final BatchBodyInterface batchNoLock;
    private BodyIdArray snapshotIds;
    private long snapshotGen = 1;

    /**
     * Bodies currently dragged by the creative grip (physics staff): gravity is
     * switched off so weight can neither pull nor tip the body over, while the
     * body stays dynamic — contacts still resolve, so it cannot be pushed
     * through the terrain.
     */
    private final IntOpenHashSet creativeDragBodies = new IntOpenHashSet();
    /**
     * Dragged bodies whose handle was removed; gravity is restored at the next
     * step start. Deferred because the staff recreates its constraint every
     * tick — an immediate restore would flap every tick.
     */
    private final IntOpenHashSet creativeRestorePending = new IntOpenHashSet();

    /**
     * Starts (or confirms) the creative drag of a body. Idempotent per grab.
     */
    public void beginCreativeDrag(final int joltId) {
        this.creativeRestorePending.remove(joltId);
        if (this.creativeDragBodies.add(joltId)) {
            final SableBody sb = this.bodiesByJoltId.get(joltId);
            if (sb != null && sb.body != null) {
                final var motion = sb.body.getMotionProperties();
                if (motion != null) {
                    motion.setGravityFactor(0.0f);
                }
                // The push-out needs every contact of the dragged body.
                this.setBodyLayer(sb, LAYER_MOVING_REPORT);
            }
        }
    }

    /**
     * Marks the end of a creative drag; gravity is restored in
     * {@link #flushCreativeRestore} at the next step start.
     */
    public void endCreativeDrag(final int joltId) {
        if (this.creativeDragBodies.contains(joltId)) {
            this.creativeRestorePending.add(joltId);
        }
    }

    public boolean isCreativeDrag(final int joltId) {
        return this.creativeDragBodies.contains(joltId);
    }

    private void flushCreativeRestore() {
        if (this.creativeRestorePending.isEmpty()) {
            return;
        }
        for (final int joltId : this.creativeRestorePending) {
            this.creativeDragBodies.remove(joltId);
            try {
                final SableBody sb = this.bodiesByJoltId.get(joltId);
                if (sb != null && sb.body != null) {
                    final var motion = sb.body.getMotionProperties();
                    if (motion != null) {
                        motion.setGravityFactor(1.0f);
                    }
                }
            } catch (final Throwable ignored) {
                // body may be gone already
            }
        }
        this.creativeRestorePending.clear();
    }

    private DoubleBuffer snapshotPos;
    private FloatBuffer snapshotRot;
    private FloatBuffer snapshotLin;
    private FloatBuffer snapshotAng;
    private ByteBuffer snapshotActive;

    public BodyInterface getBodyInterface() {
        return this.bi;
    }

    private final TempAllocator tempAllocator;
    private final JobSystem jobSystem;
    private final JobSystem singleThreadJobSystem;
    private static final int MULTITHREAD_BODY_THRESHOLD = 24;

    /**
     * A body with at least this many compound leaves makes the native narrow
     * phase dominate the step; such scenes use the multithreaded job system
     * regardless of the body count.
     */
    private static final int HEAVY_BODY_CHILDREN = 2048;

    /**
     * Awake compound leaves in the scene from which the multithreaded job system
     * is used, however few bodies there are: sixteen moving bodies of 500 leaves
     * cost as much narrow phase as one huge body.
     */
    private static final int PARALLEL_AWAKE_LEAVES = 512;

    /**
     * Compound leaves from which a sub-level only sweeps (LinearCast) while it is
     * fast enough to skip a block in one substep. A sweep casts the whole compound
     * against the world, and Jolt starts it at 0.75 x the smallest leaf radius per
     * step, so one carpet in a ship means sweeping from about 1 m/s: 16 bodies of
     * 900 blocks reached 160 ms ticks. Below one block per substep a discrete step
     * cannot miss a full block.
     */
    private static final int SPEED_SWEPT_BODY_CHILDREN = 256;

    /**
     * Jolt buoyancy iterates the body's blocks per substep; above this count the
     * loop is subsampled (with volume compensation) so a 30k-block body costs
     * the same as a 4k-block one.
     */
    private static final int BUOYANCY_MAX_CHILDREN = 4096;

    /**
     * Strong reference: the native physics system only holds a raw pointer.
     */
    private final SceneContactListener contactListener;
    /**
     * Native filter deciding which contact pairs reach {@link #contactListener}.
     */
    private final ObjectLayerPairFilterTable listenerFilter;

    public final JoltVoxelColliderData.Registry colliderRegistry = new JoltVoxelColliderData.Registry();

    /**
     * Set by the pipeline. Kept for API compatibility; box (re)building happens
     * on the server thread in the pipeline, never during rebuilds.
     */
    private volatile JoltVoxelColliderBakery colliderBakery;

    public void attachColliderBakery(final JoltVoxelColliderBakery bakery) {
        this.colliderBakery = bakery;
    }

    private final Int2ObjectOpenHashMap<SableBody> bodies = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<SableBody> bodiesByJoltId = new Int2ObjectOpenHashMap<>();
    /**
     * Bodies with a native body, dense; drives the batch snapshot.
     */
    private final ObjectArrayList<SableBody> bodyList = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> contraptionList = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> pendingCreation = new ObjectArrayList<>();
    private final Long2ObjectOpenHashMap<GlobalChunk> globalChunks = new Long2ObjectOpenHashMap<>();
    /**
     * Terrain regions by region key, and their bodies (one per terrain layer) by body id.
     */
    private final Long2ObjectOpenHashMap<TerrainRegion> terrainRegions = new Long2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<RegionLayer> terrainRegionsByBodyId = new Int2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<TerrainTile> terrainTiles = new Long2ObjectOpenHashMap<>();
    /**
     * Tiles whose compounds must be rebuilt before the next step.
     */
    private final ReferenceOpenHashSet<TerrainTile> dirtyTiles = new ReferenceOpenHashSet<>();
    /**
     * Terrain sections built with a surface mesh, and the flush count until
     * which they keep it (see {@link #MESH_MARGIN}).
     */
    private final Long2LongOpenHashMap meshedSections = new Long2LongOpenHashMap();
    private final LongArrayList meshUpgrades = new LongArrayList();
    private final it.unimi.dsi.fastutil.doubles.DoubleArrayList meshUpgradeDistances = new it.unimi.dsi.fastutil.doubles.DoubleArrayList();
    /**
     * Terrain sections holding blocks with a friction or restitution multiplier:
     * bodies overlapping them route their contacts through the listener, which
     * applies the multipliers per block pair.
     */
    private final LongOpenHashSet specialSections = new LongOpenHashSet();
    /**
     * Terrain sections holding blocks with restitution.
     */
    private final LongOpenHashSet bouncySections = new LongOpenHashSet();
    private final Long2ObjectOpenHashMap<ChunkSectionData> allChunks = new Long2ObjectOpenHashMap<>();
    /**
     * Plot section key → runtime id of the sub-level owning it (-1 when absent;
     * runtime id 0 is valid).
     */
    private final Long2IntOpenHashMap sectionOwners = new Long2IntOpenHashMap();

    {
        this.sectionOwners.defaultReturnValue(-1);
    }
    private final Long2ObjectOpenHashMap<JointRecord> joints = new Long2ObjectOpenHashMap<>();
    private final Long2ObjectOpenHashMap<RopeStrand> ropes = new Long2ObjectOpenHashMap<>();

    private final Int2ObjectOpenHashMap<Quaterniond> angularFollowPrev = new Int2ObjectOpenHashMap<>();
    private final Int2ObjectOpenHashMap<Vector3d> angularCachedOmega = new Int2ObjectOpenHashMap<>();

    public Quaterniond getAngularFollowPrev(final int joltId) {
        return this.angularFollowPrev.get(joltId);
    }

    public void setAngularFollowPrev(final int joltId, final Quaterniond q) {
        this.angularFollowPrev.computeIfAbsent(joltId, k -> new Quaterniond()).set(q);
    }

    public Vector3d getCachedAngularOmega(final int joltId) {
        return this.angularCachedOmega.get(joltId);
    }

    public void setCachedAngularOmega(final int joltId, final Vector3d omega) {
        this.angularCachedOmega.computeIfAbsent(joltId, k -> new Vector3d()).set(omega);
    }

    public void clearAngularServo(final int joltId) {
        this.angularFollowPrev.remove(joltId);
        this.angularCachedOmega.remove(joltId);
    }

    private final AtomicInteger nextRuntimeId = new AtomicInteger();
    private final AtomicLong nextJointId = new AtomicLong(1);
    private final AtomicLong nextRopeId = new AtomicLong(1);

    private final ObjectArrayList<double[]> reportedCollisions = new ObjectArrayList<>();

    /**
     * Guards reportedCollisions: Jolt fires contact callbacks from worker threads
     * when the solver runs multithreaded.
     */
    private final Object reportedLock = new Object();

    /**
     * Bodies and global chunks whose shapes must be rebuilt; block edits mark them
     * dirty and the actual rebuild runs once per simulation step.
     */
    private final LinkedHashSet<SableBody> dirtyBodies = new LinkedHashSet<>();
    private final LinkedHashSet<GlobalChunk> dirtyGlobalChunks = new LinkedHashSet<>();

    private int groundBodyId;
    private boolean disposed;

    final double gravityX;
    final double gravityY;
    final double gravityZ;
    final double universalDrag;
    /**
     * Unit gravity direction (see {@link #updateMotionQuality}).
     */
    private final boolean hasGravity;
    private final double gravityDirX;
    private final double gravityDirY;
    private final double gravityDirZ;

    /**
     * Velocity/position solver steps requested by Sable's config; the heavy-body
     * reduction restores these.
     */
    private int configuredVelocitySteps = 10;
    private int configuredPositionSteps = 4;

    JoltPhysicsScene(final double gravityX, final double gravityY, final double gravityZ, final double universalDrag) {
        JoltNative.ensureInitialized();

        this.config = SableJoltConfig.get();
        this.rapierMode = this.config.rapierCompatiblePhysics;

        this.gravityX = gravityX;
        this.gravityY = gravityY;
        this.gravityZ = gravityZ;
        this.universalDrag = universalDrag;
        final double gravity = Math.sqrt(gravityX * gravityX + gravityY * gravityY + gravityZ * gravityZ);
        this.hasGravity = gravity > 1.0e-6;
        this.gravityDirX = this.hasGravity ? gravityX / gravity : 0.0;
        this.gravityDirY = this.hasGravity ? gravityY / gravity : 0.0;
        this.gravityDirZ = this.hasGravity ? gravityZ / gravity : 0.0;

        final int numObjLayers = NUM_OBJECT_LAYERS;
        final int numBpLayers = 2;

        final ObjectLayerPairFilterTable ovoFilter = new ObjectLayerPairFilterTable(numObjLayers);
        for (final int moving : new int[]{LAYER_MOVING, LAYER_MOVING_REPORT}) {
            ovoFilter.enableCollision(moving, LAYER_MOVING);
            ovoFilter.enableCollision(moving, LAYER_MOVING_REPORT);
            ovoFilter.enableCollision(moving, LAYER_STATIC);
            ovoFilter.enableCollision(moving, LAYER_STATIC_SPECIAL);
            ovoFilter.enableCollision(moving, LAYER_ROPE);
        }
        ovoFilter.enableCollision(LAYER_STATIC, LAYER_ROPE);
        ovoFilter.enableCollision(LAYER_STATIC_SPECIAL, LAYER_ROPE);

        final BroadPhaseLayerInterfaceTable layerMap = new BroadPhaseLayerInterfaceTable(numObjLayers, numBpLayers);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_MOVING, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_MOVING_REPORT, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_ROPE, 0);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_STATIC, 1);
        layerMap.mapObjectToBroadPhaseLayer(LAYER_STATIC_SPECIAL, 1);

        final ObjectVsBroadPhaseLayerFilterTable ovbFilter = createBroadPhaseFilter(layerMap, numBpLayers, ovoFilter, numObjLayers);

        // Contacts reach Java only when one side is a report body (fast bodies,
        // bodies with special blocks or mounted contraptions, see classifyLayer)
        // or the special terrain body. Plain resting and slow contacts are
        // handled entirely in native code.
        this.listenerFilter = new ObjectLayerPairFilterTable(numObjLayers);
        for (final int layer : new int[]{LAYER_MOVING, LAYER_MOVING_REPORT, LAYER_STATIC, LAYER_STATIC_SPECIAL, LAYER_ROPE}) {
            this.listenerFilter.enableCollision(LAYER_MOVING_REPORT, layer);
        }
        this.listenerFilter.enableCollision(LAYER_MOVING, LAYER_STATIC_SPECIAL);
        this.listenerFilter.enableCollision(LAYER_ROPE, LAYER_STATIC_SPECIAL);

        this.system = new PhysicsSystem();
        // The system keeps references to the layer tables; capacities come from
        // the config.
        this.system.init(this.config.maxBodies, 0, this.config.maxBodyPairs, this.config.maxContactConstraints,
                layerMap, ovbFilter, ovoFilter);
        this.system.setGravity((float) gravityX, (float) gravityY, (float) gravityZ);

        final PhysicsSettings physics = this.system.getPhysicsSettings();
        // Contact tolerances of Sable's Rapier world (lib.rs): bodies rest 2.5 mm
        // deep (normalized_allowed_linear_error) instead of Jolt's 2 cm. The deeper
        // a body rests, the more of the next block's side it overlaps at every seam.
        physics.setPenetrationSlop(ALLOWED_PENETRATION);
        physics.setSpeculativeContactDistance(SPECULATIVE_CONTACT_DISTANCE);
        physics.setNumVelocitySteps(this.configuredVelocitySteps);
        physics.setNumPositionSteps(this.configuredPositionSteps);
        physics.setBaumgarte(0.2f);
        // Swept bodies (see updateMotionQuality) cast from 0.2 x their inner radius
        // per substep (4 m/s for a block at 2 substeps) and stop at most 5 % of it
        // deep: an impact is caught before the body sinks into the ground, as in
        // Rapier, instead of up to 12 cm deep with Jolt's defaults.
        physics.setLinearCastThreshold(LINEAR_CAST_THRESHOLD);
        physics.setLinearCastMaxPenetration(LINEAR_CAST_MAX_PENETRATION);
        // Internal edge removal compares the vertices of touching faces; block
        // corners of neighbouring sections match exactly, the tolerance only
        // absorbs float rounding of the section offsets.
        physics.setInternalEdgeRemovalVertexToleranceSq(1.0e-4f);
        // Big resting structures form huge contact islands; the splitter parallelizes
        // and bounds their solve instead of one monolithic island iteration.
        physics.setUseLargeIslandSplitter(true);
        physics.setUseManifoldReduction(true);
        if (this.rapierMode) {
            // Rapier sub-levels sleep below 0.15 m/s and 0.15 rad/s after 2 s.
            physics.setPointVelocitySleepThreshold(0.15f);
            physics.setTimeBeforeSleep(2.0f);
        }
        this.system.setPhysicsSettings(physics);
        if (this.rapierMode) {
            // Rapier's default rule: the average of both frictions (Jolt's default is
            // the geometric mean). Block multipliers are applied by the listener.
            this.system.setCombineFriction(CombineFunction.average(true));
        }

        this.bi = this.system.getBodyInterface();
        this.batchNoLock = this.system.getBodyInterfaceNoLock();
        // Fixed block for the common case; large islands fall back to malloc
        // instead of aborting the process when the block runs out.
        this.tempAllocator = new TempAllocatorImplWithMallocFallback(tempAllocatorBytes(this.config.maxContactConstraints));
        this.jobSystem = sharedJobSystem(this.config.workerThreads());
        this.singleThreadJobSystem = new JobSystemSingleThreaded(Jolt.cMaxPhysicsJobs);

        this.contactListener = new SceneContactListener();
        this.contactListener.setObjectLayerPairFilter(this.listenerFilter);
        // No body / broad-phase filters (a Skip mode would drop every callback).
        this.contactListener.setBodyFilterMode(com.github.stephengold.joltjni.enumerate.EFilterMode.Either);
        this.contactListener.setBroadPhaseLayerFilterMode(com.github.stephengold.joltjni.enumerate.EFilterMode.Either);
        this.contactListener.setEnableAdded(true);
        this.contactListener.setEnablePersisted(true);
        this.contactListener.setEnableRemoved(false);
        // Per-contact validation would be the only way to filter seam contacts
        // in Java, but jolt-jni attaches and detaches the JVM thread around every
        // callback from a native worker (~23 us each). Seams are handled natively
        // instead (terrain regions + enhanced internal edge removal).
        this.contactListener.setEnableValidate(false);
        this.system.setContactListener(this.contactListener);

        this.createGround();
    }

    public boolean isRapierCompatible() {
        return this.rapierMode;
    }

    /**
     * Builds the object-vs-broad-phase filter and verifies it against the pair
     * filter. jolt-jni 6.1.1 reads the two layer counts of this table in the
     * opposite order of its Java parameters, which leaves every object layer
     * above 1 colliding with nothing (rope points fell through everything).
     * Whichever order produces the correct table is used.
     */
    private static ObjectVsBroadPhaseLayerFilterTable createBroadPhaseFilter(final BroadPhaseLayerInterfaceTable layerMap, final int numBpLayers,
                                                                             final ObjectLayerPairFilterTable pairFilter, final int numObjLayers) {
        final ObjectVsBroadPhaseLayerFilterTable swapped = new ObjectVsBroadPhaseLayerFilterTable(layerMap, numObjLayers, pairFilter, numBpLayers);
        if (matches(swapped, layerMap, numBpLayers, pairFilter, numObjLayers)) {
            return swapped;
        }
        final ObjectVsBroadPhaseLayerFilterTable documented = new ObjectVsBroadPhaseLayerFilterTable(layerMap, numBpLayers, pairFilter, numObjLayers);
        if (matches(documented, layerMap, numBpLayers, pairFilter, numObjLayers)) {
            return documented;
        }
        Sable.LOGGER.error("[SableJolt] Could not build a consistent broad-phase layer filter; some layers may not collide");
        return swapped;
    }

    private static boolean matches(final ObjectVsBroadPhaseLayerFilterTable table, final BroadPhaseLayerInterfaceTable layerMap, final int numBpLayers,
                                   final ObjectLayerPairFilterTable pairFilter, final int numObjLayers) {
        for (int layer = 0; layer < numObjLayers; layer++) {
            for (int bp = 0; bp < numBpLayers; bp++) {
                boolean expected = false;
                for (int other = 0; other < numObjLayers && !expected; other++) {
                    expected = layerMap.getBroadPhaseLayer(other) == bp && pairFilter.shouldCollide(layer, other);
                }
                if (table.shouldCollide(layer, bp) != expected) {
                    return false;
                }
            }
        }
        return true;
    }

    private Body groundBody;

    private void createGround() {
        // The ground body MUST sit at the world origin: constraint anchors on the
        // world are measured from its center of mass, and a shifted origin would
        // add a phantom offset to every motor target.
        final BodyCreationSettings bcs = new BodyCreationSettings(
                newBoxShape(0.01f, 0.01f, 0.01f, 0.001f),
                0f, 0f, 0f, 0f, 0f, 0f, 1f, EMotionType.Static, LAYER_STATIC);
        final Body body = this.bi.createBody(bcs);
        body.setUserData(-1L);
        this.bi.addBody(body, EActivation.DontActivate);
        this.groundBodyId = body.getId();
        this.groundBody = body;
    }

    public int groundBodyId() {
        return this.groundBodyId;
    }

    //region Body records

    /**
     * One collision leaf: a box of the block at (bx, by, bz), or a merged box of
     * full cubes from (bx, by, bz) to (ex, ey, ez) inclusive. Coordinates are in
     * the owner's block space (plot or contraption-local), world space for terrain.
     */
    public static final class Child {
        public final int bx;
        public final int by;
        public final int bz;
        public final int ex;
        public final int ey;
        public final int ez;
        /**
         * Collider of the block; for a merged box, of one of its blocks (all of
         * them share the same contact behavior).
         */
        public final int colliderId;
        /**
         * The body the block belongs to; null for terrain.
         */
        @Nullable
        public final SableBody owner;

        public Child(final int bx, final int by, final int bz, final int colliderId, @Nullable final SableBody owner) {
            this(bx, by, bz, bx, by, bz, colliderId, owner);
        }

        public Child(final int bx, final int by, final int bz, final int ex, final int ey, final int ez,
                     final int colliderId, @Nullable final SableBody owner) {
            this.bx = bx;
            this.by = by;
            this.bz = bz;
            this.ex = ex;
            this.ey = ey;
            this.ez = ez;
            this.colliderId = colliderId;
            this.owner = owner;
        }

        public boolean merged() {
            return this.ex != this.bx || this.ey != this.by || this.ez != this.bz;
        }

        /**
         * The block of this leaf closest to a point in the owner's block space.
         */
        public int nearestX(final double x) {
            return Math.max(this.bx, Math.min(this.ex, (int) Math.floor(x)));
        }

        public int nearestY(final double y) {
            return Math.max(this.by, Math.min(this.ey, (int) Math.floor(y)));
        }

        public int nearestZ(final double z) {
            return Math.max(this.bz, Math.min(this.ez, (int) Math.floor(z)));
        }
    }

    private static final BodySection[] NO_SECTIONS = new BodySection[0];
    private static final SableBody[] NO_BODIES = new SableBody[0];
    private static final Child[] NO_CHILDREN = new Child[0];
    private static final int[] NO_INTS = new int[0];
    private static final float[] NO_FLOATS = new float[0];

    /**
     * The built shape of one 16³ section of a body. The shape's local origin is
     * placed at (ox, oy, oz) in the body's block space, so the section never has
     * to be rebuilt when the body's center of mass moves.
     */
    static final class BodySection {
        final long key;
        final ConstShape shape;
        /**
         * Keeps a created static compound alive while cached.
         */
        @Nullable
        final ShapeRefC ref;
        final double ox;
        final double oy;
        final double oz;
        final Child[] children;
        /**
         * Sub-shape ID bits of the section compound; 0 for a single leaf.
         */
        final int leafBits;
        /**
         * Unique solid blocks (x, y, z triples) and their buoyancy volume factor.
         */
        final int[] blocks;
        final float[] volumes;
        /**
         * Holds blocks with special contact behavior (friction, restitution, callbacks).
         */
        final boolean special;
        /**
         * Holds blocks with restitution (see {@link JoltPhysicsScene#updateMotionQuality}).
         */
        final boolean bouncy;

        BodySection(final long key, final ConstShape shape, @Nullable final ShapeRefC ref,
                    final double ox, final double oy, final double oz,
                    final Child[] children, final int leafBits, final int[] blocks, final float[] volumes, final boolean special,
                    final boolean bouncy) {
            this.key = key;
            this.shape = shape;
            this.ref = ref;
            this.ox = ox;
            this.oy = oy;
            this.oz = oz;
            this.children = children;
            this.leafBits = leafBits;
            this.blocks = blocks;
            this.volumes = volumes;
            this.special = special;
            this.bouncy = bouncy;
        }

        @Nullable
        Child leaf(final int relativeId) {
            if (this.leafBits == 0) {
                return this.children.length == 1 ? this.children[0] : null;
            }
            final int j = relativeId & ((1 << this.leafBits) - 1);
            return j < this.children.length ? this.children[j] : null;
        }
    }

    /**
     * Number of sub-shape ID bits a compound with {@code n} children uses
     * (Jolt's {@code CompoundShape::GetSubShapeIDBits}).
     */
    static int subShapeBits(final int n) {
        return n <= 1 ? 0 : 32 - Integer.numberOfLeadingZeros(n - 1);
    }

    /**
     * Bookkeeping for a single rigid body managed by this scene.
     */
    public static final class SableBody {
        public enum Kind {SUB_LEVEL, BOX, CONTRAPTION}

        public final Kind kind;
        public final int runtimeId;
        /**
         * Native body id; 0 while the body has no native body (body limit
         * reached, or a contraption embedded in its mount's compound).
         */
        public int joltId;

        /**
         * The Java wrapper of the native body, kept alive for the body's lifetime.
         * {@code new Body(long)} takes a virtual address, not a body ID, so the
         * wrapper from {@code createBody} must be retained instead of reconstructed.
         */
        public Body body;
        int listIndex = -1;
        /**
         * Current object layer of the native body.
         */
        int layer = LAYER_MOVING;

        /**
         * The assembled parent compound (sections, then embedded contraptions).
         */
        public MutableCompoundShape shape;
        /**
         * Every collision leaf, flattened in section order.
         */
        public final ObjectArrayList<Child> children = new ObjectArrayList<>();

        /**
         * Own chunk store of kinematic contraptions (they have no plot java-side).
         */
        public final Long2ObjectOpenHashMap<ChunkSectionData> chunks = new Long2ObjectOpenHashMap<>();

        public int minX;
        public int minY;
        public int minZ;
        public int maxX;
        public int maxY;
        public int maxZ;
        public boolean hasBounds;

        public final Vector3d centerOfMass = new Vector3d();

        // Section cache (see rebuildShape)
        final Long2ObjectOpenHashMap<BodySection> sections = new Long2ObjectOpenHashMap<>();
        /**
         * Plot sections this sub-level owns in the shared chunk store.
         */
        final LongOpenHashSet ownedSections = new LongOpenHashSet();
        final LongOpenHashSet dirtySections = new LongOpenHashSet();
        boolean boundsDirty = true;
        boolean parentDirty = true;
        BodySection[] assembled = NO_SECTIONS;
        SableBody[] embedded = NO_BODIES;
        int parentBits;
        int leafCount;
        /**
         * Motion quality is LinearCast ({@link JoltPhysicsScene#updateMotionQuality}).
         */
        boolean sweeping;
        /**
         * Contacts must always reach the listener: special blocks or mounted contraptions.
         */
        boolean layerSpecial;
        /**
         * Overlaps terrain with friction or restitution multipliers (see
         * {@link JoltPhysicsScene#classifyLayer}): its terrain contacts sample
         * the blocks they touch.
         */
        volatile boolean nearSpecial;
        /**
         * Holds blocks with restitution, or overlaps terrain holding some: moves
         * discretely (see {@link JoltPhysicsScene#updateMotionQuality}).
         */
        boolean bouncy;
        volatile boolean nearBouncy;
        /**
         * Where and when this body last gave the terrain around it surface
         * meshes (see {@link JoltPhysicsScene#updateMeshedSections}).
         */
        double meshedAtX = Double.NaN;
        double meshedAtY = Double.NaN;
        double meshedAtZ = Double.NaN;
        long meshedAtFlush = Long.MIN_VALUE / 2;
        /**
         * Some of its sections are still waiting for their mesh (rate limit).
         */
        boolean meshPending;
        /**
         * Upper bound of the distance from the center of mass to any block.
         */
        float boundRadius = 1.0f;
        double shapeComX;
        double shapeComY;
        double shapeComZ;

        // Unique solid blocks for buoyancy (x, y, z triples).
        int[] buoyancyBlocks = NO_INTS;
        float[] buoyancyVolumes = NO_FLOATS;
        int buoyancyCount;

        // Per-tick buoyancy of the Rapier model, applied every substep.
        boolean hasTickBuoyancy;
        double tickFx;
        double tickFy;
        double tickFz;
        double tickTx;
        double tickTy;
        double tickTz;

        // Kinematic contraptions
        public int mountId = -1;
        /**
         * Contraption pose: block-space position of its center of mass in the
         * mount (world space when unmounted) and orientation relative to it.
         */
        public final Vector3d relPos = new Vector3d();
        public final Quaterniond relRot = new Quaterniond(1.0, 0.0, 0.0, 0.0);
        /**
         * Contraption velocities in its own frame (Rapier "fake velocities").
         */
        public final Vector3d linVel = new Vector3d();
        public final Vector3d angVel = new Vector3d();
        /**
         * Contraptions embedded in this sub-level's compound.
         */
        final ObjectArrayList<SableBody> mountedContraptions = new ObjectArrayList<>(0);
        /**
         * Whether this contraption is currently part of its mount's compound.
         */
        boolean embeddedInMount;
        /**
         * Activate the body at the next compound assembly (mounted contraptions changed).
         */
        boolean wakeOnAssemble;

        /**
         * Joints and rope attachments using this body (skips the scan on COM shifts).
         */
        int attachmentCount;

        // Last applied mass properties (re-applied if the native body is created late).
        double appliedMass = Double.NaN;
        final Matrix3d appliedInertia = new Matrix3d();

        // Batch-read body state (see refreshBodySnapshot). Served while
        // snapValid (sleeping bodies keep theirs); otherwise direct JNI reads
        // are used instead. Also holds the last known good pose.
        boolean snapValid;
        boolean snapQueued;
        long snapGen;
        double snapX;
        double snapY;
        double snapZ;
        double snapComX;
        double snapComY;
        double snapComZ;
        float snapQx;
        float snapQy;
        float snapQz;
        float snapQw = 1.0f;
        float snapVx;
        float snapVy;
        float snapVz;
        float snapAx;
        float snapAy;
        float snapAz;
        /**
         * Awake as of the last snapshot; set eagerly when this scene activates
         * the body between steps.
         */
        boolean snapActive = true;

        // Deepest contact penetration this step (plus its outward normal) for
        // creatively dragged bodies — used by the end-of-step push-out.
        // Reset in flushDirty, accumulated in the contact listener.
        double dragPenDepth;
        double dragPenNx;
        double dragPenNy;
        double dragPenNz;

        // Kept for the benchmarks; the section cache replaced the full-rebuild
        // dedup and throttle.
        long rebuiltDataVersion = -1;
        long lastShapeRebuildNanos;

        public SableBody(final Kind kind, final int runtimeId) {
            this.kind = kind;
            this.runtimeId = runtimeId;
        }

        public boolean contains(final int x, final int y, final int z) {
            return this.hasBounds
                    && x >= this.minX && x <= this.maxX
                    && y >= this.minY && y <= this.maxY
                    && z >= this.minZ && z <= this.maxZ;
        }

        /**
         * Resolves a contact sub-shape ID to its leaf. The ID is decoded
         * arithmetically: the parent compound contributes the low
         * {@code parentBits} (child index), the section compound the next
         * {@code leafBits} (leaf index); embedded contraptions recurse.
         */
        @Nullable
        public Child childAt(final int subShapeId) {
            final BodySection[] secs = this.assembled;
            final SableBody[] emb = this.embedded;
            final int n = secs.length + emb.length;
            if (n == 0) {
                return null;
            }
            final int pb = this.parentBits;
            final int index = pb == 0 ? 0 : subShapeId & ((1 << pb) - 1);
            final int rest = subShapeId >>> pb;
            if (index < secs.length) {
                return secs[index].leaf(rest);
            }
            final int k = index - secs.length;
            return k < emb.length ? emb[k].childAt(rest) : null;
        }
    }

    /**
     * The geometry of one terrain layer of a section, local to the section's
     * minimum corner (its tile places that corner; world-absolute float offsets
     * lose precision far from spawn). Either a box part alone (a compound of
     * boxes, or one wrapped box), a single surface mesh, or a compound of the
     * surface meshes followed by the box part.
     */
    static final class TerrainPart {
        @Nullable
        ConstShape shape;
        /**
         * Keeps a created shape alive.
         */
        @Nullable
        ShapeRefC ref;
        /**
         * Box leaves, in sub-shape order of the box part.
         */
        Child[] children = NO_CHILDREN;
        /**
         * Sub-shape ID bits of the box part; 0 for a single leaf.
         */
        int leafBits;
        /**
         * One entry per surface mesh (the block range of the whole section).
         */
        Child[] meshChildren = NO_CHILDREN;
        /**
         * Sub-shape ID bits of the compound of meshes and box part; 0 when the
         * shape is the box part or the only mesh itself.
         */
        int partBits;

        void clear() {
            this.shape = null;
            this.ref = null;
            this.children = NO_CHILDREN;
            this.leafBits = 0;
            this.meshChildren = NO_CHILDREN;
            this.partBits = 0;
        }

        @Nullable
        Child childAt(final int subShapeId) {
            final Child[] meshes = this.meshChildren;
            if (meshes.length == 0) {
                return this.leafAt(subShapeId);
            }
            if (this.partBits == 0) {
                return this.children.length == 0 ? meshes[0] : null;
            }
            final int part = subShapeId & ((1 << this.partBits) - 1);
            return part < meshes.length ? meshes[part] : this.leafAt(subShapeId >>> this.partBits);
        }

        @Nullable
        private Child leafAt(final int subShapeId) {
            final Child[] c = this.children;
            if (c.length == 0) {
                return null;
            }
            if (this.leafBits == 0) {
                return c[0];
            }
            final int j = subShapeId & ((1 << this.leafBits) - 1);
            return j < c.length ? c[j] : null;
        }
    }

    public static final class GlobalChunk {
        final int cx;
        final int cy;
        final int cz;
        final long key;
        ChunkSectionData data;

        /**
         * Geometry per terrain layer ({@link #TERRAIN_PLAIN}, {@link #TERRAIN_SPECIAL}).
         */
        final TerrainPart[] parts = {new TerrainPart(), new TerrainPart()};
        /**
         * The tile this section belongs to.
         */
        @Nullable
        TerrainTile tile;

        GlobalChunk(final int cx, final int cy, final int cz, final ChunkSectionData data) {
            this.cx = cx;
            this.cy = cy;
            this.cz = cz;
            this.key = ChunkSectionData.packSectionPos(cx, cy, cz);
            this.data = data;
        }
    }

    /**
     * One terrain layer of a tile: the compound of its sections' parts.
     */
    static final class TileLayer {
        /**
         * Sections in the compound, in sub-shape order.
         */
        GlobalChunk[] assembled = new GlobalChunk[0];
        @Nullable
        ConstShape shape;
        /**
         * Keeps a created static compound alive.
         */
        @Nullable
        ShapeRefC ref;
    }

    /**
     * 4x4x4 terrain sections, assembled into one compound per terrain layer of
     * the region bodies.
     */
    static final class TerrainTile {
        final long key;
        final TerrainRegion region;
        final int tx;
        final int ty;
        final int tz;
        /**
         * Every section of the tile, with or without geometry.
         */
        final ReferenceOpenHashSet<GlobalChunk> members = new ReferenceOpenHashSet<>();
        final TileLayer[] layers = {new TileLayer(), new TileLayer()};

        TerrainTile(final long key, final TerrainRegion region, final int tx, final int ty, final int tz) {
            this.key = key;
            this.region = region;
            this.tx = tx;
            this.ty = ty;
            this.tz = tz;
        }

        @Nullable
        Child childAt(final int layer, final int subShapeId) {
            final GlobalChunk[] secs = this.layers[layer].assembled;
            final int bits = subShapeBits(secs.length);
            final int index = bits == 0 ? 0 : subShapeId & ((1 << bits) - 1);
            return index < secs.length ? secs[index].parts[layer].childAt(subShapeId >>> bits) : null;
        }
    }

    /**
     * One static body of a terrain region: a mutable compound of the tile
     * compounds of one terrain layer, rebuilt when tiles change.
     */
    static final class RegionLayer {
        final TerrainRegion region;
        final int layer;
        /**
         * Tiles in the body's compound, in sub-shape order.
         */
        TerrainTile[] assembled = new TerrainTile[0];
        @Nullable
        MutableCompoundShape shape;
        @Nullable
        Body body;
        int joltId;

        RegionLayer(final TerrainRegion region, final int layer) {
            this.region = region;
            this.layer = layer;
        }

        @Nullable
        Child childAt(final int subShapeId) {
            final TerrainTile[] tiles = this.assembled;
            final int bits = subShapeBits(tiles.length);
            final int index = bits == 0 ? 0 : subShapeId & ((1 << bits) - 1);
            return index < tiles.length ? tiles[index].childAt(this.layer, subShapeId >>> bits) : null;
        }
    }

    /**
     * The terrain of 64x64 sections (all heights): one static body per terrain layer.
     */
    static final class TerrainRegion {
        final long key;
        final int rx;
        final int rz;
        /**
         * Block coordinates of the body origins.
         */
        final int originX;
        final int originZ;
        final ObjectArrayList<TerrainTile> tiles = new ObjectArrayList<>();
        final RegionLayer[] layers = {new RegionLayer(this, TERRAIN_PLAIN), new RegionLayer(this, TERRAIN_SPECIAL)};

        TerrainRegion(final long key, final int rx, final int rz) {
            this.key = key;
            this.rx = rx;
            this.rz = rz;
            this.originX = rx << (REGION_SECTION_SHIFT + 4);
            this.originZ = rz << (REGION_SECTION_SHIFT + 4);
        }
    }

    public static final class JointRecord {
        public long id;
        public Constraint constraint;
        public int joltIdA = -1;
        public int joltIdB = -1;
        public boolean contactsEnabled = true;
        public boolean fixedContacts;
        public TwoBodyConstraintSettings genericSettings;
        public final double[][] limits = new double[6][2];
        public final boolean[] locked = new boolean[6];
        /** Viscous drag between the anchors (per second), see {@link #applyJointDampers}. */
        public boolean damper;
        public double linearDamping;
        public double angularDamping;
    }

    public static final class RopeAttachment {
        int mountJoltId;
        double x;
        double y;
        double z;
        public Constraint constraint;
    }

    public static final class RopeStrand {
        final IntArrayList points = new IntArrayList();
        final ObjectArrayList<DistanceConstraint> joints = new ObjectArrayList<>();
        double pointRadius;
        double firstJointLength;
        @Nullable RopeAttachment start;
        @Nullable RopeAttachment end;
    }

    //endregion

    //region Body management

    private SableBody createBody(final SableBody.Kind kind, final int runtimeId, final Vector3dc pos, final Quaterniondc rot, final int layer) {
        final SableBody previous = this.bodies.get(runtimeId);
        if (previous != null) {
            // Defensive: a second add for the same id would orphan the first body.
            this.destroyBody(previous);
        }

        final SableBody sb = new SableBody(kind, runtimeId);
        sb.layer = layer;
        sb.snapX = sb.snapComX = pos.x();
        sb.snapY = sb.snapComY = pos.y();
        sb.snapZ = sb.snapComZ = pos.z();
        sb.snapQx = (float) rot.x();
        sb.snapQy = (float) rot.y();
        sb.snapQz = (float) rot.z();
        sb.snapQw = (float) rot.w();
        this.bodies.put(runtimeId, sb);

        if (!this.createNativeBody(sb)) {
            this.pendingCreation.add(sb);
        }
        return sb;
    }

    /**
     * Creates the native body of a record at its last known pose. Returns false
     * (and keeps the record frozen at that pose) when the body limit is reached.
     */
    private boolean createNativeBody(final SableBody sb) {
        // Report layer until the first snapshot classifies the body: a body
        // loaded with velocity can hit something in its very first step.
        if (sb.layer == LAYER_MOVING) {
            sb.layer = LAYER_MOVING_REPORT;
        }
        final MutableCompoundShape shape = new MutableCompoundShape();
        final BodyCreationSettings bcs = new BodyCreationSettings(
                shape,
                sb.snapX, sb.snapY, sb.snapZ,
                sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw,
                sb.kind == SableBody.Kind.CONTRAPTION ? EMotionType.Kinematic : EMotionType.Dynamic,
                sb.layer);
        bcs.setLinearDamping((float) this.universalDrag);
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(sb.kind == SableBody.Kind.CONTRAPTION ? FRICTION_CONTRAPTION : FRICTION_SUB_LEVEL);
        bcs.setRestitution(0.0f);
        bcs.setApplyGyroscopicForce(true);
        // Drops the ghost contacts with the internal edges of whatever this body
        // touches: the seams between the blocks of another sub-level, as Rapier's
        // interior collision test does (terrain regions set it themselves).
        bcs.setEnhancedInternalEdgeRemoval(true);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateMassAndInertia);
        if (this.rapierMode && sb.kind == SableBody.Kind.SUB_LEVEL) {
            bcs.setMaxLinearVelocity(RAPIER_MAX_LINEAR_VELOCITY);
            bcs.setMaxAngularVelocity(RAPIER_MAX_ANGULAR_VELOCITY);
        }

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("sub-level " + sb.runtimeId + " is frozen until bodies free up");
            return false;
        }
        body.setUserData(sb.runtimeId);
        // Rapier creates sub-levels awake; a sleeping new body would hang in the
        // air until something woke it.
        this.bi.addBody(body, sb.kind == SableBody.Kind.CONTRAPTION ? EActivation.DontActivate : EActivation.Activate);

        sb.joltId = body.getId();
        sb.body = body;
        sb.shape = shape;
        sb.listIndex = this.bodyList.size();
        sb.snapValid = true;
        sb.snapActive = sb.kind != SableBody.Kind.CONTRAPTION;
        sb.sweeping = true; // created LinearCast, see updateMotionQuality
        this.bodyList.add(sb);
        this.bodiesByJoltId.put(sb.joltId, sb);
        sb.boundsDirty = true;
        sb.parentDirty = true;
        this.markDirty(sb);
        if (!Double.isNaN(sb.appliedMass)) {
            this.applyMassProperties(sb, sb.appliedMass, sb.appliedInertia);
        }
        return true;
    }

    private void retryPendingCreation() {
        if (this.pendingCreation.isEmpty()) {
            return;
        }
        for (int i = this.pendingCreation.size() - 1; i >= 0; i--) {
            final SableBody sb = this.pendingCreation.get(i);
            if (this.bodies.get(sb.runtimeId) != sb) {
                this.pendingCreation.remove(i);
                continue;
            }
            if (this.system.getNumBodies() >= this.config.maxBodies - 1) {
                return;
            }
            if (this.createNativeBody(sb)) {
                this.pendingCreation.remove(i);
                if (sb.kind == SableBody.Kind.SUB_LEVEL) {
                    this.attachWaitingContraptions(sb);
                }
            }
        }
    }

    private void removeFromBodyList(final SableBody sb) {
        final int index = sb.listIndex;
        if (index < 0) {
            return;
        }
        final int last = this.bodyList.size() - 1;
        final SableBody moved = this.bodyList.get(last);
        this.bodyList.set(index, moved);
        moved.listIndex = index;
        this.bodyList.remove(last);
        sb.listIndex = -1;
    }

    private void destroyBody(final SableBody sb) {
        if (this.bodies.get(sb.runtimeId) == sb) {
            this.bodies.remove(sb.runtimeId);
        }
        this.pendingCreation.remove(sb);
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.remove(sb);
        }

        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            this.contraptionList.remove(sb);
            this.detachFromMount(sb);
        } else if (sb.kind == SableBody.Kind.SUB_LEVEL) {
            // Mounted contraptions stay registered and re-attach if the
            // sub-level comes back (recovery re-adds it with the same id).
            for (final SableBody contraption : sb.mountedContraptions) {
                contraption.embeddedInMount = false;
            }
            sb.mountedContraptions.clear();
            this.dropOwnedSections(sb);
        }

        if (sb.body != null) {
            this.creativeDragBodies.remove(sb.joltId);
            this.creativeRestorePending.remove(sb.joltId);
            this.bodiesByJoltId.remove(sb.joltId);
            this.removeFromBodyList(sb);
            // Native constraints and rope attachments hold raw references to their
            // bodies: destroying a body with live attachments leaves dangling
            // native references behind, and the next system.update() dies with an
            // access violation inside joltjni.dll. Detach everything first (Sable
            // re-creates the attachments it still needs, e.g. on the remove and
            // re-add recovery path).
            this.detachBodyAttachments(sb.joltId);
            this.bi.removeBody(sb.joltId);
            this.bi.destroyBody(sb.joltId);
            sb.body = null;
            sb.joltId = 0;
        }
    }

    /**
     * Drops the plot sections of a removed sub-level. Sable reuses plot slots, so
     * stale sections would otherwise turn into ghost blocks inside the next
     * sub-level of that slot (and leak 16 KB each). Block edits re-create a
     * missing plot section from scratch, see {@link #changeBlocks}.
     */
    private void dropOwnedSections(final SableBody sb) {
        for (final long key : sb.ownedSections) {
            if (this.sectionOwners.get(key) == sb.runtimeId) {
                this.sectionOwners.remove(key);
                this.allChunks.remove(key);
            }
        }
        sb.ownedSections.clear();
    }

    /**
     * Records the sub-level owning a plot section (kept per body too, so
     * removing a sub-level never scans every section).
     */
    private void setSectionOwner(final long key, final int owner) {
        final int previous = this.sectionOwners.put(key, owner);
        if (previous == owner) {
            return;
        }
        if (previous != -1) {
            final SableBody old = this.bodies.get(previous);
            if (old != null) {
                old.ownedSections.remove(key);
            }
        }
        final SableBody sb = this.bodies.get(owner);
        if (sb != null) {
            sb.ownedSections.add(key);
        }
    }

    private int clearSectionOwner(final long key) {
        final int previous = this.sectionOwners.remove(key);
        if (previous != -1) {
            final SableBody old = this.bodies.get(previous);
            if (old != null) {
                old.ownedSections.remove(key);
            }
        }
        return previous;
    }

    /**
     * Removes every joint and rope attachment mounted on the given body, and
     * drops its servo caches. Called before the native body is destroyed so no
     * native object keeps referencing the freed body.
     */
    private void detachBodyAttachments(final int joltId) {
        if (!this.joints.isEmpty()) {
            // Collect first: removeJoint mutates the map under iteration.
            LongArrayList deadJoints = null;
            for (final var entry : this.joints.long2ObjectEntrySet()) {
                final JointRecord rec = entry.getValue();
                if (rec.joltIdA != joltId && rec.joltIdB != joltId) {
                    continue;
                }
                if (deadJoints == null) {
                    deadJoints = new LongArrayList();
                }
                deadJoints.add(entry.getLongKey());
            }
            if (deadJoints != null) {
                for (int i = 0; i < deadJoints.size(); i++) {
                    this.removeJoint(deadJoints.getLong(i));
                }
            }
        }
        if (!this.ropes.isEmpty()) {
            for (final RopeStrand strand : this.ropes.values()) {
                if (strand.start != null && strand.start.mountJoltId == joltId) {
                    this.removeRopeAttachment(strand.start);
                    strand.start = null;
                }
                if (strand.end != null && strand.end.mountJoltId == joltId) {
                    this.removeRopeAttachment(strand.end);
                    strand.end = null;
                }
            }
        }
        this.angularFollowPrev.remove(joltId);
        this.angularCachedOmega.remove(joltId);
    }

    public void createSubLevel(final int id, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody sb = this.createBody(SableBody.Kind.SUB_LEVEL, id, pos, rot, LAYER_MOVING);
        if (sb.body != null) {
            this.attachWaitingContraptions(sb);
        }
    }

    public void removeSubLevel(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null) {
            this.destroyBody(sb);
        }
    }

    public void createBox(final int id, final double mass, final double hx, final double hy, final double hz, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody previous = this.bodies.get(id);
        if (previous != null) {
            this.destroyBody(previous);
        }
        final BodyCreationSettings bcs = new BodyCreationSettings(
                newBoxShape((float) hx, (float) hy, (float) hz, 0.025f),
                pos.x(), pos.y(), pos.z(),
                (float) rot.x(), (float) rot.y(), (float) rot.z(), (float) rot.w(),
                EMotionType.Dynamic, LAYER_MOVING_REPORT);
        bcs.setMotionQuality(EMotionQuality.LinearCast);
        bcs.setFriction(FRICTION_BOX);
        bcs.setApplyGyroscopicForce(true);
        bcs.setEnhancedInternalEdgeRemoval(true);
        final MassProperties mp = new MassProperties();
        mp.setMass((float) Math.max(mass, 1.0e-3));
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("physics box " + id + " was not created");
            return;
        }
        body.setUserData(id);
        this.bi.addBody(body, EActivation.Activate);

        final SableBody sb = new SableBody(SableBody.Kind.BOX, id);
        sb.joltId = body.getId();
        sb.sweeping = true;
        sb.layer = LAYER_MOVING_REPORT;
        sb.body = body;
        sb.boundRadius = (float) Math.sqrt(hx * hx + hy * hy + hz * hz);
        sb.snapX = sb.snapComX = pos.x();
        sb.snapY = sb.snapComY = pos.y();
        sb.snapZ = sb.snapComZ = pos.z();
        sb.snapQx = (float) rot.x();
        sb.snapQy = (float) rot.y();
        sb.snapQz = (float) rot.z();
        sb.snapQw = (float) rot.w();
        sb.boundsDirty = false;
        sb.parentDirty = false;
        sb.listIndex = this.bodyList.size();
        sb.snapValid = true;
        sb.snapActive = true;
        this.bodyList.add(sb);
        this.bodies.put(id, sb);
        this.bodiesByJoltId.put(sb.joltId, sb);
    }

    public void removeBox(final int id) {
        this.removeSubLevel(id);
    }

    public SableBody body(final int runtimeId) {
        return this.bodies.get(runtimeId);
    }

    @Nullable
    public SableBody bodyByJoltId(final int joltId) {
        return this.bodiesByJoltId.get(joltId);
    }

    //endregion

    //region Pose / mass / teleport

    /**
     * Reusable destination for {@link #getPose} writes; avoids per-read allocations.
     */
    public static class PoseCache {
        public double p1;
        public double p2;
        public double p3;
        public double p4;
        public double p5;
        public double p6;
        public double p7;
    }

    public void getPose(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null || (sb.snapValid)) {
            // Fresh snapshot, or the last known pose of a frozen record.
            store.p1 = sb.snapX;
            store.p2 = sb.snapY;
            store.p3 = sb.snapZ;
            store.p4 = sb.snapQx;
            store.p5 = sb.snapQy;
            store.p6 = sb.snapQz;
            store.p7 = sb.snapQw;
            return;
        }
        final Body body = sb.body;
        final RVec3 pos = this.tmpCom.get();
        final Quat rot = this.tmpQuat.get();
        body.getPositionAndRotation(pos, rot);
        store.p1 = pos.xx();
        store.p2 = pos.yy();
        store.p3 = pos.zz();
        store.p4 = rot.getX();
        store.p5 = rot.getY();
        store.p6 = rot.getZ();
        store.p7 = rot.getW();
    }

    /**
     * Updates the center of mass of a body. The body itself was already moved
     * by Sable (teleport by the rotated shift); the compound children follow at
     * the next step without rebuilding any section, and joints anchored on the
     * body keep their block-space anchor (Rapier re-anchors every tick).
     */
    public void setCenterOfMass(final int id, final double x, final double y, final double z) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        final double dx = x - sb.centerOfMass.x;
        final double dy = y - sb.centerOfMass.y;
        final double dz = z - sb.centerOfMass.z;
        if (dx == 0.0 && dy == 0.0 && dz == 0.0) {
            return;
        }
        sb.centerOfMass.set(x, y, z);
        sb.parentDirty = true;
        this.markDirty(sb);
        if (sb.attachmentCount > 0 && sb.body != null) {
            this.notifyComShift(sb.joltId, (float) dx, (float) dy, (float) dz);
        }
        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            this.markMountDirty(sb);
        }
    }

    public void setLocalBounds(final int id, final int minX, final int minY, final int minZ, final int maxX, final int maxY, final int maxZ) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.hasBounds && sb.minX == minX && sb.minY == minY && sb.minZ == minZ
                && sb.maxX == maxX && sb.maxY == maxY && sb.maxZ == maxZ) {
            return;
        }
        sb.minX = minX;
        sb.minY = minY;
        sb.minZ = minZ;
        sb.maxX = maxX;
        sb.maxY = maxY;
        sb.maxZ = maxZ;
        sb.hasBounds = true;
        sb.boundsDirty = true;
        this.markDirty(sb);
    }

    /**
     * Overrides the mass and inertia tensor of a body. The center of mass is
     * set separately via {@link #setCenterOfMass}.
     */
    public void setMassProperties(final int id, final double mass, final Matrix3dc inertiaTensor) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        // Zero or non-finite input would make Jolt compute an infinite inverse
        // mass and poison the body with NaN on the next step.
        if (!(mass > 0.0) || !Double.isFinite(mass) || !isFinite(inertiaTensor)) {
            return;
        }
        sb.appliedMass = mass;
        sb.appliedInertia.set(inertiaTensor);
        if (sb.body != null) {
            this.applyMassProperties(sb, mass, inertiaTensor);
        }
    }

    private static boolean isFinite(final Matrix3dc m) {
        return Double.isFinite(m.m00()) && Double.isFinite(m.m01()) && Double.isFinite(m.m02())
                && Double.isFinite(m.m10()) && Double.isFinite(m.m11()) && Double.isFinite(m.m12())
                && Double.isFinite(m.m20()) && Double.isFinite(m.m21()) && Double.isFinite(m.m22());
    }

    private void applyMassProperties(final SableBody sb, final double mass, final Matrix3dc inertiaTensor) {
        final Body body = sb.body;
        if (body == null || !body.isDynamic()) {
            return;
        }
        final MotionProperties motion = body.getMotionProperties();
        if (motion == null) {
            return;
        }

        // Reused across calls: mass stats arrive every tick per body, and the old
        // code allocated a MassProperties + Mat44 + varargs float[] per call.
        // Memory layout preserved from the old varargs sequence.
        final MassProperties mp = this.massPropsScratch;
        mp.setMass((float) mass);
        final Mat44 inertia = this.inertiaScratch;
        inertia.setElement(0, 0, (float) inertiaTensor.m00());
        inertia.setElement(0, 1, (float) inertiaTensor.m10());
        inertia.setElement(0, 2, (float) inertiaTensor.m20());
        inertia.setElement(0, 3, 0.0f);
        inertia.setElement(1, 0, (float) inertiaTensor.m01());
        inertia.setElement(1, 1, (float) inertiaTensor.m11());
        inertia.setElement(1, 2, (float) inertiaTensor.m21());
        inertia.setElement(1, 3, 0.0f);
        inertia.setElement(2, 0, (float) inertiaTensor.m02());
        inertia.setElement(2, 1, (float) inertiaTensor.m12());
        inertia.setElement(2, 2, (float) inertiaTensor.m22());
        inertia.setElement(2, 3, 0.0f);
        inertia.setElement(3, 0, 0.0f);
        inertia.setElement(3, 1, 0.0f);
        inertia.setElement(3, 2, 0.0f);
        inertia.setElement(3, 3, 1.0f);
        mp.setInertia(inertia);
        motion.setMassProperties(ALLOWED_DOFS_ALL, mp);
    }

    public void teleportObject(final int id, final double x, final double y, final double z, final double i, final double j, final double k, final double r) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (!(Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(z)
                && Double.isFinite(i) && Double.isFinite(j) && Double.isFinite(k) && Double.isFinite(r))) {
            return;
        }
        // The snapshot keeps serving the new pose (velocities are untouched by
        // a teleport); it is also the restore point of the poison watchdog.
        sb.snapX = sb.snapComX = x;
        sb.snapY = sb.snapComY = y;
        sb.snapZ = sb.snapComZ = z;
        sb.snapQx = (float) i;
        sb.snapQy = (float) j;
        sb.snapQz = (float) k;
        sb.snapQw = (float) r;
        if (sb.body != null) {
            this.bi.setPositionAndRotation(sb.joltId, x, y, z, (float) i, (float) j, (float) k, (float) r, EActivation.Activate);
            sb.snapActive = true;
        }
    }

    public void wakeUpObject(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null && sb.body != null) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void addLinearAngularVelocities(final int id, final double lx, final double ly, final double lz,
                                           final double ax, final double ay, final double az, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(lx) && Double.isFinite(ly) && Double.isFinite(lz)
                && Double.isFinite(ax) && Double.isFinite(ay) && Double.isFinite(az))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }
        final boolean fresh = sb.snapValid;
        float bvx, bvy, bvz, bax, bay, baz;
        if (fresh) {
            bvx = sb.snapVx;
            bvy = sb.snapVy;
            bvz = sb.snapVz;
            bax = sb.snapAx;
            bay = sb.snapAy;
            baz = sb.snapAz;
        } else {
            final Vec3 v = this.tmpVec.get();
            body.getLinearVelocity(v);
            bvx = v.getX();
            bvy = v.getY();
            bvz = v.getZ();
            body.getAngularVelocity(v);
            bax = v.getX();
            bay = v.getY();
            baz = v.getZ();
        }
        bvx += (float) lx;
        bvy += (float) ly;
        bvz += (float) lz;
        bax += (float) ax;
        bay += (float) ay;
        baz += (float) az;
        this.bi.setLinearAndAngularVelocity(sb.joltId, bvx, bvy, bvz, bax, bay, baz);
        if (fresh) {
            sb.snapVx = bvx;
            sb.snapVy = bvy;
            sb.snapVz = bvz;
            sb.snapAx = bax;
            sb.snapAy = bay;
            sb.snapAz = baz;
        }
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void getLinearVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null) {
            store.p1 = 0.0;
            store.p2 = 0.0;
            store.p3 = 0.0;
            return;
        }
        if (sb.snapValid) {
            store.p1 = sb.snapVx;
            store.p2 = sb.snapVy;
            store.p3 = sb.snapVz;
            return;
        }
        final Vec3 v = this.tmpVec.get();
        sb.body.getLinearVelocity(v);
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    public void getAngularVelocity(final int id, final PoseCache store) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null) {
            return;
        }
        if (sb.body == null) {
            store.p1 = 0.0;
            store.p2 = 0.0;
            store.p3 = 0.0;
            return;
        }
        if (sb.snapValid) {
            store.p1 = sb.snapAx;
            store.p2 = sb.snapAy;
            store.p3 = sb.snapAz;
            return;
        }
        final Vec3 v = this.tmpVec.get();
        sb.body.getAngularVelocity(v);
        store.p1 = v.getX();
        store.p2 = v.getY();
        store.p3 = v.getZ();
    }

    /**
     * Current rotation of a body: the snapshot when fresh (impulses arrive
     * between steps, so it is exact), a native read otherwise.
     */
    private Quat currentRotation(final SableBody sb) {
        final Quat rot = this.tmpQuat.get();
        if (sb.snapValid) {
            rot.set(sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw);
        } else {
            sb.body.getRotation(rot);
        }
        return rot;
    }

    /**
     * Applies an impulse at a local offset (relative to the center of mass), replicating the rapier semantics.
     */
    public void applyForce(final int id, final double x, final double y, final double z,
                           final double fx, final double fy, final double fz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(x) && Double.isFinite(y) && Double.isFinite(z)
                && Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = this.currentRotation(sb);
        final float qx = rot.getX();
        final float qy = rot.getY();
        final float qz = rot.getZ();
        final float qw = rot.getW();

        Vector3f r = this.rotateInto((float) fx, (float) fy, (float) fz, qx, qy, qz, qw);
        final float impX = r.x, impY = r.y, impZ = r.z;
        r = this.rotateInto((float) x, (float) y, (float) z, qx, qy, qz, qw);
        final float offX = r.x, offY = r.y, offZ = r.z;
        final RVec3 com = this.tmpCom.get();
        body.getCenterOfMassPosition(com);

        // Jolt does not wake bodies on AddImpulse; activation is required separately
        final Vec3 imp = this.tmpVec.get();
        imp.set(impX, impY, impZ);
        final RVec3 point = this.tmpPoint.get();
        point.set(com.xx() + offX, com.yy() + offY, com.zz() + offZ);
        body.addImpulse(imp, point);
        this.invalidateSnapshot(sb); // impulse changed the velocity
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    public void applyForceAndTorque(final int id, final double fx, final double fy, final double fz,
                                    final double tx, final double ty, final double tz, final boolean wakeUp) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.body == null) {
            return;
        }
        if (!(Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz)
                && Double.isFinite(tx) && Double.isFinite(ty) && Double.isFinite(tz))) {
            return;
        }
        final Body body = sb.body;
        if (!wakeUp && !body.isActive()) {
            return;
        }

        final Quat rot = this.currentRotation(sb);
        final float rotX = rot.getX();
        final float rotY = rot.getY();
        final float rotZ = rot.getZ();
        final float rotW = rot.getW();

        Vector3f r = this.rotateInto((float) fx, (float) fy, (float) fz, rotX, rotY, rotZ, rotW);
        body.addImpulse(r.x, r.y, r.z);
        r = this.rotateInto((float) tx, (float) ty, (float) tz, rotX, rotY, rotZ, rotW);
        body.addAngularImpulse(r.x, r.y, r.z);
        this.invalidateSnapshot(sb); // impulse changed the velocity
        if (wakeUp) {
            this.bi.activateBody(sb.joltId);
            sb.snapActive = true;
        }
    }

    /**
     * Scratch buffer for {@link #rotateInto}; ThreadLocal because callers run
     * both on the server thread and on buoyancy/Jolt job threads.
     */
    private final ThreadLocal<Vector3f> rotateOut = ThreadLocal.withInitial(Vector3f::new);
    private final ThreadLocal<Quat> tmpQuat = ThreadLocal.withInitial(Quat::new);
    private final ThreadLocal<RVec3> tmpCom = ThreadLocal.withInitial(RVec3::new);
    private final ThreadLocal<RVec3> tmpPoint = ThreadLocal.withInitial(RVec3::new);
    private final ThreadLocal<Vec3> tmpVec = ThreadLocal.withInitial(Vec3::new);

    /**
     * Rotates a vector by a quaternion, writing the result into a per-thread
     * scratch buffer (allocation-free). The result is valid only until the next
     * call on the same thread — copy the components out before rotating again.
     */
    private Vector3f rotateInto(final float vx, final float vy, final float vz,
                                final float qx, final float qy, final float qz, final float qw) {
        // t = 2 * cross(q.xyz, v)
        final float tx = 2.0f * (qy * vz - qz * vy);
        final float ty = 2.0f * (qz * vx - qx * vz);
        final float tz = 2.0f * (qx * vy - qy * vx);
        // v + w*t + cross(q.xyz, t)
        final Vector3f out = this.rotateOut.get();
        out.x = vx + qw * tx + (qy * tz - qz * ty);
        out.y = vy + qw * ty + (qz * tx - qx * tz);
        out.z = vz + qw * tz + (qx * ty - qy * tx);
        return out;
    }

    /**
     * Converts a constraint anchor into the body-local frame required by Jolt's
     * {@code EConstraintSpace.LocalToBodyCom}. Sable passes anchors in the
     * body's block space (the plot of a sub-level), which is the body's local
     * frame, so only the center of mass is subtracted — exactly Rapier's
     * {@code local_anchor = pos - center_of_mass}. For the static ground the
     * world point is used as-is.
     */
    public RVec3 toLocalAnchor(@Nullable final SableBody sb, final double x, final double y, final double z) {
        if (sb == null) {
            return new RVec3(x, y, z);
        }
        return new RVec3(x - sb.centerOfMass.x, y - sb.centerOfMass.y, z - sb.centerOfMass.z);
    }

    //endregion

    //region Voxel geometry

    private static boolean isSolidBlock(final int packed, @Nullable final JoltVoxelColliderData entry) {
        final int colliderId = ChunkSectionData.colliderIdOf(packed);
        final int voxelState = ChunkSectionData.voxelStateOf(packed);
        // voxel states: 0=empty, 1=face, 2=edge, 3=corner, 4=interior
        if (voxelState < 1 || voxelState > 3 || colliderId <= 0 || entry == null) {
            return false;
        }
        // fluids never produce collision; they only contribute buoyancy and drag
        if (entry.isFluid) {
            return false;
        }
        return entry.hasBoxes();
    }

    /**
     * Brings a body's compound up to date: rebuilds dirty sections, then
     * reassembles the small parent compound when sections, embedded
     * contraptions or the center of mass changed. A center of mass shift
     * therefore costs one cheap reassembly instead of a full rebuild, and block
     * edits only rebuild the touched sections — no throttling needed.
     */
    private void rebuildShape(final SableBody sb) {
        this.rebuildShape(sb, true);
    }

    /**
     * @param parallelSections build many dirty sections on the worker pool; must
     *                         be false when already running on that pool (a
     *                         worker waiting for queued sub-tasks deadlocks it)
     */
    private void rebuildShape(final SableBody sb, final boolean parallelSections) {
        if (sb.kind == SableBody.Kind.BOX) {
            return;
        }
        // Benchmarks reset this to force a full rebuild.
        if (sb.rebuiltDataVersion == -1L) {
            sb.boundsDirty = true;
            sb.rebuiltDataVersion = 0L;
        }

        boolean sectionsChanged = false;
        if (sb.boundsDirty) {
            sb.boundsDirty = false;
            this.collectSectionWindow(sb);
            sectionsChanged = true;
        }

        if (!sb.dirtySections.isEmpty()) {
            final long[] keys = sb.dirtySections.toLongArray();
            sb.dirtySections.clear();
            final BodySection[] built = new BodySection[keys.length];
            final IntConsumer build = i -> built[i] = this.buildBodySection(sb, keys[i]);
            if (parallelSections && keys.length >= 16) {
                runParallelIndices(keys.length, build);
            } else {
                for (int i = 0; i < keys.length; i++) {
                    build.accept(i);
                }
            }
            for (int i = 0; i < keys.length; i++) {
                if (built[i] == null) {
                    sb.sections.remove(keys[i]);
                } else {
                    sb.sections.put(keys[i], built[i]);
                }
            }
            sectionsChanged = true;
        }

        if (sectionsChanged) {
            this.indexSections(sb);
        }

        if (sectionsChanged || sb.parentDirty
                || sb.shapeComX != sb.centerOfMass.x || sb.shapeComY != sb.centerOfMass.y || sb.shapeComZ != sb.centerOfMass.z) {
            sb.parentDirty = false;
            this.assembleParent(sb);
            sb.lastShapeRebuildNanos = System.nanoTime();
        }
    }

    /**
     * Marks every section the body covers for rebuild and drops sections outside.
     * Sub-levels take their sections from the shared chunk store, filtered by
     * their bounds (like the rapier octree rebuild from main_level_chunks);
     * contraptions own a dedicated store.
     */
    private void collectSectionWindow(final SableBody sb) {
        if (sb.kind == SableBody.Kind.CONTRAPTION) {
            sb.sections.keySet().removeIf((long key) -> !sb.chunks.containsKey(key));
            sb.dirtySections.addAll(sb.chunks.keySet());
            return;
        }
        if (!sb.hasBounds) {
            sb.sections.clear();
            sb.dirtySections.clear();
            return;
        }
        final int cMinX = sb.minX >> 4, cMinY = sb.minY >> 4, cMinZ = sb.minZ >> 4;
        final int cMaxX = sb.maxX >> 4, cMaxY = sb.maxY >> 4, cMaxZ = sb.maxZ >> 4;
        sb.sections.keySet().removeIf((long key) -> {
            final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
            return cx < cMinX || cx > cMaxX || cy < cMinY || cy > cMaxY || cz < cMinZ || cz > cMaxZ;
        });
        sb.dirtySections.removeIf((long key) -> {
            final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
            return cx < cMinX || cx > cMaxX || cy < cMinY || cy > cMaxY || cz < cMinZ || cz > cMaxZ;
        });
        final long window = (long) (cMaxX - cMinX + 1) * (cMaxY - cMinY + 1) * (cMaxZ - cMinZ + 1);
        if (window <= this.allChunks.size()) {
            for (int cx = cMinX; cx <= cMaxX; cx++) {
                for (int cy = cMinY; cy <= cMaxY; cy++) {
                    for (int cz = cMinZ; cz <= cMaxZ; cz++) {
                        final long key = ChunkSectionData.packSectionPos(cx, cy, cz);
                        if (this.allChunks.containsKey(key)) {
                            sb.dirtySections.add(key);
                        }
                    }
                }
            }
        } else {
            for (final long key : this.allChunks.keySet()) {
                final int cx = unpackChunkX(key), cy = unpackChunkY(key), cz = unpackChunkZ(key);
                if (cx >= cMinX && cx <= cMaxX && cy >= cMinY && cy <= cMaxY && cz >= cMinZ && cz <= cMaxZ) {
                    sb.dirtySections.add(key);
                }
            }
        }
    }

    /**
     * Refreshes the derived arrays after the section set changed: parent order
     * (sorted by key for determinism), flattened leaves and buoyancy blocks.
     */
    private void indexSections(final SableBody sb) {
        final BodySection[] ordered = sb.sections.values().toArray(NO_SECTIONS);
        Arrays.sort(ordered, (a, b) -> Long.compare(a.key, b.key));
        sb.assembled = ordered;

        int leaves = 0;
        int blocks = 0;
        for (final BodySection s : ordered) {
            leaves += s.children.length;
            blocks += s.volumes.length;
        }
        sb.children.clear();
        sb.children.ensureCapacity(leaves);
        final int[] buoyancyBlocks = blocks == 0 ? NO_INTS : new int[blocks * 3];
        final float[] buoyancyVolumes = blocks == 0 ? NO_FLOATS : new float[blocks];
        int b = 0;
        for (final BodySection s : ordered) {
            sb.children.addElements(sb.children.size(), s.children);
            System.arraycopy(s.blocks, 0, buoyancyBlocks, b * 3, s.blocks.length);
            System.arraycopy(s.volumes, 0, buoyancyVolumes, b, s.volumes.length);
            b += s.volumes.length;
        }
        sb.leafCount = leaves;
        if (sb.hasBounds) {
            final double dx = sb.maxX - sb.minX + 1.0, dy = sb.maxY - sb.minY + 1.0, dz = sb.maxZ - sb.minZ + 1.0;
            sb.boundRadius = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        sb.buoyancyBlocks = buoyancyBlocks;
        sb.buoyancyVolumes = buoyancyVolumes;
        sb.buoyancyCount = blocks;

        // CCD (LinearCast) sweeps the ENTIRE compound against the world every step;
        // for huge bodies that costs seconds per tick while they fall. Large bodies
        // start discrete; updateMotionQuality sweeps them again while they are fast.
        if (sb.body != null && sb.kind == SableBody.Kind.SUB_LEVEL && leaves >= SPEED_SWEPT_BODY_CHILDREN && sb.sweeping) {
            this.bi.setMotionQuality(sb.joltId, EMotionQuality.Discrete);
            sb.sweeping = false;
        }
    }

    /**
     * Shared identity rotation for compound {@code addShape} calls. The Jolt
     * wrappers read the components immediately and never retain the reference,
     * so one instance serves all threads; {@code Quat.sIdentity()} would
     * allocate a new object per leaf (thousands per rebuild).
     */
    private static final Quat IDENTITY_QUAT = new Quat(0.0f, 0.0f, 0.0f, 1.0f);

    /** Scratch offset for parent inserts (one use at a time per thread). */
    private final ThreadLocal<Vec3> sectionVecScratch = ThreadLocal.withInitial(Vec3::new);
    private final ThreadLocal<Quat> sectionQuatScratch = ThreadLocal.withInitial(Quat::new);

    /**
     * Builds the parent compound from the cached section shapes (offset by the
     * current center of mass) plus the shapes of embedded contraptions, and
     * installs it on the native body.
     */
    private void assembleParent(final SableBody sb) {
        final double comX = sb.centerOfMass.x;
        final double comY = sb.centerOfMass.y;
        final double comZ = sb.centerOfMass.z;
        final MutableCompoundShape parent = new MutableCompoundShape();
        final Vec3 offset = this.sectionVecScratch.get();
        final BodySection[] secs = sb.assembled;
        for (final BodySection s : secs) {
            offset.set((float) (s.ox - comX), (float) (s.oy - comY), (float) (s.oz - comZ));
            parent.addShape(offset, IDENTITY_QUAT, s.shape);
        }

        SableBody[] embedded = NO_BODIES;
        if (sb.kind == SableBody.Kind.SUB_LEVEL && !sb.mountedContraptions.isEmpty()) {
            final ObjectArrayList<SableBody> list = new ObjectArrayList<>(sb.mountedContraptions.size());
            final Quat rotation = this.sectionQuatScratch.get();
            for (final SableBody c : sb.mountedContraptions) {
                if (c.shape == null || c.assembled.length == 0) {
                    continue;
                }
                offset.set((float) (c.relPos.x - comX), (float) (c.relPos.y - comY), (float) (c.relPos.z - comZ));
                rotation.set((float) c.relRot.x, (float) c.relRot.y, (float) c.relRot.z, (float) c.relRot.w);
                parent.addShape(offset, rotation, c.shape);
                list.add(c);
            }
            embedded = list.toArray(NO_BODIES);
        }

        sb.shape = parent;
        sb.embedded = embedded;
        boolean special = embedded.length > 0;
        boolean bouncy = false;
        for (final BodySection s : secs) {
            special |= s.special;
            bouncy |= s.bouncy;
        }
        sb.layerSpecial = special;
        sb.bouncy = bouncy;
        if (special && sb.body != null && sb.layer == LAYER_MOVING) {
            this.setBodyLayer(sb, LAYER_MOVING_REPORT);
        }
        sb.parentBits = subShapeBits(secs.length + embedded.length);
        sb.shapeComX = comX;
        sb.shapeComY = comY;
        sb.shapeComZ = comZ;
        if (sb.body != null) {
            // Mounted contraptions changing wake the body, as Rapier does when a
            // body's colliders change; block edits are woken by Sable itself.
            this.bi.setShape(sb.joltId, parent, false, sb.wakeOnAssemble ? EActivation.Activate : EActivation.DontActivate);
            if (sb.wakeOnAssemble) {
                sb.snapActive = true;
            }
            sb.wakeOnAssemble = false;
        }

        if (JoltDebugLogging.STAFF) {
            Sable.LOGGER.info("[SableJolt] rebuild body {}: kind={} sections={} leaves={} embedded={} bounds=({},{},{})..({},{},{}) com=({},{},{})",
                    sb.runtimeId, sb.kind, secs.length, sb.leafCount, embedded.length,
                    sb.minX, sb.minY, sb.minZ, sb.maxX, sb.maxY, sb.maxZ, comX, comY, comZ);
        }
    }

    /**
     * Reusable leaf accumulator of one compound: flat primitive arrays instead
     * of one {@code Vec3} + list node per leaf. A 16³ section holds at most a
     * few thousand leaves; the arrays grow once per thread and are reused by
     * every later section.
     */
    private static final class LeafList {
        float[] ox = new float[256];
        float[] oy = new float[256];
        float[] oz = new float[256];
        ConstShape[] shapes = new ConstShape[256];
        Child[] children = new Child[256];
        int count;

        void add(final float x, final float y, final float z, final ConstShape shape, final Child child) {
            final int i = this.count;
            if (i == this.ox.length) {
                final int cap = i << 1;
                this.ox = Arrays.copyOf(this.ox, cap);
                this.oy = Arrays.copyOf(this.oy, cap);
                this.oz = Arrays.copyOf(this.oz, cap);
                this.shapes = Arrays.copyOf(this.shapes, cap);
                this.children = Arrays.copyOf(this.children, cap);
            }
            this.ox[i] = x;
            this.oy[i] = y;
            this.oz[i] = z;
            this.shapes[i] = shape;
            this.children[i] = child;
            this.count = i + 1;
        }

        void release() {
            // Do not pin native shapes between rebuilds.
            Arrays.fill(this.shapes, 0, this.count, null);
            Arrays.fill(this.children, 0, this.count, null);
            this.count = 0;
        }
    }

    /**
     * Reusable per-thread scratch of the section builders.
     */
    private static final class SectionScratch {
        /**
         * Leaves of plain blocks (all leaves of a sub-level section) and of
         * blocks with special contact behavior (terrain only).
         */
        final LeafList plain = new LeafList();
        final LeafList special = new LeafList();
        int[] blocks = new int[768];
        float[] volumes = new float[256];
        int blockCount;
        /**
         * Per cell of the section: merge class (0 = not mergeable), collider and
         * whether the block is enclosed (interior), see {@link #emitMergedBoxes}.
         */
        final int[] mergeClass = new int[ChunkSectionData.BLOCKS];
        final int[] mergeCollider = new int[ChunkSectionData.BLOCKS];
        final boolean[] mergeInterior = new boolean[ChunkSectionData.BLOCKS];
        final boolean[] mergeUsed = new boolean[ChunkSectionData.BLOCKS];
        /**
         * Surface mesh scratch (terrain): solid full cubes of the section, the
         * exposed faces of one plane, and the triangles (9 floats each).
         */
        final boolean[] fullCube = new boolean[ChunkSectionData.BLOCKS];
        final boolean[] faceMask = new boolean[256];
        float[] triangles = new float[9 * 1024];
        int triangleFloats;

        void ensureBlocks(final int needed) {
            if (needed <= this.volumes.length) {
                return;
            }
            int cap = this.volumes.length;
            while (cap < needed) {
                cap <<= 1;
            }
            this.blocks = Arrays.copyOf(this.blocks, cap * 3);
            this.volumes = Arrays.copyOf(this.volumes, cap);
        }

        void reset() {
            this.plain.release();
            this.special.release();
            this.blockCount = 0;
            this.triangleFloats = 0;
            Arrays.fill(this.mergeClass, 0);
        }

        void releaseRefs() {
            this.plain.release();
            this.special.release();
            this.blockCount = 0;
        }

        void addTriangle(final float ax, final float ay, final float az, final float bx, final float by, final float bz,
                         final float cx, final float cy, final float cz) {
            int i = this.triangleFloats;
            if (i + 9 > this.triangles.length) {
                this.triangles = Arrays.copyOf(this.triangles, this.triangles.length << 1);
            }
            final float[] t = this.triangles;
            t[i++] = ax;
            t[i++] = ay;
            t[i++] = az;
            t[i++] = bx;
            t[i++] = by;
            t[i++] = bz;
            t[i++] = cx;
            t[i++] = cy;
            t[i++] = cz;
            this.triangleFloats = i;
        }
    }

    private static final ThreadLocal<SectionScratch> SECTION_SCRATCH =
            ThreadLocal.withInitial(SectionScratch::new);

    /**
     * Merge class of blocks without special contact behavior: they merge with
     * each other whatever block they are.
     */
    private static final int MERGE_DEFAULT = Integer.MAX_VALUE;


    /**
     * Whether a collider is one solid full cube (its neighbours' faces against it are hidden).
     */
    private static boolean isFullCube(@Nullable final JoltVoxelColliderData entry) {
        return entry != null && !entry.isFluid && entry.boxes.size() == 1 && entry.boxes.get(0).fullCube();
    }

    /**
     * Classifies one cell for {@link #emitMergedBoxes}: a solid full cube with a
     * single box and no contact callback merges with neighbours of the same
     * contact behavior (interior blocks included: they fill the merged boxes).
     *
     * @return whether the cell was classified mergeable
     */
    private static boolean classifyMergeable(final SectionScratch s, final int idx, final int packed, final int colliderId,
                                             @Nullable final JoltVoxelColliderData entry) {
        final int state = ChunkSectionData.voxelStateOf(packed);
        if (colliderId <= 0 || entry == null || entry.isFluid || entry.contactEvents != null || state < 1 || state > 4
                || entry.boxes.size() != 1 || !entry.boxes.get(0).fullCube()) {
            s.mergeClass[idx] = 0;
            return false;
        }
        s.mergeClass[idx] = entry.needsSpecialContacts() ? colliderId : MERGE_DEFAULT;
        s.mergeCollider[idx] = colliderId;
        s.mergeInterior[idx] = state == 4;
        return true;
    }

    /**
     * Greedily merges the mergeable cells of a section into boxes (along x, then
     * z, then y) and appends them as leaves. Boxes made only of interior blocks
     * are enclosed and skipped, as Rapier skips interior blocks. Merging is what
     * keeps big flat contacts cheap: a 30 x 30 plate on flat ground touches a
     * handful of boxes instead of 3600 block pairs (Rapier gets there by only
     * colliding the corner blocks of flat surfaces, dispatcher.rs).
     *
     * @param anyClass  merge every mergeable block regardless of its contact
     *                  behavior (volume boxes under a surface mesh)
     * @param exactRuns stack rows only while the next row's run spans exactly
     *                  the same x range (boxes that are contact surfaces)
     * @param recess    inset of every box face (boxes under a surface mesh)
     */
    private void emitMergedBoxes(final SectionScratch s, final LeafList out, final int baseX, final int baseY, final int baseZ,
                                 @Nullable final SableBody owner, final boolean anyClass, final boolean exactRuns, final float recess) {
        final int[] cls = s.mergeClass;
        if (anyClass) {
            for (int i = 0; i < cls.length; i++) {
                if (cls[i] != 0) {
                    cls[i] = MERGE_DEFAULT;
                }
            }
        }
        final boolean[] used = s.mergeUsed;
        Arrays.fill(used, false);
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                for (int x = 0; x < 16; x++) {
                    final int idx = ChunkSectionData.index(x, y, z);
                    final int c = cls[idx];
                    if (c == 0 || used[idx]) {
                        continue;
                    }
                    int x1 = x;
                    while (x1 + 1 < 16 && !used[idx + (x1 + 1 - x)] && cls[idx + (x1 + 1 - x)] == c) {
                        x1++;
                    }
                    // With exact runs, rows are stacked only while the next row's
                    // run spans exactly the same x range: a box must not take part
                    // of a longer run, which would leave the rest of that run as a
                    // box whose corner sits in the middle of this box's edge. Jolt
                    // only drops seam contacts between faces with matching corners.
                    int z1 = z;
                    zLoop:
                    while (z1 + 1 < 16) {
                        for (int xx = x; xx <= x1; xx++) {
                            final int n = ChunkSectionData.index(xx, y, z1 + 1);
                            if (used[n] || cls[n] != c) {
                                break zLoop;
                            }
                        }
                        if (exactRuns && x > 0) {
                            final int before = ChunkSectionData.index(x - 1, y, z1 + 1);
                            if (!used[before] && cls[before] == c) {
                                break;
                            }
                        }
                        if (exactRuns && x1 < 15) {
                            final int after = ChunkSectionData.index(x1 + 1, y, z1 + 1);
                            if (!used[after] && cls[after] == c) {
                                break;
                            }
                        }
                        z1++;
                    }
                    int y1 = y;
                    yLoop:
                    while (y1 + 1 < 16) {
                        for (int zz = z; zz <= z1; zz++) {
                            for (int xx = x; xx <= x1; xx++) {
                                final int n = ChunkSectionData.index(xx, y1 + 1, zz);
                                if (used[n] || cls[n] != c) {
                                    break yLoop;
                                }
                            }
                        }
                        y1++;
                    }
                    boolean exposed = false;
                    for (int yy = y; yy <= y1; yy++) {
                        for (int zz = z; zz <= z1; zz++) {
                            for (int xx = x; xx <= x1; xx++) {
                                final int n = ChunkSectionData.index(xx, yy, zz);
                                used[n] = true;
                                exposed |= !s.mergeInterior[n];
                            }
                        }
                    }
                    if (!exposed) {
                        continue;
                    }
                    final float hx = (x1 - x + 1) * 0.5f;
                    final float hy = (y1 - y + 1) * 0.5f;
                    final float hz = (z1 - z + 1) * 0.5f;
                    out.add(x + hx, y + hy, z + hz, this.sharedBoxShape(hx - recess, hy - recess, hz - recess),
                            new Child(baseX + x, baseY + y, baseZ + z, baseX + x1, baseY + y1, baseZ + z1, s.mergeCollider[idx], owner));
                }
            }
        }
    }

    // region Surface meshes

    /**
     * Section offsets of the face neighbour in each direction (-x, +x, -y, +y, -z, +z).
     */
    private static final int[] FACE_DX = {-1, 1, 0, 0, 0, 0};
    private static final int[] FACE_DY = {0, 0, -1, 1, 0, 0};
    private static final int[] FACE_DZ = {0, 0, 0, 0, -1, 1};

    /**
     * Builds the surface mesh of the mergeable full cubes of a terrain section
     * into {@code s.triangles}: their faces not covered by a solid full cube
     * (the neighbouring sections decide the faces on the section border). Each
     * face plane is cut into 4x4 patches aligned with the section: a patch full
     * of faces is two triangles, or a fan from its center when an edge borders a
     * partly filled patch or the section border (that edge is then split at
     * every block); a partly filled patch is two triangles per face. Neighbouring
     * patches of all sections thus share their vertices exactly, and no triangle
     * is thin: Jolt drops every contact with an edge between two coplanar faces
     * by matching their vertices (enhanced internal edge removal), which merged
     * boxes cannot provide where they meet with a T-junction, and slivers make
     * its contact normals drift.
     *
     * @return the number of triangles
     */
    private int buildSurfaceMesh(final SectionScratch s, final GlobalChunk chunk) {
        s.triangleFloats = 0;
        final int[] cls = s.mergeClass;
        final boolean[] full = s.fullCube;
        final int[][] neighbours = new int[6][];
        for (int d = 0; d < 6; d++) {
            final GlobalChunk n = this.globalChunks.get(ChunkSectionData.packSectionPos(chunk.cx + FACE_DX[d], chunk.cy + FACE_DY[d], chunk.cz + FACE_DZ[d]));
            neighbours[d] = n == null ? null : n.data.array();
        }
        final boolean[] mask = s.faceMask;
        final int[] patches = new int[16];
        for (int d = 0; d < 6; d++) {
            final int axis = d >> 1;
            final boolean positive = (d & 1) == 1;
            final int[] across = neighbours[d];
            for (int layer = 0; layer < 16; layer++) {
                final int next = positive ? layer + 1 : layer - 1;
                final boolean border = next < 0 || next > 15;
                boolean any = false;
                for (int v = 0; v < 16; v++) {
                    for (int u = 0; u < 16; u++) {
                        final int idx = cellIndex(axis, layer, u, v);
                        boolean face = cls[idx] != 0;
                        if (face) {
                            if (!border) {
                                face = !full[cellIndex(axis, next, u, v)];
                            } else if (across != null) {
                                final int packed = across[cellIndex(axis, next & 15, u, v)];
                                final int colliderId = ChunkSectionData.colliderIdOf(packed);
                                face = colliderId == 0 || !isFullCube(this.colliderRegistry.get(colliderId - 1));
                            }
                        }
                        mask[u + (v << 4)] = face;
                        any |= face;
                    }
                }
                if (!any) {
                    continue;
                }
                final int plane = positive ? layer + 1 : layer;
                for (int p = 0; p < 16; p++) {
                    final int u0 = (p & 3) << 2, v0 = (p >> 2) << 2;
                    int count = 0;
                    for (int v = v0; v < v0 + 4; v++) {
                        for (int u = u0; u < u0 + 4; u++) {
                            if (mask[u + (v << 4)]) {
                                count++;
                            }
                        }
                    }
                    patches[p] = count;
                }
                // 8x8 quadrants of four full patches become one patch when every
                // patch along their edges is full or empty (a fan from their
                // center over corners and edge midpoints keeps triangles wide).
                int quads = 0;
                for (int q = 0; q < 4; q++) {
                    final int pu0 = (q & 1) << 1, pv0 = (q >> 1) << 1;
                    final int p00 = pu0 + (pv0 << 2);
                    if (patches[p00] != 16 || patches[p00 + 1] != 16 || patches[p00 + 4] != 16 || patches[p00 + 5] != 16) {
                        continue;
                    }
                    boolean balanced = true;
                    for (int k = 0; k < 2 && balanced; k++) {
                        balanced = !partial(this.ringPatch(patches, chunk, d, layer, pu0 - 1, pv0 + k))
                                && !partial(this.ringPatch(patches, chunk, d, layer, pu0 + 2, pv0 + k))
                                && !partial(this.ringPatch(patches, chunk, d, layer, pu0 + k, pv0 - 1))
                                && !partial(this.ringPatch(patches, chunk, d, layer, pu0 + k, pv0 + 2));
                    }
                    if (balanced) {
                        quads |= 1 << q;
                    }
                }
                for (int q = 0; q < 4; q++) {
                    if ((quads & (1 << q)) == 0) {
                        continue;
                    }
                    final int qu = q & 1, qv = q >> 1;
                    final int pu0 = qu << 1, pv0 = qv << 1;
                    final int u0 = pu0 << 2, v0 = pv0 << 2;
                    // An edge gets its midpoint unless the other side is a
                    // quadrant patch of this section, or holds no faces at all.
                    final boolean bottom = qv == 1 && (quads & (1 << (q - 2))) != 0 ? false
                            : this.ringPatch(patches, chunk, d, layer, pu0, pv0 - 1) + this.ringPatch(patches, chunk, d, layer, pu0 + 1, pv0 - 1) != 0;
                    final boolean right = qu == 0 && (quads & (1 << (q + 1))) != 0 ? false
                            : this.ringPatch(patches, chunk, d, layer, pu0 + 2, pv0) + this.ringPatch(patches, chunk, d, layer, pu0 + 2, pv0 + 1) != 0;
                    final boolean top = qv == 0 && (quads & (1 << (q + 2))) != 0 ? false
                            : this.ringPatch(patches, chunk, d, layer, pu0, pv0 + 2) + this.ringPatch(patches, chunk, d, layer, pu0 + 1, pv0 + 2) != 0;
                    final boolean left = qu == 1 && (quads & (1 << (q - 1))) != 0 ? false
                            : this.ringPatch(patches, chunk, d, layer, pu0 - 1, pv0) + this.ringPatch(patches, chunk, d, layer, pu0 - 1, pv0 + 1) != 0;
                    emitPatch(s, axis, positive, plane, u0, v0, 8, bottom, right, top, left, 4);
                }
                for (int p = 0; p < 16; p++) {
                    final int count = patches[p];
                    if (count == 0) {
                        continue;
                    }
                    final int pu = p & 3, pv = p >> 2;
                    if ((quads & (1 << ((pu >> 1) + ((pv >> 1) << 1)))) != 0) {
                        continue;
                    }
                    final int u0 = pu << 2, v0 = pv << 2;
                    if (count < 16) {
                        for (int v = v0; v < v0 + 4; v++) {
                            for (int u = u0; u < u0 + 4; u++) {
                                if (mask[u + (v << 4)]) {
                                    face(s, axis, positive, plane, u, v, u + 1, v, u + 1, v + 1);
                                    face(s, axis, positive, plane, u, v, u + 1, v + 1, u, v + 1);
                                }
                            }
                        }
                        continue;
                    }
                    // An edge is split where the plane goes on in a partly filled
                    // patch, of this section or of the neighbouring one (which
                    // decides the same way about its side of the edge).
                    final boolean bottom = partial(pv == 0 ? this.neighbourPatch(chunk, d, layer, 0, -1, pu, 3) : patches[p - 4]);
                    final boolean right = partial(pu == 3 ? this.neighbourPatch(chunk, d, layer, 1, 0, 0, pv) : patches[p + 1]);
                    final boolean top = partial(pv == 3 ? this.neighbourPatch(chunk, d, layer, 0, 1, pu, 0) : patches[p + 4]);
                    final boolean left = partial(pu == 0 ? this.neighbourPatch(chunk, d, layer, -1, 0, 3, pv) : patches[p - 1]);
                    emitPatch(s, axis, positive, plane, u0, v0, 4, bottom, right, top, left, 1);
                }
            }
        }
        return s.triangleFloats / 9;
    }

    /**
     * Triangulates a full square patch of a face plane: two triangles, or a
     * fan from its center when edges are split (every {@code step} blocks).
     */
    private static void emitPatch(final SectionScratch s, final int axis, final boolean positive, final int plane,
                                  final int u0, final int v0, final int size,
                                  final boolean bottom, final boolean right, final boolean top, final boolean left, final int step) {
        if (!bottom && !right && !top && !left) {
            face(s, axis, positive, plane, u0, v0, u0 + size, v0, u0 + size, v0 + size);
            face(s, axis, positive, plane, u0, v0, u0 + size, v0 + size, u0, v0 + size);
            return;
        }
        final int cu = u0 + (size >> 1), cv = v0 + (size >> 1);
        final int sb = bottom ? step : size, sr = right ? step : size, st = top ? step : size, sl = left ? step : size;
        for (int i = 0; i < size; i += sb) {
            face(s, axis, positive, plane, cu, cv, u0 + i, v0, u0 + i + sb, v0);
        }
        for (int i = 0; i < size; i += sr) {
            face(s, axis, positive, plane, cu, cv, u0 + size, v0 + i, u0 + size, v0 + i + sr);
        }
        for (int i = size; i > 0; i -= st) {
            face(s, axis, positive, plane, cu, cv, u0 + i, v0 + size, u0 + i - st, v0 + size);
        }
        for (int i = size; i > 0; i -= sl) {
            face(s, axis, positive, plane, cu, cv, u0, v0 + i, u0, v0 + i - sl);
        }
    }

    private static boolean partial(final int patchCount) {
        return patchCount != 0 && patchCount != 16;
    }

    /**
     * Face count of a 4x4 patch of the current plane by patch coordinates,
     * which may lie one patch outside this section.
     */
    private int ringPatch(final int[] patches, final GlobalChunk chunk, final int d, final int layer, final int pu, final int pv) {
        if (pu >= 0 && pu < 4 && pv >= 0 && pv < 4) {
            return patches[pu + (pv << 2)];
        }
        return this.neighbourPatch(chunk, d, layer, pu < 0 ? -1 : pu > 3 ? 1 : 0, pv < 0 ? -1 : pv > 3 ? 1 : 0, pu & 3, pv & 3);
    }

    /**
     * Number of faces in a 4x4 patch of the same face plane (direction
     * {@code d}, cell layer {@code layer}) of the section next to {@code chunk}
     * along the plane's u or v axis (0 when that section is not loaded).
     */
    private int neighbourPatch(final GlobalChunk chunk, final int d, final int layer, final int du, final int dv, final int pu, final int pv) {
        final int axis = d >> 1;
        // in-plane axes (u, v) of each face axis: x: (y, z), y: (z, x), z: (x, y)
        final int ox = axis == 1 ? dv : axis == 2 ? du : 0;
        final int oy = axis == 0 ? du : axis == 2 ? dv : 0;
        final int oz = axis == 0 ? dv : axis == 1 ? du : 0;
        final GlobalChunk n = this.globalChunks.get(ChunkSectionData.packSectionPos(chunk.cx + ox, chunk.cy + oy, chunk.cz + oz));
        if (n == null) {
            return 0;
        }
        final int[] data = n.data.array();
        final boolean positive = (d & 1) == 1;
        final int next = positive ? layer + 1 : layer - 1;
        int[] beyond = data;
        if (next < 0 || next > 15) {
            final GlobalChunk b = this.globalChunks.get(ChunkSectionData.packSectionPos(n.cx + FACE_DX[d], n.cy + FACE_DY[d], n.cz + FACE_DZ[d]));
            beyond = b == null ? null : b.data.array();
        }
        int count = 0;
        for (int v = pv << 2; v < (pv << 2) + 4; v++) {
            for (int u = pu << 2; u < (pu << 2) + 4; u++) {
                if (!this.isMeshCube(data[cellIndex(axis, layer, u, v)])) {
                    continue;
                }
                if (beyond == null) {
                    count++;
                    continue;
                }
                final int covering = ChunkSectionData.colliderIdOf(beyond[cellIndex(axis, next & 15, u, v)]);
                if (covering == 0 || !isFullCube(this.colliderRegistry.get(covering - 1))) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Whether a block is part of the terrain surface mesh: the blocks
     * {@link #classifyMergeable} accepts.
     */
    private boolean isMeshCube(final int packed) {
        final int colliderId = ChunkSectionData.colliderIdOf(packed);
        if (colliderId <= 0) {
            return false;
        }
        final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
        final int state = ChunkSectionData.voxelStateOf(packed);
        return entry != null && entry.contactEvents == null && state >= 1 && state <= 4 && isFullCube(entry);
    }

    /**
     * Emits one triangle given counter-clockwise in plane coordinates (u, v),
     * wound to face +axis or -axis.
     */
    private static void face(final SectionScratch s, final int axis, final boolean positive, final int plane,
                             final int u0, final int v0, final int u1, final int v1, final int u2, final int v2) {
        if (positive) {
            planeTriangle(s, axis, plane, u0, v0, u1, v1, u2, v2);
        } else {
            planeTriangle(s, axis, plane, u0, v0, u2, v2, u1, v1);
        }
    }

    /**
     * Index of a section cell given by its coordinate along a face axis and the
     * two in-plane coordinates (u, v), where (u, v, axis) is right-handed:
     * x: (y, z), y: (z, x), z: (x, y).
     */
    private static int cellIndex(final int axis, final int a, final int u, final int v) {
        return switch (axis) {
            case 0 -> ChunkSectionData.index(a, u, v);
            case 1 -> ChunkSectionData.index(v, a, u);
            default -> ChunkSectionData.index(u, v, a);
        };
    }

    private static void planeTriangle(final SectionScratch s, final int axis, final int plane,
                                      final int u0, final int v0, final int u1, final int v1, final int u2, final int v2) {
        switch (axis) {
            case 0 -> s.addTriangle(plane, u0, v0, plane, u1, v1, plane, u2, v2);
            case 1 -> s.addTriangle(v0, plane, u0, v1, plane, u1, v2, plane, u2);
            default -> s.addTriangle(u0, v0, plane, u1, v1, plane, u2, v2, plane);
        }
    }

    // endregion

    /**
     * Builds one section of a body from its chunk data, or returns null when
     * the section holds no collision leaf. Safe to call from worker threads:
     * reads only the chunk stores and the collider registry.
     */
    @Nullable
    private BodySection buildBodySection(final SableBody sb, final long key) {
        final boolean contraption = sb.kind == SableBody.Kind.CONTRAPTION;
        final ChunkSectionData data = contraption ? sb.chunks.get(key) : this.allChunks.get(key);
        if (data == null) {
            return null;
        }
        final int cx = unpackChunkX(key);
        final int cy = unpackChunkY(key);
        final int cz = unpackChunkZ(key);
        final int blockMinX = cx << 4;
        final int blockMinY = cy << 4;
        final int blockMinZ = cz << 4;

        final SectionScratch scratch = SECTION_SCRATCH.get();
        scratch.reset();
        final LeafList leaves = scratch.plain;
        boolean special = false;
        boolean bouncy = false;

        final boolean checkBounds = !contraption;
        final boolean hasBounds = sb.hasBounds;
        final int bMinX = sb.minX, bMinY = sb.minY, bMinZ = sb.minZ;
        final int bMaxX = sb.maxX, bMaxY = sb.maxY, bMaxZ = sb.maxZ;

        String rejectLog = null;
        final int[] blocks = data.array();

        try {
            // NOTE: children stay 1:1 with compound leaf sub-shapes — the ID
            // decoding relies on the leaf order matching this traversal.
            for (int by = 0; by < 16; by++) {
                final int worldY = blockMinY + by;
                final int rowBase = (by << 8);
                for (int bz = 0; bz < 16; bz++) {
                    final int worldZ = blockMinZ + bz;
                    final int cellBase = rowBase + (bz << 4);
                    for (int bx = 0; bx < 16; bx++) {
                        final int packed = blocks[cellBase + bx];
                        if (packed == 0) {
                            continue;
                        }
                        final int colliderId = ChunkSectionData.colliderIdOf(packed);
                        // Single registry lookup per block; reused for boxes + shapes.
                        final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
                        final boolean solid = isSolidBlock(packed, entry);
                        final int worldX = blockMinX + bx;

                        if (JoltDebugLogging.STAFF && colliderId > 0 && (!solid || (checkBounds && !(hasBounds
                                && worldX >= bMinX && worldX <= bMaxX
                                && worldY >= bMinY && worldY <= bMaxY
                                && worldZ >= bMinZ && worldZ <= bMaxZ)))) {
                            final String reason = !solid
                                    ? "state=" + ChunkSectionData.voxelStateOf(packed) + (entry == null ? " entry=null" : " boxes=" + entry.boxes.size())
                                    : "outOfBounds";
                            rejectLog = appendReject(rejectLog, worldX, worldY, worldZ, reason);
                        }

                        if (checkBounds && !(hasBounds
                                && worldX >= bMinX && worldX <= bMaxX
                                && worldY >= bMinY && worldY <= bMaxY
                                && worldZ >= bMinZ && worldZ <= bMaxZ)) {
                            continue;
                        }
                        if (solid) {
                            final int blockIndex = scratch.blockCount;
                            scratch.ensureBlocks(blockIndex + 1);
                            scratch.blocks[blockIndex * 3] = worldX;
                            scratch.blocks[blockIndex * 3 + 1] = worldY;
                            scratch.blocks[blockIndex * 3 + 2] = worldZ;
                            scratch.volumes[blockIndex] = entry.volume;
                            special |= entry.needsSpecialContacts();
                            bouncy |= entry.restitution > 0.0f;
                            scratch.blockCount = blockIndex + 1;
                        }
                        // Full cubes become merged boxes after the scan.
                        if (classifyMergeable(scratch, cellBase + bx, packed, colliderId, entry) || !solid) {
                            continue;
                        }

                        final ObjectArrayList<JoltVoxelColliderData.VoxelBox> secBoxes = entry.boxes;
                        final int elementsSize = secBoxes.size();
                        final Object[] elements = secBoxes.elements();

                        for (int i = 0; i < elementsSize; i++) {
                            final JoltVoxelColliderData.VoxelBox box =
                                    (JoltVoxelColliderData.VoxelBox) elements[i];
                            // Offsets relative to the section corner: small floats,
                            // exact even at plot coordinates in the millions.
                            leaves.add(bx + (box.minX + box.maxX) * 0.5f, by + (box.minY + box.maxY) * 0.5f, bz + (box.minZ + box.maxZ) * 0.5f,
                                    entry.shape(i), new Child(worldX, worldY, worldZ, colliderId, sb));
                        }
                    }
                }
            }
            this.emitMergedBoxes(scratch, leaves, blockMinX, blockMinY, blockMinZ, sb, false, true, 0.0f);

            if (JoltDebugLogging.STAFF && rejectLog != null) {
                Sable.LOGGER.info("[SableJolt] rebuild body {}: rejected: {}", sb.runtimeId, rejectLog);
            }
            final int leafCount = leaves.count;
            if (leafCount == 0) {
                return null;
            }

            final Child[] children = Arrays.copyOf(leaves.children, leafCount);
            final int[] uniqueBlocks = Arrays.copyOf(scratch.blocks, scratch.blockCount * 3);
            final float[] volumes = Arrays.copyOf(scratch.volumes, scratch.blockCount);

            if (leafCount == 1) {
                // A single leaf goes into the parent directly: Jolt's compound
                // create() would return the leaf itself and drop its offset.
                return new BodySection(key, leaves.shapes[0], null,
                        blockMinX + (double) leaves.ox[0], blockMinY + (double) leaves.oy[0], blockMinZ + (double) leaves.oz[0],
                        children, 0, uniqueBlocks, volumes, special, bouncy);
            }

            final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
            for (int i = 0; i < leafCount; i++) {
                settings.addShape(leaves.ox[i], leaves.oy[i], leaves.oz[i], leaves.shapes[i]);
            }
            final ShapeResult result = settings.create();
            if (result.hasError()) {
                Sable.LOGGER.error("[SableJolt] body {}: section ({},{},{}) shape failed: {}",
                        sb.runtimeId, cx, cy, cz, result.getError());
                return null;
            }
            final ShapeRefC ref = result.get();
            // The compound keeps the settings' frame (leaves relative to the
            // section corner); a parent places that frame's origin, so the
            // section origin is the corner itself.
            return new BodySection(key, ref.getPtr(), ref,
                    blockMinX, blockMinY, blockMinZ,
                    children, subShapeBits(leafCount), uniqueBlocks, volumes, special, bouncy);
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] body {}: failed to build section ({},{},{})", sb.runtimeId, cx, cy, cz, t);
            return null;
        } finally {
            scratch.releaseRefs();
        }
    }

    private static String appendReject(final String log, final int x, final int y, final int z, final String reason) {
        final String entry = "block(" + x + "," + y + "," + z + ") " + reason;
        return log == null ? entry : log + "; " + entry;
    }

    //endregion

    //region Chunks

    /**
     * Stores a chunk section. Global sections get (or refresh) their terrain
     * shape; plot sections mark the owning sub-level's section dirty.
     *
     * @param ownerId runtime id of the sub-level owning a plot section (-1 if unknown)
     */
    public void addChunk(final int x, final int y, final int z, final int[] data, final boolean global, final int ownerId) {
        if (this.disposed) {
            return;
        }
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        final ChunkSectionData existing = this.allChunks.get(key);
        if (global && existing != null && Arrays.equals(existing.array(), data)) {
            // Re-upload of identical terrain (e.g. Sable adopting a section the
            // pipeline uploaded ahead of it): keep the shape as is.
            final GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk != null) {
                return;
            }
        }

        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        this.allChunks.put(key, section);

        if (global) {
            this.clearSectionOwner(key);
            GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk == null) {
                chunk = new GlobalChunk(x, y, z, section);
                this.globalChunks.put(key, chunk);
            } else {
                chunk.data = section;
            }
            // Built before the next step, once even if its neighbours arrive in
            // the same tick: their surface meshes depend on each other.
            this.dirtyGlobalChunks.add(chunk);
            this.markNeighboursDirty(x, y, z);
            if (JoltDebugLogging.STAFF) {
                Sable.LOGGER.info("[SableJolt] addChunk GLOBAL at ({},{},{}) totalGlobal={}", x, y, z, this.globalChunks.size());
            }
            return;
        }

        this.removeGlobalChunk(key);
        if (ownerId != -1) {
            this.setSectionOwner(key, ownerId);
            final SableBody sb = this.bodies.get(ownerId);
            if (sb != null && sb.kind == SableBody.Kind.SUB_LEVEL) {
                sb.dirtySections.add(key);
                this.markDirty(sb);
            }
            return;
        }

        // Owner unknown: rebuild every sub-level whose bounds cover the section,
        // mirroring the rapier shared chunk store behavior.
        for (final SableBody sb : this.bodyList) {
            if (sb.kind != SableBody.Kind.SUB_LEVEL || !sb.hasBounds) {
                continue;
            }
            final boolean intersects = (x << 4) <= sb.maxX && ((x << 4) + 15) >= sb.minX
                    && (y << 4) <= sb.maxY && ((y << 4) + 15) >= sb.minY
                    && (z << 4) <= sb.maxZ && ((z << 4) + 15) >= sb.minZ;
            if (intersects) {
                sb.dirtySections.add(key);
                this.markDirty(sb);
            }
        }
    }

    /**
     * Marks a body's shape for rebuild; the rebuild itself is deferred to the next
     * simulation step so bursts of block edits cost a single rebuild.
     */
    private void markDirty(final SableBody sb) {
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.add(sb);
        }
    }

    private void markMountDirty(final SableBody contraption) {
        if (!contraption.embeddedInMount) {
            return;
        }
        final SableBody mount = this.bodies.get(contraption.mountId);
        if (mount != null) {
            mount.parentDirty = true;
            mount.wakeOnAssemble = true;
            this.markDirty(mount);
        }
    }

    private long flushCounter;

    private void flushDirty() {
        this.flushCreativeRestore();
        if (!this.creativeDragBodies.isEmpty()) {
            for (final int joltId : this.creativeDragBodies) {
                final SableBody sb = this.bodyByJoltId(joltId);
                if (sb != null) {
                    sb.dragPenDepth = 0.0;
                }
            }
        }
        this.flushCounter++;
        if ((this.flushCounter & 31) == 0) {
            this.retryPendingCreation();
            this.retryRegionBodies();
        }
        if (!this.dirtyBodies.isEmpty()) {
            final ObjectArrayList<SableBody> list;
            synchronized (this.dirtyBodies) {
                list = new ObjectArrayList<>(this.dirtyBodies);
                this.dirtyBodies.clear();
            }
            // Contraptions first: mounts embed their freshly assembled shapes.
            final ObjectArrayList<SableBody> contraptions = new ObjectArrayList<>();
            final ObjectArrayList<SableBody> others = new ObjectArrayList<>(list.size());
            for (final SableBody sb : list) {
                if (this.bodies.get(sb.runtimeId) != sb) {
                    continue; // removed since it was marked
                }
                (sb.kind == SableBody.Kind.CONTRAPTION ? contraptions : others).add(sb);
            }
            if (!contraptions.isEmpty()) {
                this.rebuildAll(contraptions);
                for (final SableBody c : contraptions) {
                    if (c.embeddedInMount) {
                        final SableBody mount = this.bodies.get(c.mountId);
                        if (mount != null && !others.contains(mount)) {
                            mount.parentDirty = true;
                            mount.wakeOnAssemble = true;
                            others.add(mount);
                        }
                    }
                }
            }
            this.rebuildAll(others);
        }
        this.updateMeshedSections();
        if (!this.dirtyGlobalChunks.isEmpty()) {
            this.refreshDirtyGlobalChunks();
        }
        this.flushTerrain();
    }

    /**
     * Gives the terrain sections around awake bodies their surface mesh (see
     * {@link #MESH_MARGIN}) and takes it back from sections no body came near
     * for a while. A body refreshes its sections when it moved a few blocks
     * since it last did, or every half keep time.
     */
    private void updateMeshedSections() {
        final long now = this.flushCounter;
        final LongArrayList upgrades = this.meshUpgrades;
        final it.unimi.dsi.fastutil.doubles.DoubleArrayList distances = this.meshUpgradeDistances;
        upgrades.clear();
        distances.clear();
        boolean pending = false;
        for (final SableBody sb : this.bodyList) {
            if (sb.body == null || !sb.snapActive || (sb.kind != SableBody.Kind.SUB_LEVEL && sb.kind != SableBody.Kind.BOX)) {
                continue;
            }
            final double dx = sb.snapComX - sb.meshedAtX, dy = sb.snapComY - sb.meshedAtY, dz = sb.snapComZ - sb.meshedAtZ;
            if (!sb.meshPending && dx * dx + dy * dy + dz * dz < 16.0 && now - sb.meshedAtFlush < MESH_KEEP_STEPS / 2) {
                continue;
            }
            sb.meshedAtX = sb.snapComX;
            sb.meshedAtY = sb.snapComY;
            sb.meshedAtZ = sb.snapComZ;
            sb.meshedAtFlush = now;
            sb.meshPending = false;
            final double r = sb.boundRadius + MESH_MARGIN;
            final double look = MESH_LOOKAHEAD, cap = MESH_MAX_LOOKAHEAD;
            final double ax = Math.max(-cap, Math.min(cap, sb.snapVx * look));
            final double ay = Math.max(-cap, Math.min(cap, sb.snapVy * look));
            final double az = Math.max(-cap, Math.min(cap, sb.snapVz * look));
            final int x0 = floor(sb.snapComX + Math.min(0.0, ax) - r) >> 4, x1 = floor(sb.snapComX + Math.max(0.0, ax) + r) >> 4;
            final int y0 = floor(sb.snapComY + Math.min(0.0, ay) - r) >> 4, y1 = floor(sb.snapComY + Math.max(0.0, ay) + r) >> 4;
            final int z0 = floor(sb.snapComZ + Math.min(0.0, az) - r) >> 4, z1 = floor(sb.snapComZ + Math.max(0.0, az) + r) >> 4;
            for (int x = x0; x <= x1; x++) {
                for (int y = y0; y <= y1; y++) {
                    for (int z = z0; z <= z1; z++) {
                        final long key = ChunkSectionData.packSectionPos(x, y, z);
                        if (!this.globalChunks.containsKey(key)) {
                            continue;
                        }
                        if (this.meshedSections.containsKey(key)) {
                            this.meshedSections.put(key, now + MESH_KEEP_STEPS);
                            continue;
                        }
                        final double ex = (x << 4) + 8 - sb.snapComX, ey = (y << 4) + 8 - sb.snapComY, ez = (z << 4) + 8 - sb.snapComZ;
                        upgrades.add(key);
                        distances.add(ex * ex + ey * ey + ez * ez);
                        sb.meshPending = true;
                    }
                }
            }
        }
        if (!upgrades.isEmpty()) {
            // nearest first, a bounded number per step; bodies with sections left
            // over come back next step (meshPending)
            final int n = upgrades.size();
            final int[] order = new int[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            it.unimi.dsi.fastutil.ints.IntArrays.quickSort(order, (a, b) -> Double.compare(distances.getDouble(a), distances.getDouble(b)));
            int added = 0;
            for (int i = 0; i < n && added < MESH_UPGRADES_PER_FLUSH; i++) {
                final long key = upgrades.getLong(order[i]);
                if (this.meshedSections.containsKey(key)) {
                    continue;
                }
                this.meshedSections.put(key, now + MESH_KEEP_STEPS);
                this.dirtyGlobalChunks.add(this.globalChunks.get(key));
                added++;
            }
            pending = added < n;
        }
        if (!pending) {
            for (final SableBody sb : this.bodyList) {
                sb.meshPending = false;
            }
        }
        if ((now & 63) == 0 && !this.meshedSections.isEmpty()) {
            final var it = this.meshedSections.long2LongEntrySet().fastIterator();
            while (it.hasNext()) {
                final var e = it.next();
                if (e.getLongValue() < now) {
                    final GlobalChunk chunk = this.globalChunks.get(e.getLongKey());
                    it.remove();
                    if (chunk != null) {
                        this.dirtyGlobalChunks.add(chunk);
                    }
                }
            }
        }
    }

    private void rebuildAll(final ObjectArrayList<SableBody> list) {
        if (list.size() >= 8) {
            // One body per task; sections stay sequential inside (no nested pool use).
            this.runParallel(list, sb -> {
                try {
                    this.rebuildShape(sb, false);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                }
            });
        } else {
            for (final SableBody sb : list) {
                try {
                    this.rebuildShape(sb, true);
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] failed to rebuild body {}", sb.runtimeId, t);
                }
            }
        }
    }

    public boolean hasChunk(final int sectionX, final int sectionY, final int sectionZ) {
        return this.allChunks.containsKey(ChunkSectionData.packSectionPos(sectionX, sectionY, sectionZ));
    }

    public void removeChunk(final int x, final int y, final int z, final boolean global) {
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        this.allChunks.remove(key);

        if (global) {
            this.removeGlobalChunk(key);
            return;
        }
        final int owner = this.clearSectionOwner(key);
        final SableBody sb = owner == -1 ? null : this.bodies.get(owner);
        if (sb != null && sb.kind == SableBody.Kind.SUB_LEVEL) {
            sb.dirtySections.add(key);
            this.markDirty(sb);
        }
    }

    private void removeGlobalChunk(final long key) {
        final GlobalChunk chunk = this.globalChunks.remove(key);
        if (chunk == null) {
            return;
        }
        this.fluidSections.remove(key);
        this.specialSections.remove(key);
        this.bouncySections.remove(key);
        // a removed chunk must not be touched by a pending flush
        this.dirtyGlobalChunks.remove(chunk);
        final TerrainTile tile = chunk.tile;
        if (tile != null) {
            tile.members.remove(chunk);
            chunk.tile = null;
            this.dirtyTiles.add(tile);
        }
        // the neighbours' faces against this section are exposed now
        this.markNeighboursDirty(chunk.cx, chunk.cy, chunk.cz);
    }

    /**
     * Queues the 26 neighbours of a terrain section for a rebuild: their surface
     * meshes hide the faces against full cubes of this section, and shape the
     * patches along their borders after the patches of this section (and of
     * its neighbours, see {@link #neighbourPatch}).
     */
    private void markNeighboursDirty(final int cx, final int cy, final int cz) {
        for (int ox = -1; ox <= 1; ox++) {
            for (int oy = -1; oy <= 1; oy++) {
                for (int oz = -1; oz <= 1; oz++) {
                    if (ox != 0 || oy != 0 || oz != 0) {
                        this.markSectionDirty(cx + ox, cy + oy, cz + oz);
                    }
                }
            }
        }
    }

    /**
     * Queues the neighbours of a terrain section whose surface mesh depends on
     * the block at section-local (lx, ly, lz): those within one patch (4 blocks)
     * of it across their border.
     */
    private void markNeighboursDirty(final int cx, final int cy, final int cz, final int lx, final int ly, final int lz) {
        final int x0 = lx < 4 ? -1 : 0, x1 = lx > 11 ? 1 : 0;
        final int y0 = ly < 4 ? -1 : 0, y1 = ly > 11 ? 1 : 0;
        final int z0 = lz < 4 ? -1 : 0, z1 = lz > 11 ? 1 : 0;
        for (int ox = x0; ox <= x1; ox++) {
            for (int oy = y0; oy <= y1; oy++) {
                for (int oz = z0; oz <= z1; oz++) {
                    if (ox != 0 || oy != 0 || oz != 0) {
                        this.markSectionDirty(cx + ox, cy + oy, cz + oz);
                    }
                }
            }
        }
    }

    private void markSectionDirty(final int cx, final int cy, final int cz) {
        final GlobalChunk chunk = this.globalChunks.get(ChunkSectionData.packSectionPos(cx, cy, cz));
        if (chunk != null && this.meshedSections.containsKey(chunk.key)) {
            this.dirtyGlobalChunks.add(chunk);
        }
    }

    public void changeBlock(final int x, final int y, final int z, final int newState) {
        this.changeBlocks(new int[]{x}, new int[]{y}, new int[]{z}, new int[]{newState}, null, 1);
    }

    /**
     * Applies block edits. Each edit goes to the owner of its section: the
     * sub-level owning the plot section (only that section is rebuilt), or the
     * terrain body. Edits in sections no sub-level needs are ignored, like the
     * rapier pipeline does.
     *
     * @param owners per edit, the runtime id of the sub-level whose plot holds
     *               the block, or -1 for the world; null to use the recorded owners
     */
    public void changeBlocks(final int[] wx, final int[] wy, final int[] wz,
                             final int[] packed, @Nullable final int[] owners, final int count) {
        if (this.disposed) {
            return;
        }
        for (int i = 0; i < count; i++) {
            final long key = ChunkSectionData.packSectionPos(wx[i] >> 4, wy[i] >> 4, wz[i] >> 4);
            int owner = owners != null ? owners[i] : this.sectionOwners.getOrDefault(key, -1);
            ChunkSectionData data = this.allChunks.get(key);
            SableBody sb = owner == -1 ? null : this.bodies.get(owner);
            if (sb != null && sb.kind != SableBody.Kind.SUB_LEVEL) {
                sb = null;
            }
            if (data == null) {
                if (sb == null) {
                    continue;
                }
                // A plot section without data starts out as air: re-create it so
                // blocks placed into a reused plot slot always collide.
                data = new ChunkSectionData();
                this.allChunks.put(key, data);
            }
            data.set(wx[i] & 15, wy[i] & 15, wz[i] & 15, packed[i]);

            if (sb != null) {
                this.setSectionOwner(key, owner);
                sb.dirtySections.add(key);
                this.markDirty(sb);
                continue;
            }
            final GlobalChunk chunk = this.globalChunks.get(key);
            if (chunk != null) {
                this.dirtyGlobalChunks.add(chunk);
                // Blocks near the section border show or hide faces of the
                // neighbouring sections' surface meshes, or reshape their patches.
                this.markNeighboursDirty(wx[i] >> 4, wy[i] >> 4, wz[i] >> 4, wx[i] & 15, wy[i] & 15, wz[i] & 15);
            }
        }
    }

    /**
     * Legacy overload without owners: uses the recorded section owners.
     */
    public void changeBlocks(final int[] wx, final int[] wy, final int[] wz,
                             final int[] packed, final int count) {
        this.changeBlocks(wx, wy, wz, packed, null, count);
    }

    /**
     * Rebuilds a terrain section's geometry from its data, local to the section
     * corner, in two layers. Blocks without a contact callback: a surface mesh
     * of the full cubes' exposed faces (see {@link #buildSurfaceMesh}) over the
     * same cubes merged into boxes and inset by {@link #TERRAIN_RECESS}, plus
     * one box per collision box of other solid blocks. Their friction and
     * restitution multipliers are applied by the contact listener per block
     * pair ({@link SceneContactListener#sampleBlockPairs}). Blocks with a
     * contact callback: one box per collision box, in the region's second body.
     */
    private int buildGlobalChunkShape(final GlobalChunk chunk) {
        final SectionScratch scratch = SECTION_SCRATCH.get();
        scratch.reset();

        final int baseX = chunk.cx << 4;
        final int baseY = chunk.cy << 4;
        final int baseZ = chunk.cz << 4;
        final int[] data = chunk.data.array();

        boolean anyFluid = false;
        boolean special = false;
        boolean bouncy = false;
        int plainCollider = 0;
        int specialCollider = 0;
        try {
            // NOTE: children stay 1:1 with compound leaf sub-shapes, in this order.
            for (int idx = 0; idx < ChunkSectionData.BLOCKS; idx++) {
                final int packed = data[idx];
                scratch.fullCube[idx] = false;
                if (packed == 0) {
                    continue;
                }
                final int colliderId = ChunkSectionData.colliderIdOf(packed);
                final JoltVoxelColliderData entry = colliderId == 0 ? null : this.colliderRegistry.get(colliderId - 1);
                if (entry == null) {
                    continue;
                }
                // fluids are marked for buoyancy even though they never collide
                if (entry.isFluid) {
                    anyFluid = true;
                }
                scratch.fullCube[idx] = isFullCube(entry);
                if (classifyMergeable(scratch, idx, packed, colliderId, entry)) {
                    if (scratch.mergeClass[idx] == MERGE_DEFAULT) {
                        if (plainCollider == 0) {
                            plainCollider = colliderId;
                        }
                    } else {
                        special = true;
                        bouncy |= entry.restitution > 0.0f;
                        if (specialCollider == 0) {
                            specialCollider = colliderId;
                        }
                    }
                    continue;
                }
                if (!isSolidBlock(packed, entry)) {
                    continue;
                }
                special |= entry.contactEvents == null && entry.needsSpecialContacts();
                bouncy |= entry.restitution > 0.0f;
                final LeafList leaves = entry.contactEvents != null ? scratch.special : scratch.plain;
                final int bx = idx & 15;
                final int bz = (idx >> 4) & 15;
                final int by = idx >> 8;
                final ObjectArrayList<JoltVoxelColliderData.VoxelBox> boxes = entry.boxes;
                final int size = boxes.size();
                final Object[] elements = boxes.elements();
                for (int i = 0; i < size; i++) {
                    final JoltVoxelColliderData.VoxelBox box = (JoltVoxelColliderData.VoxelBox) elements[i];
                    leaves.add(bx + (box.minX + box.maxX) * 0.5f, by + (box.minY + box.maxY) * 0.5f, bz + (box.minZ + box.maxZ) * 0.5f,
                            entry.shape(i), new Child(baseX + bx, baseY + by, baseZ + bz, colliderId, null));
                }
            }
            final int meshCollider = plainCollider != 0 ? plainCollider : specialCollider;
            if (meshCollider != 0 && this.meshedSections.containsKey(chunk.key)) {
                // The mesh carries the contacts of full cubes; under it the boxes
                // only hold deep penetrations: merged whatever the block, plain
                // greedy merging (T-junctions do not matter there).
                final int triangles = this.buildSurfaceMesh(scratch, chunk);
                this.emitMergedBoxes(scratch, scratch.plain, baseX, baseY, baseZ, null, true, false, TERRAIN_RECESS);
                this.assembleTerrainPart(chunk, chunk.parts[TERRAIN_PLAIN], scratch.plain, scratch, triangles, meshCollider);
            } else {
                // Away from moving bodies: full cubes merged per contact behavior
                // into boxes that are the contact surface themselves.
                this.emitMergedBoxes(scratch, scratch.plain, baseX, baseY, baseZ, null, false, true, 0.0f);
                this.assembleTerrainPart(chunk, chunk.parts[TERRAIN_PLAIN], scratch.plain, scratch, 0, 0);
            }
            this.assembleTerrainPart(chunk, chunk.parts[TERRAIN_SPECIAL], scratch.special, scratch, 0, 0);
            return (anyFluid ? SECTION_FLUID : 0) | (special ? SECTION_SPECIAL : 0) | (bouncy ? SECTION_BOUNCY : 0);
        } finally {
            scratch.releaseRefs();
        }
    }

    /**
     * Creates one terrain layer of a section from its box leaves and (plain
     * layer) the surface mesh triangles in the scratch.
     */
    private void assembleTerrainPart(final GlobalChunk chunk, final TerrainPart part, final LeafList leaves,
                                     final SectionScratch scratch, final int triangles, final int meshCollider) {
        part.clear();
        final int baseX = chunk.cx << 4;
        final int baseY = chunk.cy << 4;
        final int baseZ = chunk.cz << 4;

        // Surface meshes, at most MAX_MESH_TRIANGLES each (sub-shape id bits).
        final int meshCount = (triangles + MAX_MESH_TRIANGLES - 1) / MAX_MESH_TRIANGLES;
        final ShapeRefC[] meshes = new ShapeRefC[meshCount];
        int built = 0;
        for (int m = 0; m < meshCount; m++) {
            final int first = m * MAX_MESH_TRIANGLES;
            final int count = Math.min(MAX_MESH_TRIANGLES, triangles - first);
            final FloatBuffer buffer = Jolt.newDirectFloatBuffer(count * 9);
            buffer.put(scratch.triangles, first * 9, count * 9).flip();
            final ShapeResult result = new MeshShapeSettings(buffer).create();
            if (result.hasError()) {
                Sable.LOGGER.error("[SableJolt] terrain section ({},{},{}) surface mesh failed: {}", chunk.cx, chunk.cy, chunk.cz, result.getError());
                continue;
            }
            meshes[built++] = result.get();
        }
        final Child[] meshChildren = new Child[built];
        for (int m = 0; m < built; m++) {
            meshChildren[m] = new Child(baseX, baseY, baseZ, baseX + 15, baseY + 15, baseZ + 15, meshCollider, null);
        }

        // The box part: one wrapped box, or a compound of boxes in the section frame.
        final int n = leaves.count;
        ConstShape boxes = null;
        ShapeRefC boxesRef = null;
        if (n >= 2) {
            final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
            for (int i = 0; i < n; i++) {
                settings.addShape(leaves.ox[i], leaves.oy[i], leaves.oz[i], leaves.shapes[i]);
            }
            final ShapeResult result = settings.create();
            if (result.hasError()) {
                Sable.LOGGER.error("[SableJolt] terrain section ({},{},{}) shape failed: {}", chunk.cx, chunk.cy, chunk.cz, result.getError());
                return;
            }
            boxesRef = result.get();
            boxes = boxesRef.getPtr();
        }
        final Child[] children = n == 0 ? NO_CHILDREN : Arrays.copyOf(leaves.children, n);

        if (built == 0) {
            if (n == 0) {
                return;
            }
            part.children = children;
            if (n == 1) {
                // A single leaf cannot carry an offset on its own, and a static
                // compound of one child collapses into it: wrap it.
                final Vec3 offset = this.sectionVecScratch.get();
                offset.set(leaves.ox[0], leaves.oy[0], leaves.oz[0]);
                final MutableCompoundShape single = new MutableCompoundShape();
                single.addShape(offset, IDENTITY_QUAT, leaves.shapes[0]);
                part.shape = single;
                return;
            }
            part.shape = boxes;
            part.ref = boxesRef;
            part.leafBits = subShapeBits(n);
            return;
        }
        part.meshChildren = meshChildren;
        if (built == 1 && n == 0) {
            // The mesh is already in the section frame.
            part.ref = meshes[0];
            part.shape = meshes[0].getPtr();
            return;
        }
        // Meshes first, then the box part: a mesh's triangle ids then sit above
        // the few part bits instead of above the leaf bits (sub-shape ids are
        // 32 bits), and the leaf ids stay exact.
        final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
        for (int m = 0; m < built; m++) {
            settings.addShape(0.0f, 0.0f, 0.0f, meshes[m].getPtr());
        }
        if (n == 1) {
            settings.addShape(leaves.ox[0], leaves.oy[0], leaves.oz[0], leaves.shapes[0]);
        } else if (n >= 2) {
            settings.addShape(0.0f, 0.0f, 0.0f, boxes);
        }
        final ShapeResult result = settings.create();
        if (result.hasError()) {
            Sable.LOGGER.error("[SableJolt] terrain section ({},{},{}) shape failed: {}", chunk.cx, chunk.cy, chunk.cz, result.getError());
            part.clear();
            return;
        }
        part.ref = result.get();
        part.shape = part.ref.getPtr();
        part.children = children;
        part.leafBits = subShapeBits(n);
        part.partBits = subShapeBits(built + (n > 0 ? 1 : 0));
    }

    /**
     * Rebuilds a terrain section's shape and queues its tile for reassembly.
     */
    /**
     * Section kinds {@link #buildGlobalChunkShape} reports.
     */
    private static final int SECTION_FLUID = 1;
    private static final int SECTION_SPECIAL = 2;
    private static final int SECTION_BOUNCY = 4;

    /**
     * Rebuilds the dirty terrain sections (on the worker pool when there are
     * many: a section only reads the chunk stores and the collider registry)
     * and queues their tiles for reassembly.
     */
    private void refreshDirtyGlobalChunks() {
        final GlobalChunk[] chunks = new GlobalChunk[this.dirtyGlobalChunks.size()];
        int n = 0;
        for (final GlobalChunk chunk : this.dirtyGlobalChunks) {
            // Staleness guard: a chunk that was removed can linger in dirtyGlobalChunks.
            if (this.globalChunks.get(chunk.key) == chunk) {
                chunks[n++] = chunk;
            }
        }
        this.dirtyGlobalChunks.clear();
        final int[] kinds = new int[n];
        final IntConsumer build = i -> {
            try {
                kinds[i] = this.buildGlobalChunkShape(chunks[i]);
            } catch (final Throwable t) {
                kinds[i] = -1;
                Sable.LOGGER.error("[SableJolt] failed to refresh global chunk ({},{},{})", chunks[i].cx, chunks[i].cy, chunks[i].cz, t);
            }
        };
        if (n >= 8) {
            runParallelIndices(n, build);
        } else {
            for (int i = 0; i < n; i++) {
                build.accept(i);
            }
        }
        for (int i = 0; i < n; i++) {
            final GlobalChunk chunk = chunks[i];
            final int kind = kinds[i];
            if (kind < 0) {
                continue;
            }
            setMember(this.fluidSections, chunk.key, (kind & SECTION_FLUID) != 0);
            setMember(this.specialSections, chunk.key, (kind & SECTION_SPECIAL) != 0);
            setMember(this.bouncySections, chunk.key, (kind & SECTION_BOUNCY) != 0);
            TerrainTile tile = chunk.tile;
            if (tile == null) {
                tile = this.tileFor(chunk.cx, chunk.cy, chunk.cz);
                tile.members.add(chunk);
                chunk.tile = tile;
            }
            this.dirtyTiles.add(tile);
        }
    }

    private static void setMember(final LongOpenHashSet set, final long key, final boolean member) {
        if (member) {
            set.add(key);
        } else {
            set.remove(key);
        }
    }

    private TerrainTile tileFor(final int cx, final int cy, final int cz) {
        final int tx = cx >> TILE_SHIFT;
        final int ty = cy >> TILE_SHIFT;
        final int tz = cz >> TILE_SHIFT;
        final long tileKey = ChunkSectionData.packSectionPos(tx, ty, tz);
        TerrainTile tile = this.terrainTiles.get(tileKey);
        if (tile == null) {
            final int rx = (cx + REGION_HALF_SECTIONS) >> REGION_SECTION_SHIFT;
            final int rz = (cz + REGION_HALF_SECTIONS) >> REGION_SECTION_SHIFT;
            final long regionKey = ((long) rx << 32) | (rz & 0xFFFFFFFFL);
            TerrainRegion region = this.terrainRegions.get(regionKey);
            if (region == null) {
                region = new TerrainRegion(regionKey, rx, rz);
                this.terrainRegions.put(regionKey, region);
            }
            tile = new TerrainTile(tileKey, region, tx, ty, tz);
            this.terrainTiles.put(tileKey, tile);
            region.tiles.add(tile);
        }
        return tile;
    }

    /**
     * Reassembles the tiles changed since the last step and the regions they
     * belong to. Runs on the server thread before the native update.
     */
    private void flushTerrain() {
        if (this.dirtyTiles.isEmpty()) {
            return;
        }
        final ReferenceOpenHashSet<TerrainRegion> regions = new ReferenceOpenHashSet<>();
        for (final TerrainTile tile : this.dirtyTiles) {
            try {
                this.assembleTile(tile);
            } catch (final Throwable t) {
                Sable.LOGGER.error("[SableJolt] failed to assemble terrain tile ({},{},{})", tile.tx, tile.ty, tile.tz, t);
            }
            regions.add(tile.region);
        }
        this.dirtyTiles.clear();
        for (final TerrainRegion region : regions) {
            try {
                this.assembleRegion(region);
            } catch (final Throwable t) {
                Sable.LOGGER.error("[SableJolt] failed to assemble terrain region ({},{})", region.rx, region.rz, t);
            }
        }
    }

    /**
     * Builds a tile's compounds from the parts of its sections. A tile without
     * sections is forgotten.
     */
    private void assembleTile(final TerrainTile tile) {
        if (tile.members.isEmpty()) {
            this.terrainTiles.remove(tile.key);
            tile.region.tiles.remove(tile);
        }
        for (int layer = 0; layer < tile.layers.length; layer++) {
            this.assembleTileLayer(tile, layer);
        }
    }

    private void assembleTileLayer(final TerrainTile tile, final int layer) {
        final TileLayer tl = tile.layers[layer];
        final ObjectArrayList<GlobalChunk> parts = new ObjectArrayList<>(tile.members.size());
        for (final GlobalChunk chunk : tile.members) {
            if (chunk.parts[layer].shape != null) {
                parts.add(chunk);
            }
        }
        tl.ref = null;
        if (parts.isEmpty()) {
            tl.shape = null;
            tl.assembled = new GlobalChunk[0];
            return;
        }
        // Deterministic sub-shape order.
        parts.sort((a, b) -> Long.compare(a.key, b.key));
        final GlobalChunk[] assembled = parts.toArray(new GlobalChunk[0]);
        final int tileX = tile.tx << (TILE_SHIFT + 4);
        final int tileY = tile.ty << (TILE_SHIFT + 4);
        final int tileZ = tile.tz << (TILE_SHIFT + 4);
        if (assembled.length == 1) {
            // A static compound of one child would collapse into it.
            final GlobalChunk only = assembled[0];
            final Vec3 offset = this.sectionVecScratch.get();
            offset.set((only.cx << 4) - tileX, (only.cy << 4) - tileY, (only.cz << 4) - tileZ);
            final MutableCompoundShape single = new MutableCompoundShape();
            single.addShape(offset, IDENTITY_QUAT, only.parts[layer].shape);
            tl.shape = single;
        } else {
            final StaticCompoundShapeSettings settings = new StaticCompoundShapeSettings();
            for (final GlobalChunk chunk : assembled) {
                settings.addShape((chunk.cx << 4) - tileX, (chunk.cy << 4) - tileY, (chunk.cz << 4) - tileZ, chunk.parts[layer].shape);
            }
            final ShapeResult result = settings.create();
            if (result.hasError()) {
                Sable.LOGGER.error("[SableJolt] terrain tile ({},{},{}) shape failed: {}", tile.tx, tile.ty, tile.tz, result.getError());
                tl.shape = null;
                tl.assembled = new GlobalChunk[0];
                return;
            }
            tl.ref = result.get();
            tl.shape = tl.ref.getPtr();
        }
        tl.assembled = assembled;
    }

    /**
     * Rebuilds a region's compounds from its tiles and installs them on the
     * region's static bodies (created on first use, destroyed when empty).
     * Replacing a shape also drops the contact cache of the body, so bodies
     * resting on changed terrain collide with its new geometry.
     */
    private void assembleRegion(final TerrainRegion region) {
        for (final RegionLayer rl : region.layers) {
            this.assembleRegionLayer(rl);
        }
        if (region.tiles.isEmpty()) {
            boolean bodies = false;
            for (final RegionLayer rl : region.layers) {
                bodies |= rl.body != null;
            }
            if (!bodies) {
                this.terrainRegions.remove(region.key);
            }
        }
    }

    private void assembleRegionLayer(final RegionLayer rl) {
        final TerrainRegion region = rl.region;
        final ObjectArrayList<TerrainTile> parts = new ObjectArrayList<>(region.tiles.size());
        for (final TerrainTile tile : region.tiles) {
            if (tile.layers[rl.layer].shape != null) {
                parts.add(tile);
            }
        }
        if (parts.isEmpty()) {
            this.destroyRegionBody(rl);
            this.regionsAwaitingBody.remove(rl);
            return;
        }
        final MutableCompoundShape shape = new MutableCompoundShape();
        final Vec3 offset = this.sectionVecScratch.get();
        for (final TerrainTile tile : parts) {
            offset.set((tile.tx << (TILE_SHIFT + 4)) - region.originX, tile.ty << (TILE_SHIFT + 4), (tile.tz << (TILE_SHIFT + 4)) - region.originZ);
            shape.addShape(offset, IDENTITY_QUAT, tile.layers[rl.layer].shape);
        }
        final TerrainTile[] assembled = parts.toArray(new TerrainTile[0]);
        if (rl.body == null) {
            final boolean special = rl.layer == TERRAIN_SPECIAL;
            final BodyCreationSettings bcs = new BodyCreationSettings(shape,
                    region.originX, 0.0, region.originZ, 0, 0, 0, 1, EMotionType.Static, special ? LAYER_STATIC_SPECIAL : LAYER_STATIC);
            if (this.rapierMode) {
                bcs.setFriction(FRICTION_TERRAIN_RAPIER);
            }
            // Every pair with terrain drops the ghost contacts at block seams.
            bcs.setEnhancedInternalEdgeRemoval(true);
            if (special) {
                // One manifold per block pair: each block applies its own friction,
                // restitution and callback in the listener.
                bcs.setUseManifoldReduction(false);
            }
            final Body body;
            try {
                body = this.bi.createBody(bcs);
            } catch (final IllegalStateException e) {
                this.regionsAwaitingBody.add(rl);
                this.warnCapacity("terrain region (" + region.rx + "," + region.rz + ") is waiting for a free body");
                return;
            }
            this.regionsAwaitingBody.remove(rl);
            body.setUserData(-1L);
            rl.shape = shape;
            rl.assembled = assembled;
            this.bi.addBody(body, EActivation.DontActivate);
            rl.body = body;
            rl.joltId = body.getId();
            this.terrainRegionsByBodyId.put(rl.joltId, rl);
        } else {
            rl.shape = shape;
            rl.assembled = assembled;
            this.bi.setShape(rl.joltId, shape, false, EActivation.DontActivate);
        }
    }

    /**
     * Region bodies that could not be created (body limit reached); retried.
     */
    private final ReferenceOpenHashSet<RegionLayer> regionsAwaitingBody = new ReferenceOpenHashSet<>();

    private void retryRegionBodies() {
        if (this.regionsAwaitingBody.isEmpty()) {
            return;
        }
        for (final RegionLayer rl : new ObjectArrayList<>(this.regionsAwaitingBody)) {
            if (this.terrainRegions.get(rl.region.key) != rl.region || rl.body != null) {
                this.regionsAwaitingBody.remove(rl);
                continue;
            }
            this.assembleRegionLayer(rl);
        }
    }

    private void destroyRegionBody(final RegionLayer rl) {
        if (rl.body != null) {
            this.terrainRegionsByBodyId.remove(rl.joltId);
            this.bi.removeBody(rl.joltId);
            this.bi.destroyBody(rl.joltId);
            rl.body = null;
            rl.joltId = 0;
            rl.shape = null;
            rl.assembled = new TerrainTile[0];
        }
    }

    /**
     * Test support: builds the sub-shape id of every terrain leaf and surface
     * mesh with Jolt's own id composition and checks that the contact decoding
     * finds the same leaf.
     *
     * @return the number of leaves and meshes decoded wrongly
     */
    int verifyTerrainDecoding() {
        int wrong = 0;
        for (final TerrainRegion region : this.terrainRegions.values()) {
            for (final RegionLayer rl : region.layers) {
                if (rl.shape == null) {
                    continue;
                }
                for (int ti = 0; ti < rl.assembled.length; ti++) {
                    final TileLayer tl = rl.assembled[ti].layers[rl.layer];
                    final SubShapeIdCreator tileId = rl.shape.getSubShapeIdFromIndex(ti, new SubShapeIdCreator());
                    for (int si = 0; si < tl.assembled.length; si++) {
                        final TerrainPart part = tl.assembled[si].parts[rl.layer];
                        final SubShapeIdCreator sectionId = tl.assembled.length == 1 ? tileId
                                : ((CompoundShape) tl.shape).getSubShapeIdFromIndex(si, tileId);
                        wrong += verifyPart(rl, part, sectionId);
                    }
                }
            }
        }
        return wrong;
    }

    private static int verifyPart(final RegionLayer rl, final TerrainPart part, final SubShapeIdCreator sectionId) {
        int wrong = 0;
        final int meshes = part.meshChildren.length;
        SubShapeIdCreator boxesId = sectionId;
        @Nullable ConstShape boxes = part.shape;
        if (meshes > 0) {
            if (part.partBits == 0) {
                return rl.childAt(sectionId.getId()) == part.meshChildren[0] ? 0 : 1;
            }
            final CompoundShape compound = (CompoundShape) part.shape;
            for (int m = 0; m < meshes; m++) {
                if (rl.childAt(compound.getSubShapeIdFromIndex(m, sectionId).getId()) != part.meshChildren[m]) {
                    wrong++;
                }
            }
            if (part.children.length == 0) {
                return wrong;
            }
            boxesId = compound.getSubShapeIdFromIndex(meshes, sectionId);
            boxes = compound.getSubShape(meshes).getShape();
        }
        for (int li = 0; li < part.children.length; li++) {
            final SubShapeIdCreator leafId = part.children.length == 1 ? boxesId
                    : ((CompoundShape) boxes).getSubShapeIdFromIndex(li, boxesId);
            if (rl.childAt(leafId.getId()) != part.children[li]) {
                wrong++;
            }
        }
        return wrong;
    }

    /**
     * Collider of the solid terrain block at a world position, or null (air,
     * fluid, unloaded). Read by the contact listener during the step.
     */
    @Nullable
    JoltVoxelColliderData terrainEntryAt(final int x, final int y, final int z) {
        final GlobalChunk chunk = this.globalChunks.get(ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4));
        return chunk == null ? null : this.solidEntry(chunk.data.get(x & 15, y & 15, z & 15));
    }

    /**
     * Collider of the solid block of a voxel body at a world position, or null.
     */
    @Nullable
    JoltVoxelColliderData bodyEntryAt(final SableBody body, final double wx, final double wy, final double wz) {
        final Vector3f r = this.rotateInto((float) (wx - body.snapComX), (float) (wy - body.snapComY), (float) (wz - body.snapComZ),
                -body.snapQx, -body.snapQy, -body.snapQz, body.snapQw);
        final int x = floor(body.centerOfMass.x + r.x), y = floor(body.centerOfMass.y + r.y), z = floor(body.centerOfMass.z + r.z);
        final long key = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
        final ChunkSectionData data = body.kind == SableBody.Kind.CONTRAPTION ? body.chunks.get(key) : this.allChunks.get(key);
        return data == null || !body.contains(x, y, z) && body.kind != SableBody.Kind.CONTRAPTION ? null : this.solidEntry(data.get(x & 15, y & 15, z & 15));
    }

    @Nullable
    private JoltVoxelColliderData solidEntry(final int packed) {
        final int colliderId = ChunkSectionData.colliderIdOf(packed);
        if (colliderId == 0) {
            return null;
        }
        final JoltVoxelColliderData entry = this.colliderRegistry.get(colliderId - 1);
        return entry == null || entry.isFluid || !entry.hasBoxes() ? null : entry;
    }

    private static final int NEAR_SPECIAL = 1;
    private static final int NEAR_BOUNCY = 2;

    /**
     * Which kinds of special terrain sections a body's bounding sphere
     * overlaps: blocks with a friction or restitution multiplier
     * ({@link #NEAR_SPECIAL}), blocks with restitution ({@link #NEAR_BOUNCY}).
     */
    private int nearSpecialTerrain(final SableBody sb) {
        final int any = (this.specialSections.isEmpty() ? 0 : NEAR_SPECIAL) | (this.bouncySections.isEmpty() ? 0 : NEAR_BOUNCY);
        if (any == 0) {
            return 0;
        }
        final double r = sb.boundRadius + 1.0;
        final int x0 = floor(sb.snapComX - r) >> 4, x1 = floor(sb.snapComX + r) >> 4;
        final int y0 = floor(sb.snapComY - r) >> 4, y1 = floor(sb.snapComY + r) >> 4;
        final int z0 = floor(sb.snapComZ - r) >> 4, z1 = floor(sb.snapComZ + r) >> 4;
        if ((long) (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1) > 512) {
            return any;
        }
        int near = 0;
        for (int x = x0; x <= x1; x++) {
            for (int y = y0; y <= y1; y++) {
                for (int z = z0; z <= z1; z++) {
                    final long key = ChunkSectionData.packSectionPos(x, y, z);
                    if (this.specialSections.contains(key)) {
                        near |= NEAR_SPECIAL;
                    }
                    if (this.bouncySections.contains(key)) {
                        near |= NEAR_BOUNCY;
                    }
                    if (near == any) {
                        return near;
                    }
                }
            }
        }
        return near;
    }

    /**
     * Test support: number of surface meshes and region bodies per terrain layer.
     */
    int[] terrainStats() {
        int bodiesPlain = 0, bodiesSpecial = 0, meshes = 0;
        for (final TerrainRegion region : this.terrainRegions.values()) {
            bodiesPlain += region.layers[TERRAIN_PLAIN].body != null ? 1 : 0;
            bodiesSpecial += region.layers[TERRAIN_SPECIAL].body != null ? 1 : 0;
        }
        for (final GlobalChunk chunk : this.globalChunks.values()) {
            meshes += chunk.parts[TERRAIN_PLAIN].meshChildren.length;
        }
        return new int[]{bodiesPlain, bodiesSpecial, meshes};
    }

    private long lastCapacityWarnNanos;

    private void warnCapacity(final String what) {
        final long now = System.nanoTime();
        if (now - this.lastCapacityWarnNanos > 30_000_000_000L) {
            this.lastCapacityWarnNanos = now;
            Sable.LOGGER.warn("[SableJolt] Physics body limit reached ({} bodies, max_bodies={}): {}. "
                            + "Raise max_bodies in config/sable_jolt_physics.properties.",
                    this.system.getNumBodies(), this.config.maxBodies, what);
        }
    }

    //endregion

    //region Kinematic contraptions

    /**
     * Creates a kinematic contraption. Contraptions mounted on a sub-level
     * become part of the sub-level's compound (Rapier attaches their collider
     * to the mount's rigid body): they push and support their mount and never
     * collide with it. World contraptions get their own kinematic body.
     */
    public void createKinematicContraption(final int mountId, final int id, final Vector3dc pos, final Quaterniondc rot) {
        if (this.disposed) {
            return;
        }
        final SableBody previous = this.bodies.get(id);
        if (previous != null) {
            this.destroyBody(previous);
        }
        final SableBody sb;
        if (mountId == -1) {
            sb = this.createBody(SableBody.Kind.CONTRAPTION, id, pos, rot, LAYER_MOVING);
        } else {
            sb = new SableBody(SableBody.Kind.CONTRAPTION, id);
            this.bodies.put(id, sb);
            final SableBody mount = this.bodies.get(mountId);
            if (mount != null && mount.kind == SableBody.Kind.SUB_LEVEL) {
                mount.mountedContraptions.add(sb);
                sb.embeddedInMount = true;
                mount.parentDirty = true;
                mount.wakeOnAssemble = true;
                this.markDirty(mount);
            }
        }
        sb.mountId = mountId;
        sb.relPos.set(pos);
        sb.relRot.set(rot);
        this.contraptionList.add(sb);
    }

    private void attachWaitingContraptions(final SableBody mount) {
        for (final SableBody c : this.contraptionList) {
            if (c.mountId == mount.runtimeId && !c.embeddedInMount) {
                mount.mountedContraptions.add(c);
                c.embeddedInMount = true;
                mount.parentDirty = true;
                mount.wakeOnAssemble = true;
                this.markDirty(mount);
            }
        }
    }

    private void detachFromMount(final SableBody contraption) {
        if (!contraption.embeddedInMount) {
            return;
        }
        contraption.embeddedInMount = false;
        final SableBody mount = this.bodies.get(contraption.mountId);
        if (mount != null && mount.mountedContraptions.remove(contraption)) {
            mount.parentDirty = true;
            mount.wakeOnAssemble = true;
            this.markDirty(mount);
        }
    }

    /**
     * Section keys of global chunks that contain at least one fluid block; buoyancy
     * is skipped entirely when empty.
     */
    private final LongOpenHashSet fluidSections = new LongOpenHashSet();

    public void removeKinematicContraption(final int id) {
        final SableBody sb = this.bodies.get(id);
        if (sb != null && sb.kind == SableBody.Kind.CONTRAPTION) {
            this.destroyBody(sb);
        }
    }

    /**
     * Uploads a kinematic contraption's pose: its position in the mount's block
     * space (world space when unmounted), orientation and velocities in its own
     * frame. All arguments are copied immediately; no allocations are performed.
     */
    public void setKinematicContraptionTransform(final int id, final Vector3dc centerOfMass, final Vector3dc pos,
                                                 final Quaterniondc rot, final Vector3dc linVel, final Vector3dc angVel) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.kind != SableBody.Kind.CONTRAPTION) {
            return;
        }
        if (!(Double.isFinite(pos.x()) && Double.isFinite(pos.y()) && Double.isFinite(pos.z())
                && Double.isFinite(rot.x()) && Double.isFinite(rot.y()) && Double.isFinite(rot.z()) && Double.isFinite(rot.w()))) {
            return;
        }
        if (centerOfMass != null && Double.isFinite(centerOfMass.x()) && Double.isFinite(centerOfMass.y()) && Double.isFinite(centerOfMass.z())
                && (centerOfMass.x() != sb.centerOfMass.x || centerOfMass.y() != sb.centerOfMass.y || centerOfMass.z() != sb.centerOfMass.z)) {
            sb.centerOfMass.set(centerOfMass);
            sb.parentDirty = true;
            this.markDirty(sb);
        }
        final boolean moved = !sb.relPos.equals(pos.x(), pos.y(), pos.z())
                || sb.relRot.x != rot.x() || sb.relRot.y != rot.y() || sb.relRot.z != rot.z() || sb.relRot.w != rot.w();
        sb.relPos.set(pos);
        sb.relRot.set(rot);
        sb.linVel.set(linVel);
        sb.angVel.set(angVel);
        if (moved) {
            this.markMountDirty(sb);
        }
    }

    public void addKinematicContraptionChunkSection(final int id, final int x, final int y, final int z, final int[] data) {
        final SableBody sb = this.bodies.get(id);
        if (sb == null || sb.kind != SableBody.Kind.CONTRAPTION) {
            return;
        }
        final ChunkSectionData section = new ChunkSectionData();
        System.arraycopy(data, 0, section.array(), 0, ChunkSectionData.BLOCKS);
        final long key = ChunkSectionData.packSectionPos(x, y, z);
        sb.chunks.put(key, section);
        sb.dirtySections.add(key);
        this.markDirty(sb);
    }

    //endregion

    //region Simulation

    /**
     * Runs once per game tick before the substeps: the Rapier buoyancy model
     * computes its forces here and applies them in every substep (Rapier's
     * forces persist across the tick's substeps).
     */
    public void prePhysicsTick() {
        if (this.disposed) {
            return;
        }
        if (this.rapierMode) {
            this.computeTickBuoyancy();
        }
    }

    private long lastUpdateErrorNanos;

    /**
     * Upper bound of Jolt's {@code ContactConstraint}, as far as this allocator is concerned: measured
     * with jolt-jni 6.1.1 at about 485 bytes. A smaller native struct only leaves the block roomier and a
     * larger one only brings back the malloc fallback, so this does not have to be exact.
     */
    private static final long CONTACT_CONSTRAINT_BYTES = 512L;
    /**
     * Everything else an update takes from the allocator: body pairs, islands, solver steps.
     */
    private static final long OTHER_TEMP_BYTES = 8L * 1024L * 1024L;

    /**
     * Size of the fixed temp block. Each update that has an awake body asks for one buffer big enough for
     * {@code max_contact_constraints}, however few contacts there are: 32 MB at the default. When that
     * does not fit in the block it comes from malloc, and on a server thread whose malloc arena has other
     * data in it glibc then builds and tears down a whole heap, with mmap and munmap, in every substep:
     * 55 microseconds per update, the larger part of what physics cost on a busy server (spark, 2026-11-06).
     * The block is only reserved address space until it gets used, and only the used part of it is touched.
     */
    private static int tempAllocatorBytes(final int maxContactConstraints) {
        return (int) Math.min(1L << 30, OTHER_TEMP_BYTES + maxContactConstraints * CONTACT_CONSTRAINT_BYTES);
    }

    /**
     * Squared speed of one block per substep ({@link #SPEED_SWEPT_BODY_CHILDREN}).
     */
    private float sweepSpeedSq = 1600.0f;
    /**
     * Length of the current substep (s).
     */
    volatile double lastTimeStep = 1.0 / 40.0;
    /**
     * Speed bound from which a body reports its contacts (see {@link #REPORT_SPEED}).
     */
    private float reportSpeed = REPORT_SPEED;

    /**
     * Steps the simulation by one substep.
     */
    public void step(final double timeStep) {
        if (this.disposed) {
            return;
        }
        final long stepStartNanos = System.nanoTime();
        this.sweepSpeedSq = (float) (1.0 / (timeStep * timeStep));
        this.lastTimeStep = timeStep;
        // A body hitting the world reports from 25 * dt m/s (see report).
        this.reportSpeed = (float) Math.min(REPORT_SPEED, 0.9 * 25.0 * timeStep);
        this.flushDirty();
        if (this.rapierMode) {
            this.applyTickBuoyancy();
        } else {
            this.computeBuoyancy();
        }
        this.applyJointDampers(timeStep);
        this.updateContraptionMotion((float) timeStep);
        // small scenes: a single-threaded job system avoids the wake/sync overhead
        // of the thread pool, which dominates when the solver has almost no work.
        // A single body with a huge compound is NOT a small scene: its narrow phase
        // dominates the step, so heavy bodies force the multithreaded job system.
        // The flag is maintained by refreshBodySnapshot (one tick stale at most).
        final boolean heavyBody = this.snapshotHeavyBody;
        final JobSystem jobs = this.bodyList.size() >= MULTITHREAD_BODY_THRESHOLD || heavyBody
                || this.snapshotAwakeLeaves >= PARALLEL_AWAKE_LEAVES
                ? this.jobSystem : this.singleThreadJobSystem;

        // Adaptive solver depth (Jolt mode): a huge compound body resting on the
        // ground forms a contact island with thousands of points; halving the
        // iteration depth there roughly halves the solver cost. Exception: while
        // a creative drag is active the full depth is kept.
        final boolean reducedSolver = !this.rapierMode && heavyBody && this.creativeDragBodies.isEmpty();
        if (reducedSolver != this.reducedSolverApplied) {
            final PhysicsSettings physics = this.system.getPhysicsSettings();
            physics.setNumVelocitySteps(reducedSolver ? Math.max(5, this.configuredVelocitySteps / 2) : this.configuredVelocitySteps);
            physics.setNumPositionSteps(reducedSolver ? Math.max(2, this.configuredPositionSteps / 2) : this.configuredPositionSteps);
            this.system.setPhysicsSettings(physics);
            this.reducedSolverApplied = reducedSolver;
        }

        final int errors = this.system.update((float) timeStep, 1, this.tempAllocator, jobs);
        if (errors != EPhysicsUpdateError.None) {
            this.warnUpdateErrors(errors);
        }
        // refreshBodySnapshot also applies the huge-body tunneling guard inline
        // (velocities are already in registers there) instead of a separate pass.
        this.refreshBodySnapshot();
        if (JoltDebugLogging.STAFF) {
            this.staffWorldDump();
        }
        // Failsafe (Jolt mode): a huge compound landing with deep penetration can
        // collapse the solver into an ever-growing contact island. If the step
        // overran its budget, put oversized slow bodies to sleep so the world can
        // recover instead of freezing every subsequent tick.
        final long updateNanos = System.nanoTime() - stepStartNanos;
        if (!this.rapierMode && updateNanos > 250_000_000L) {
            for (final SableBody sb : this.bodyList) {
                if (sb.leafCount >= HEAVY_BODY_CHILDREN && sb.body.isActive()
                        && !this.creativeDragBodies.contains(sb.joltId)) {
                    if (sb.snapVx * sb.snapVx + sb.snapVy * sb.snapVy + sb.snapVz * sb.snapVz < 0.5f) {
                        this.bi.deactivateBody(sb.joltId);
                    }
                }
            }
        }
        this.pushDraggedOutOfPenetration();
    }

    private void warnUpdateErrors(final int errors) {
        final long now = System.nanoTime();
        if (now - this.lastUpdateErrorNanos < 30_000_000_000L) {
            return;
        }
        this.lastUpdateErrorNanos = now;
        final StringBuilder what = new StringBuilder();
        if ((errors & EPhysicsUpdateError.ManifoldCacheFull) != 0 || (errors & EPhysicsUpdateError.ContactConstraintsFull) != 0) {
            what.append(" max_contact_constraints=").append(this.config.maxContactConstraints);
        }
        if ((errors & EPhysicsUpdateError.BodyPairCacheFull) != 0) {
            what.append(" max_body_pairs=").append(this.config.maxBodyPairs);
        }
        Sable.LOGGER.warn("[SableJolt] Physics capacity exceeded (error flags {}), some contacts were skipped this step;"
                + " raise{} in config/sable_jolt_physics.properties", errors, what);
    }

    /**
     * Allowed embed depth (m) of a dragged body before the push-out engages;
     * contact slop below this is normal solver behavior.
     */
    private static final double DRAG_PUSH_SLOP = 0.12;
    /**
     * Max push-out distance (m) per step for a dragged body.
     */
    private static final double DRAG_PUSH_MAX = 1.5;

    /**
     * Nudges creatively dragged bodies out of deep penetration. Rotating or
     * dragging a huge body can sweep its parts meters into the terrain within
     * a single tick — faster than the solver's per-step recovery — which would
     * otherwise ratchet the body underground. The snapshot fields are updated
     * together with the body so later reads stay consistent.
     */
    private void pushDraggedOutOfPenetration() {
        if (this.creativeDragBodies.isEmpty()) {
            return;
        }
        for (final int joltId : this.creativeDragBodies) {
            final SableBody sb = this.bodyByJoltId(joltId);
            if (sb == null || sb.dragPenDepth <= DRAG_PUSH_SLOP) {
                continue;
            }
            final double push = Math.min(sb.dragPenDepth - DRAG_PUSH_SLOP, DRAG_PUSH_MAX);
            sb.snapX += sb.dragPenNx * push;
            sb.snapY += sb.dragPenNy * push;
            sb.snapZ += sb.dragPenNz * push;
            sb.snapComX = sb.snapX;
            sb.snapComY = sb.snapY;
            sb.snapComZ = sb.snapZ;
            this.bi.setPositionAndRotation(joltId, sb.snapX, sb.snapY, sb.snapZ,
                    sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw, EActivation.DontActivate);
        }
    }

    /**
     * Whether any tracked body currently holds a huge compound; maintained by
     * {@link #refreshBodySnapshot} so the pre-step job-system choice needs no
     * dedicated scan (one tick stale at most — a pure performance heuristic).
     */
    private boolean snapshotHeavyBody;
    private int snapshotAwakeLeaves;

    /**
     * Native list of the awake bodies, filled by Jolt every step.
     */
    private final BodyIdVector activeIds = new BodyIdVector();
    /**
     * Bodies read by the previous refresh because they were awake: the ones
     * that fell asleep during this step are read one last time.
     */
    private ObjectArrayList<SableBody> lastAwake = new ObjectArrayList<>();
    private ObjectArrayList<SableBody> currentAwake = new ObjectArrayList<>();
    /**
     * Bodies whose snapshot must be re-read (created, or velocity changed by an impulse).
     */
    private final ObjectArrayList<SableBody> staleSnapshots = new ObjectArrayList<>();
    private final ObjectArrayList<SableBody> extraSnapshots = new ObjectArrayList<>();
    private java.nio.LongBuffer snapshotUser;

    /**
     * Marks a body's snapshot stale: reads fall back to the native body until
     * the next refresh re-reads it.
     */
    private void invalidateSnapshot(final SableBody sb) {
        sb.snapValid = false;
        if (!sb.snapQueued) {
            sb.snapQueued = true;
            this.staleSnapshots.add(sb);
        }
    }

    /**
     * Reads the pose and velocity of the bodies that can have moved this step:
     * the awake ones (Jolt's active list, read in place), the ones that fell
     * asleep during the step and stale ones. Sleeping bodies keep their
     * snapshot, so a world of resting sub-levels costs nothing here; the old
     * read of every body cost ~12 ms per substep at 10k bodies. Reads use the
     * lock-free body interface: nothing else touches bodies between steps.
     * Cached values are served by {@link #getPose}, the velocity getters and
     * the buoyancy passes.
     */
    private void refreshBodySnapshot() {
        final long gen = ++this.snapshotGen;
        final ObjectArrayList<SableBody> awake = this.currentAwake;
        awake.clear();
        boolean heavyBody = false;

        this.system.getActiveBodies(com.github.stephengold.joltjni.enumerate.EBodyType.RigidBody, this.activeIds);
        final int activeCount = this.activeIds.size();
        if (activeCount > 0) {
            final BodyIdArray ids = this.activeIds.data();
            this.ensureSnapshotBuffers(activeCount);
            this.batchNoLock.getUserData(ids, this.snapshotUser);
            this.readSnapshotBatch(ids, false);
            for (int idx = 0; idx < activeCount; idx++) {
                final long user = this.snapshotUser.get(idx);
                if (user < 0L || user > Integer.MAX_VALUE) {
                    continue; // rope points
                }
                final SableBody sb = this.bodies.get((int) user);
                if (sb == null || sb.body == null) {
                    continue;
                }
                sb.snapActive = true;
                heavyBody |= this.applySnapshot(sb, idx, gen);
                awake.add(sb);
            }
        }

        // Bodies that fell asleep this step, and stale snapshots.
        final ObjectArrayList<SableBody> extras = this.extraSnapshots;
        extras.clear();
        for (final SableBody sb : this.lastAwake) {
            if (sb.snapGen != gen && sb.body != null && this.bodies.get(sb.runtimeId) == sb) {
                sb.snapGen = gen;
                extras.add(sb);
            }
        }
        for (final SableBody sb : this.staleSnapshots) {
            sb.snapQueued = false;
            if (sb.snapGen != gen && sb.body != null && this.bodies.get(sb.runtimeId) == sb) {
                sb.snapGen = gen;
                extras.add(sb);
            }
        }
        this.staleSnapshots.clear();
        if (!extras.isEmpty()) {
            final int n = extras.size();
            if (this.snapshotIds == null || this.snapshotIds.length() < n) {
                final int capacity = Math.max(64, n + (n >> 1));
                this.snapshotIds = new BodyIdArray(capacity);
                // Every slot always holds a valid id: unused ones the static ground.
                for (int i = 0; i < capacity; i++) {
                    this.snapshotIds.set(i, this.groundBodyId);
                }
                this.snapshotIdsFilled = 0;
            }
            this.ensureSnapshotBuffers(this.snapshotIds.length());
            for (int i = 0; i < n; i++) {
                this.snapshotIds.set(i, extras.get(i).joltId);
            }
            for (int i = n; i < this.snapshotIdsFilled; i++) {
                this.snapshotIds.set(i, this.groundBodyId);
            }
            this.snapshotIdsFilled = n;
            this.readSnapshotBatch(this.snapshotIds, true);
            for (int i = 0; i < n; i++) {
                final SableBody sb = extras.get(i);
                sb.snapActive = this.snapshotActive.get(i) != 0;
                heavyBody |= this.applySnapshot(sb, i, gen);
                if (sb.snapActive) {
                    awake.add(sb);
                }
            }
        }

        // Contact routing for the next step (only bodies that changed this step).
        int awakeLeaves = 0;
        for (final SableBody sb : awake) {
            this.classifyLayer(sb);
            awakeLeaves += sb.leafCount;
        }
        this.snapshotAwakeLeaves = awakeLeaves;
        for (final SableBody sb : extras) {
            if (!sb.snapActive) {
                this.classifyLayer(sb);
            }
        }

        this.currentAwake = this.lastAwake;
        this.lastAwake = awake;
        this.snapshotHeavyBody = heavyBody;
    }

    /**
     * Leading slots of {@link #snapshotIds} holding body ids (the rest is ground padding).
     */
    private int snapshotIdsFilled;

    private void setBodyLayer(final SableBody sb, final int layer) {
        if (sb.body == null || sb.layer == layer) {
            return;
        }
        this.bi.setObjectLayer(sb.joltId, layer);
        sb.layer = layer;
    }

    /**
     * Report layer for bodies whose contacts need Java (special blocks of their
     * own or of the terrain they overlap, mounted contraptions, creative drag)
     * or that move fast enough for an impact report (linear speed plus spin
     * times the body radius). Contacts with terrain blocks that have a contact
     * callback reach Java through that terrain body's own layer.
     */
    private void classifyLayer(final SableBody sb) {
        if (sb.body == null || sb.layer == LAYER_ROPE) {
            return;
        }
        final int near = sb.snapActive ? this.nearSpecialTerrain(sb) : 0;
        sb.nearSpecial = (near & NEAR_SPECIAL) != 0;
        sb.nearBouncy = (near & NEAR_BOUNCY) != 0;
        final int desired;
        if (sb.layerSpecial || this.creativeDragBodies.contains(sb.joltId)) {
            desired = LAYER_MOVING_REPORT;
        } else if (!sb.snapActive) {
            desired = LAYER_MOVING;
        } else if (sb.nearSpecial) {
            desired = LAYER_MOVING_REPORT;
        } else {
            final float v = (float) Math.sqrt(sb.snapVx * sb.snapVx + sb.snapVy * sb.snapVy + sb.snapVz * sb.snapVz);
            final float w = (float) Math.sqrt(sb.snapAx * sb.snapAx + sb.snapAy * sb.snapAy + sb.snapAz * sb.snapAz);
            final float bound = v + w * sb.boundRadius;
            // Hysteresis: a layer change re-files the body in the broad phase,
            // so bodies hovering around the threshold must not switch every step.
            if (sb.layer == LAYER_MOVING_REPORT) {
                desired = bound >= this.reportSpeed * 0.6f ? LAYER_MOVING_REPORT : LAYER_MOVING;
            } else {
                desired = bound >= this.reportSpeed ? LAYER_MOVING_REPORT : LAYER_MOVING;
            }
        }
        this.setBodyLayer(sb, desired);
    }

    private void readSnapshotBatch(final ConstBodyIdArray ids, final boolean withActive) {
        // Center of mass positions skip the per-body shape lookup of getPositions;
        // for these bodies both are the same point.
        this.batchNoLock.getCenterOfMassPositions(ids, this.snapshotPos);
        this.batchNoLock.getRotations(ids, this.snapshotRot);
        this.batchNoLock.getLinearVelocities(ids, this.snapshotLin);
        this.batchNoLock.getAngularVelocities(ids, this.snapshotAng);
        if (withActive) {
            this.batchNoLock.areActive(ids, this.snapshotActive);
        }
    }

    /**
     * Stores one batch-read entry into a body's snapshot (with the poison
     * watchdog and the Jolt-mode tunneling guard). Returns whether the body
     * holds a huge compound.
     */
    private boolean applySnapshot(final SableBody sb, final int idx, final long gen) {
        sb.snapGen = gen;
        sb.snapValid = true;
        final double px = this.snapshotPos.get(idx * 3);
        final double py = this.snapshotPos.get(idx * 3 + 1);
        final double pz = this.snapshotPos.get(idx * 3 + 2);
        // Every compound here keeps its center of mass at the shape origin
        // (bodies are positioned at their center of mass), so the body
        // position is the center of mass position.
        final double cx = px;
        final double cy = py;
        final double cz = pz;
        final float qx = this.snapshotRot.get(idx * 4);
        final float qy = this.snapshotRot.get(idx * 4 + 1);
        final float qz = this.snapshotRot.get(idx * 4 + 2);
        final float qw = this.snapshotRot.get(idx * 4 + 3);
        float vx = this.snapshotLin.get(idx * 3);
        float vy = this.snapshotLin.get(idx * 3 + 1);
        float vz = this.snapshotLin.get(idx * 3 + 2);
        final float ax = this.snapshotAng.get(idx * 3);
        final float ay = this.snapshotAng.get(idx * 3 + 1);
        final float az = this.snapshotAng.get(idx * 3 + 2);
        final boolean heavy = sb.leafCount >= HEAVY_BODY_CHILDREN;
        // Poison watchdog: non-finite state (NaN/Inf position, rotation or
        // velocity) makes the NEXT native update read garbage and die with
        // EXCEPTION_ACCESS_VIOLATION inside joltjni.dll. Quarantine the body
        // instead: restore the last finite snapshot, zero the velocities
        // and put it to sleep so the solver never sees the poison. The
        // cached fields keep serving the last good values downstream.
        if (!(Double.isFinite(px) && Double.isFinite(py) && Double.isFinite(pz)
                && Double.isFinite(cx) && Double.isFinite(cy) && Double.isFinite(cz)
                && Float.isFinite(qx) && Float.isFinite(qy) && Float.isFinite(qz) && Float.isFinite(qw)
                && Float.isFinite(vx) && Float.isFinite(vy) && Float.isFinite(vz)
                && Float.isFinite(ax) && Float.isFinite(ay) && Float.isFinite(az))) {
            this.quarantinePoisonedBody(sb);
            sb.snapVx = sb.snapVy = sb.snapVz = 0.0f;
            sb.snapAx = sb.snapAy = sb.snapAz = 0.0f;
            sb.snapActive = false;
            return heavy;
        }
        sb.snapX = px;
        sb.snapY = py;
        sb.snapZ = pz;
        sb.snapComX = cx;
        sb.snapComY = cy;
        sb.snapComZ = cz;
        sb.snapQx = qx;
        sb.snapQy = qy;
        sb.snapQz = qz;
        sb.snapQw = qw;
        // Tunneling guard for huge bodies (no CCD) in Jolt mode: cap speed
        // below one block per tick. Rapier-compatible mode keeps every speed.
        if (heavy && !this.rapierMode && sb.kind == SableBody.Kind.SUB_LEVEL) {
            final float speedSq = vx * vx + vy * vy + vz * vz;
            if (speedSq > 144.0f) {
                final float scale = 12.0f / (float) Math.sqrt(speedSq);
                vx *= scale;
                vy *= scale;
                vz *= scale;
                this.bi.setLinearVelocity(sb.joltId, vx, vy, vz);
            }
        }
        this.updateMotionQuality(sb, heavy, vx, vy, vz);
        sb.snapVx = vx;
        sb.snapVy = vy;
        sb.snapVz = vz;
        sb.snapAx = ax;
        sb.snapAy = ay;
        sb.snapAz = az;
        return heavy && sb.snapActive;
    }

    /**
     * Speed along gravity (m/s) from which a body sweeps (LinearCast). Slower
     * vertical motion stays within the speculative contact distance per substep.
     */
    private static final float SWEEP_VERTICAL_SPEED = 1.0f;

    /**
     * Chooses between swept (LinearCast) and discrete motion for the next step.
     * A body only sweeps while it moves along gravity: falling, landing or
     * bouncing. A sweep casts the body along its whole motion, and a body that
     * slides on the ground rests a little inside it, so the cast stops it at the
     * side of the next block of every seam and launches it (Rapier has no
     * working CCD against voxels at all: its voxel shape does not support
     * casts). Bodies sliding or flying level move discretely, as in Rapier.
     * Large bodies additionally only sweep while faster than one block per
     * substep (casting a whole compound is expensive), huge ones never.
     * Bodies that can bounce (blocks with restitution of their own or in the
     * terrain around them) never sweep either: a swept body's impact is
     * resolved at the single point its cast hit, so a flat landing on a
     * bouncy block would spin the body; the discrete contacts bounce it from
     * its whole contact face.
     */
    private void updateMotionQuality(final SableBody sb, final boolean heavy, final float vx, final float vy, final float vz) {
        if (sb.kind != SableBody.Kind.SUB_LEVEL && sb.kind != SableBody.Kind.BOX) {
            return;
        }
        boolean sweep = false;
        if (!heavy && !sb.bouncy && !sb.nearBouncy) {
            final double along = this.hasGravity
                    ? Math.abs(vx * this.gravityDirX + vy * this.gravityDirY + vz * this.gravityDirZ)
                    : Double.MAX_VALUE;
            sweep = along >= SWEEP_VERTICAL_SPEED
                    && (sb.leafCount < SPEED_SWEPT_BODY_CHILDREN || vx * vx + vy * vy + vz * vz > this.sweepSpeedSq);
        }
        if (sweep != sb.sweeping) {
            sb.sweeping = sweep;
            this.bi.setMotionQuality(sb.joltId, sweep ? EMotionQuality.LinearCast : EMotionQuality.Discrete);
        }
    }

    private long lastPoisonLogNanos;

    /**
     * Quarantines a body whose native state read back non-finite: restores the
     * last finite pose (always valid: initialized at creation and teleport),
     * zeroes velocities and deactivates it so the next {@code system.update}
     * never ingests the poison. Rate-limited logging.
     */
    private void quarantinePoisonedBody(final SableBody sb) {
        final long now = System.nanoTime();
        if (now - this.lastPoisonLogNanos > 1_000_000_000L) {
            this.lastPoisonLogNanos = now;
            Sable.LOGGER.error("[SableJolt] body {} (jolt {}) read back non-finite state; quarantined at last good pose",
                    sb.runtimeId, sb.joltId);
        }
        try {
            this.bi.setPositionAndRotation(sb.joltId, sb.snapX, sb.snapY, sb.snapZ,
                    sb.snapQx, sb.snapQy, sb.snapQz, sb.snapQw, EActivation.DontActivate);
            this.bi.setLinearAndAngularVelocity(sb.joltId, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            this.bi.deactivateBody(sb.joltId);
        } catch (final Throwable ignored) {
            // best effort: the body is already unusable
        }
    }

    private void ensureSnapshotBuffers(final int n) {
        if (this.snapshotPos != null && this.snapshotPos.capacity() >= n * 3) {
            return;
        }
        final int cap = Math.max(64, n);
        this.snapshotPos = ByteBuffer.allocateDirect(cap * 3 * 8).order(ByteOrder.nativeOrder()).asDoubleBuffer();
        this.snapshotRot = ByteBuffer.allocateDirect(cap * 4 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotLin = ByteBuffer.allocateDirect(cap * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotAng = ByteBuffer.allocateDirect(cap * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        this.snapshotActive = ByteBuffer.allocateDirect(cap).order(ByteOrder.nativeOrder());
        this.snapshotUser = ByteBuffer.allocateDirect(cap * 8).order(ByteOrder.nativeOrder()).asLongBuffer();
    }

    /**
     * Splits per-body work into slices for the worker pool. The calling thread
     * processes the first slice and the method returns after all slices finish.
     */
    private void runParallel(final List<SableBody> bodies, final java.util.function.Consumer<SableBody> action) {
        runParallelIndices(bodies.size(), i -> action.accept(bodies.get(i)));
    }

    /**
     * Runs {@code action} for every index in [0, n) on the shared worker pool;
     * the calling thread takes the first slice. Returns after all finished.
     */
    private static void runParallelIndices(final int n, final IntConsumer action) {
        final ExecutorService pool = sharedWorkers();
        final int workers = Math.min(sharedWorkerCount + 1, n);
        if (workers <= 1) {
            for (int i = 0; i < n; i++) {
                action.accept(i);
            }
            return;
        }

        // Slices are collected first so the latch matches the number of tasks
        // actually submitted; otherwise an empty trailing slice would leave the
        // caller waiting on counts that never happen (permanent server stall).
        final int sliceSize = (n + workers - 1) / workers;
        final IntArrayList starts = new IntArrayList(workers);
        for (int w = 1; w < workers; w++) {
            final int from = w * sliceSize;
            if (from >= n) {
                break;
            }
            starts.add(from);
        }

        final CountDownLatch latch = new CountDownLatch(starts.size());
        for (int s = 0; s < starts.size(); s++) {
            final int from = starts.getInt(s);
            final int to = Math.min(n, from + sliceSize);
            pool.execute(() -> {
                try {
                    for (int i = from; i < to; i++) {
                        action.accept(i);
                    }
                } catch (final Throwable t) {
                    Sable.LOGGER.error("[SableJolt] parallel task failed", t);
                } finally {
                    latch.countDown();
                }
            });
        }
        final int headEnd = Math.min(sliceSize, n);
        for (int i = 0; i < headEnd; i++) {
            action.accept(i);
        }
        boolean interrupted = false;
        while (true) {
            try {
                latch.await();
                break;
            } catch (final InterruptedException e) {
                // Never return before the workers are done: they still touch
                // shared state.
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Whether the reduced solver iteration depth is currently applied (heavy-body
     * scenes only; restored when the heavy body goes away).
     */
    private boolean reducedSolverApplied;

    private long lastWorldDump;

    private void staffWorldDump() {
        final long now = System.currentTimeMillis();
        if (now - this.lastWorldDump < 2000) {
            return;
        }
        this.lastWorldDump = now;

        final StringBuilder sb = new StringBuilder("[SableJolt:staff] world: globalChunks=").append(this.globalChunks.size())
                .append(" fluidSections=").append(this.fluidSections.size())
                .append(" allChunks=").append(this.allChunks.size())
                .append(" bodies=").append(this.bodies.size());
        for (final SableBody body : this.bodyList) {
            sb.append(" | #").append(body.runtimeId).append("/").append(body.kind)
                    .append(" pos=(").append((float) body.snapComX).append(",").append((float) body.snapComY).append(",").append((float) body.snapComZ).append(")")
                    .append(" leaves=").append(body.leafCount)
                    .append(" bounds=(").append(body.minX).append(",").append(body.minY).append(",").append(body.minZ)
                    .append(")..(").append(body.maxX).append(",").append(body.maxY).append(",").append(body.maxZ).append(")")
                    .append(" com=(").append((float) body.centerOfMass.x).append(",").append((float) body.centerOfMass.y).append(",").append((float) body.centerOfMass.z).append(")")
                    .append(" active=").append(body.body.isActive());
        }
        Sable.LOGGER.info(sb.toString());
    }

    /**
     * Releases everything. Safe to call more than once; afterwards every
     * operation is a no-op (handles held by other mods may still call in).
     */
    public void dispose() {
        if (this.disposed) {
            return;
        }
        this.disposed = true;
        try {
            this.system.setContactListener(null);
            this.system.removeAllConstraints();
            this.system.destroyAllBodies();
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] failed to clear the physics world", t);
        }
        this.bodies.clear();
        this.bodiesByJoltId.clear();
        this.bodyList.clear();
        this.contraptionList.clear();
        this.pendingCreation.clear();
        this.globalChunks.clear();
        this.terrainRegions.clear();
        this.terrainRegionsByBodyId.clear();
        this.terrainTiles.clear();
        this.dirtyTiles.clear();
        this.specialSections.clear();
        this.bouncySections.clear();
        this.meshedSections.clear();
        this.regionsAwaitingBody.clear();
        this.allChunks.clear();
        this.sectionOwners.clear();
        this.joints.clear();
        this.damperJoints.clear();
        this.ropes.clear();
        synchronized (this.dirtyBodies) {
            this.dirtyBodies.clear();
        }
        this.dirtyGlobalChunks.clear();
        this.fluidSections.clear();
        this.angularFollowPrev.clear();
        this.angularCachedOmega.clear();
        this.creativeDragBodies.clear();
        this.creativeRestorePending.clear();
        this.ropePointBodies.clear();
        this.callbackResults = new Long2ObjectOpenHashMap<>();
        synchronized (this.pendingCallbacksLock) {
            this.pendingCallbacks.clear();
        }
        synchronized (this.reportedLock) {
            this.reportedCollisions.clear();
        }
        // The native listener holds a global reference to its Java object; it
        // must be freed explicitly or the whole scene stays reachable forever.
        try {
            this.system.close();
            this.contactListener.close();
            this.tempAllocator.close();
            this.singleThreadJobSystem.close();
        } catch (final Throwable t) {
            Sable.LOGGER.error("[SableJolt] failed to release native physics objects", t);
        }
    }

    public boolean isDisposed() {
        return this.disposed;
    }

    public int nextRuntimeId() {
        return this.nextRuntimeId.getAndIncrement();
    }

    //endregion

    //region Buoyancy

    /**
     * Fluid lookup with a one-section cache (per call site).
     */
    private final class FluidProbe {
        private long cachedKey = Long.MIN_VALUE;
        @Nullable
        private ChunkSectionData cached;

        /**
         * Returns the fluid level of a world block: 0 = none, 1..8 = level,
         * 9 = full (source / level not uploaded).
         */
        int fluidAt(final int x, final int y, final int z) {
            final long key = ChunkSectionData.packSectionPos(x >> 4, y >> 4, z >> 4);
            if (key != this.cachedKey) {
                this.cachedKey = key;
                if (!JoltPhysicsScene.this.fluidSections.contains(key)) {
                    this.cached = null;
                } else {
                    final GlobalChunk chunk = JoltPhysicsScene.this.globalChunks.get(key);
                    this.cached = chunk == null ? null : chunk.data;
                }
            }
            final ChunkSectionData data = this.cached;
            if (data == null) {
                return 0;
            }
            final int packed = data.get(x & 15, y & 15, z & 15);
            final int colliderId = ChunkSectionData.colliderIdOf(packed);
            if (colliderId == 0) {
                return 0;
            }
            final JoltVoxelColliderData entry = JoltPhysicsScene.this.colliderRegistry.get(colliderId - 1);
            if (entry == null || !entry.isFluid) {
                return 0;
            }
            final int level = ChunkSectionData.fluidLevelOf(packed);
            return level > 0 ? level : 9;
        }

        /**
         * Height of the fluid surface inside a world block (0 = no fluid): a
         * fluid block under another fluid block is full, otherwise its level
         * sets the surface (a source block is 8/9 high, as in Minecraft).
         */
        double fluidHeight(final int x, final int y, final int z) {
            final int level = this.fluidAt(x, y, z);
            if (level == 0) {
                return 0.0;
            }
            if (this.fluidAt(x, y + 1, z) != 0) {
                return 1.0;
            }
            return level >= 9 ? 1.0 : level / 9.0;
        }
    }

    private boolean isAwake(final SableBody sb) {
        return sb.snapValid ? sb.snapActive : sb.body.isActive();
    }

    private ObjectArrayList<SableBody> collectBuoyancyCandidates() {
        final ObjectArrayList<SableBody> active = new ObjectArrayList<>();
        for (final SableBody sb : this.bodyList) {
            if (sb.kind == SableBody.Kind.SUB_LEVEL && sb.buoyancyCount > 0
                    && !this.creativeDragBodies.contains(sb.joltId) && this.isAwake(sb)) {
                active.add(sb);
            }
        }
        return active;
    }

    /**
     * Rapier buoyancy (buoyancy.rs), once per game tick: every solid block of a
     * sub-level is an axis-aligned cube (eight quarter-size samples for small
     * bodies) whose overlap with world fluid blocks yields a constant upward
     * force (10.5 per submerged volume, scaled by the block's volume property)
     * and a drag of -1.7 * v * volume, both at the sample point.
     */
    private void computeTickBuoyancy() {
        for (final SableBody sb : this.bodyList) {
            sb.hasTickBuoyancy = false;
        }
        if (this.fluidSections.isEmpty()) {
            return;
        }
        final ObjectArrayList<SableBody> active = this.collectBuoyancyCandidates();
        if (active.isEmpty()) {
            return;
        }
        if (active.size() >= 12) {
            this.runParallel(active, this::rapierBuoyancyForBody);
        } else {
            for (final SableBody sb : active) {
                this.rapierBuoyancyForBody(sb);
            }
        }
    }

    private void applyTickBuoyancy() {
        for (final SableBody sb : this.bodyList) {
            // Forces on a sleeping body would accumulate until it wakes up.
            if (!sb.hasTickBuoyancy || sb.body == null || this.creativeDragBodies.contains(sb.joltId) || !this.isAwake(sb)) {
                continue;
            }
            sb.body.addForce((float) sb.tickFx, (float) sb.tickFy, (float) sb.tickFz);
            sb.body.addTorque((float) sb.tickTx, (float) sb.tickTy, (float) sb.tickTz);
        }
    }

    private void rapierBuoyancyForBody(final SableBody sb) {
        final boolean fresh = sb.snapValid;
        final double comX, comY, comZ;
        final float qx, qy, qz, qw;
        final double lvx, lvy, lvz, avx, avy, avz;
        if (fresh) {
            comX = sb.snapComX;
            comY = sb.snapComY;
            comZ = sb.snapComZ;
            qx = sb.snapQx;
            qy = sb.snapQy;
            qz = sb.snapQz;
            qw = sb.snapQw;
            lvx = sb.snapVx;
            lvy = sb.snapVy;
            lvz = sb.snapVz;
            avx = sb.snapAx;
            avy = sb.snapAy;
            avz = sb.snapAz;
        } else {
            final Body body = sb.body;
            final RVec3 comPos = body.getCenterOfMassPosition();
            comX = comPos.xx();
            comY = comPos.yy();
            comZ = comPos.zz();
            final Quat rot = body.getRotation();
            qx = rot.getX();
            qy = rot.getY();
            qz = rot.getZ();
            qw = rot.getW();
            final Vec3 lin = body.getLinearVelocity();
            final Vec3 ang = body.getAngularVelocity();
            lvx = lin.getX();
            lvy = lin.getY();
            lvz = lin.getZ();
            avx = ang.getX();
            avy = ang.getY();
            avz = ang.getZ();
        }

        // "complex" (small) bodies use eight quarter-size samples per block.
        final boolean complex = (sb.maxX - sb.minX) + (sb.maxY - sb.minY) + (sb.maxZ - sb.minZ) < 10;
        final double half = complex ? 0.25 : 0.5;
        final int samples = complex ? 8 : 1;

        final FluidProbe probe = new FluidProbe();
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double tx = 0.0, ty = 0.0, tz = 0.0;
        boolean any = false;

        final double baseX = sb.centerOfMass.x;
        final double baseY = sb.centerOfMass.y;
        final double baseZ = sb.centerOfMass.z;
        final int[] blocks = sb.buoyancyBlocks;
        final float[] volumes = sb.buoyancyVolumes;
        final int count = sb.buoyancyCount;
        for (int b = 0; b < count; b++) {
            final double cx = blocks[b * 3] + 0.5 - baseX;
            final double cy = blocks[b * 3 + 1] + 0.5 - baseY;
            final double cz = blocks[b * 3 + 2] + 0.5 - baseZ;
            final double floatStrength = volumes[b];
            for (int s = 0; s < samples; s++) {
                double lx = cx, ly = cy, lz = cz;
                if (complex) {
                    lx += ((s & 1) * 2 - 1) * 0.25;
                    ly += (((s >> 1) & 1) * 2 - 1) * 0.25;
                    lz += (((s >> 2) & 1) * 2 - 1) * 0.25;
                }
                // world sample point = com + R * local
                final double tX = 2.0 * (qy * lz - qz * ly);
                final double tY = 2.0 * (qz * lx - qx * lz);
                final double tZ = 2.0 * (qx * ly - qy * lx);
                final double rx = lx + qw * tX + (qy * tZ - qz * tY);
                final double ry = ly + qw * tY + (qz * tX - qx * tZ);
                final double rz = lz + qw * tZ + (qx * tY - qy * tX);
                final double px = comX + rx;
                final double py = comY + ry;
                final double pz = comZ + rz;

                final int x0 = floor(px - half), x1 = floor(px + half);
                final int y0 = floor(py - half), y1 = floor(py + half);
                final int z0 = floor(pz - half), z1 = floor(pz + half);
                double submerged = 0.0;
                for (int wx = x0; wx <= x1; wx++) {
                    final double ox = Math.min(px + half, wx + 1.0) - Math.max(px - half, wx);
                    if (ox <= 0.0) {
                        continue;
                    }
                    for (int wy = y0; wy <= y1; wy++) {
                        final double oy = Math.min(py + half, wy + 1.0) - Math.max(py - half, wy);
                        if (oy <= 0.0) {
                            continue;
                        }
                        for (int wz = z0; wz <= z1; wz++) {
                            final double oz = Math.min(pz + half, wz + 1.0) - Math.max(pz - half, wz);
                            if (oz <= 0.0) {
                                continue;
                            }
                            if (probe.fluidAt(wx, wy, wz) != 0) {
                                submerged += ox * oy * oz;
                            }
                        }
                    }
                }
                if (submerged <= 0.0) {
                    continue;
                }
                any = true;
                // velocity at the sample point
                final double vx = lvx + avy * rz - avz * ry;
                final double vy = lvy + avz * rx - avx * rz;
                final double vz = lvz + avx * ry - avy * rx;
                final double sfx = -vx * FLUID_DRAG * submerged;
                final double sfy = -vy * FLUID_DRAG * submerged + RAPIER_FLOAT_FORCE * submerged * floatStrength;
                final double sfz = -vz * FLUID_DRAG * submerged;
                fx += sfx;
                fy += sfy;
                fz += sfz;
                tx += ry * sfz - rz * sfy;
                ty += rz * sfx - rx * sfz;
                tz += rx * sfy - ry * sfx;
            }
        }

        if (any && Double.isFinite(fx) && Double.isFinite(fy) && Double.isFinite(fz)
                && Double.isFinite(tx) && Double.isFinite(ty) && Double.isFinite(tz)) {
            sb.tickFx = fx;
            sb.tickFy = fy;
            sb.tickFz = fz;
            sb.tickTx = tx;
            sb.tickTy = ty;
            sb.tickTz = tz;
            sb.hasTickBuoyancy = true;
        }
    }

    private void computeBuoyancy() {
        if (this.fluidSections.isEmpty()) {
            return;
        }
        final ObjectArrayList<SableBody> active = this.collectBuoyancyCandidates();
        if (active.isEmpty()) {
            return;
        }

        if (active.size() >= 12) {
            this.runParallel(active, this::buoyancyForBody);
        } else {
            for (final SableBody sb : active) {
                this.buoyancyForBody(sb);
            }
        }
    }

    /**
     * Jolt buoyancy model: per block, the submerged height of the rotated cube
     * against the fluid surface (fluid levels respected), drag at the submerged
     * midpoint and one gravity-proportional float force at the submerged
     * centroid. Very large bodies are subsampled with volume compensation.
     */
    private void buoyancyForBody(final SableBody sb) {
        final Body body = sb.body;

        // Reads come from the batch snapshot when fresh (one tick stale is
        // irrelevant for bulk fluid forces) and fall back to direct JNI reads
        // otherwise.
        final boolean fresh = sb.snapValid;
        final double comX, comY, comZ;
        final float qx, qy, qz, qw;
        final float lvx, lvy, lvz, avx, avy, avz;
        if (fresh) {
            comX = sb.snapComX;
            comY = sb.snapComY;
            comZ = sb.snapComZ;
            qx = sb.snapQx;
            qy = sb.snapQy;
            qz = sb.snapQz;
            qw = sb.snapQw;
            lvx = sb.snapVx;
            lvy = sb.snapVy;
            lvz = sb.snapVz;
            avx = sb.snapAx;
            avy = sb.snapAy;
            avz = sb.snapAz;
        } else {
            final RVec3 comPos = body.getCenterOfMassPosition();
            comX = comPos.xx();
            comY = comPos.yy();
            comZ = comPos.zz();
            final Quat rot = body.getRotation();
            qx = rot.getX();
            qy = rot.getY();
            qz = rot.getZ();
            qw = rot.getW();
            final Vec3 lin = body.getLinearVelocity();
            final Vec3 ang = body.getAngularVelocity();
            lvx = lin.getX();
            lvy = lin.getY();
            lvz = lin.getZ();
            avx = ang.getX();
            avy = ang.getY();
            avz = ang.getZ();
        }

        // Rotated unit axes give the world-space Y extent of every child cube:
        // an extent that stays correct (and smooth) for any body orientation.
        Vector3f r = this.rotateInto(1.0f, 0.0f, 0.0f, qx, qy, qz, qw);
        final double xAxY = r.y;
        r = this.rotateInto(0.0f, 1.0f, 0.0f, qx, qy, qz, qw);
        final double yAxY = r.y;
        r = this.rotateInto(0.0f, 0.0f, 1.0f, qx, qy, qz, qw);
        final double halfY = 0.5 * (Math.abs(xAxY) + Math.abs(yAxY) + Math.abs(r.y));

        // Accumulated force + torque: summing F and (p - com) x F over all
        // submerged samples in doubles and applying once is exactly equivalent
        // to one JNI call per sample (linearity).
        double accFx = 0.0, accFy = 0.0, accFz = 0.0;
        double accTx = 0.0, accTy = 0.0, accTz = 0.0;

        // Accumulated submerged volume and its centroid: the float force is applied
        // at the centroid of the submerged part, so a tilted body gets a smooth
        // righting torque and settles level.
        double floatVolume = 0.0;
        double centroidX = 0.0, centroidY = 0.0, centroidZ = 0.0;

        final FluidProbe probe = new FluidProbe();

        final double comBaseX = sb.centerOfMass.x;
        final double comBaseY = sb.centerOfMass.y;
        final double comBaseZ = sb.centerOfMass.z;

        // One sample per unique solid block (multi-box blocks such as stairs
        // count once). Very large bodies are subsampled with volume
        // compensation: fluid forces are a bulk effect.
        final int[] blocks = sb.buoyancyBlocks;
        final float[] volumes = sb.buoyancyVolumes;
        final int blockCount = sb.buoyancyCount;
        final int stride = Math.max(1, blockCount / BUOYANCY_MAX_CHILDREN);
        for (int bi = 0; bi < blockCount; bi += stride) {
            final double sampleScale = stride;
            final float lpx = (float) (blocks[bi * 3] + 0.5 - comBaseX);
            final float lpy = (float) (blocks[bi * 3 + 1] + 0.5 - comBaseY);
            final float lpz = (float) (blocks[bi * 3 + 2] + 0.5 - comBaseZ);

            // Inline rotateInto: same math, no ThreadLocal.get / Vector3f per block.
            final float tX = 2.0f * (qy * lpz - qz * lpy);
            final float tY = 2.0f * (qz * lpx - qx * lpz);
            final float tZ = 2.0f * (qx * lpy - qy * lpx);
            final double wx = comX + lpx + qw * tX + (qy * tZ - qz * tY);
            final double wy = comY + lpy + qw * tY + (qz * tX - qx * tZ);
            final double wz = comZ + lpz + qw * tZ + (qx * tY - qy * tX);

            final int wbx = floor(wx);
            final int wby = floor(wy);
            final int wbz = floor(wz);
            // Fluid at the block's center, or right below it: a block floating
            // high still dips its lower half into the water underneath.
            int fluidY = wby;
            double fluidHeight = probe.fluidHeight(wbx, wby, wbz);
            if (fluidHeight <= 0.0) {
                fluidY = wby - 1;
                fluidHeight = probe.fluidHeight(wbx, fluidY, wbz);
                if (fluidHeight <= 0.0) {
                    continue;
                }
            }
            final float mult = volumes[bi];

            // Vertical span of the rotated unit cube vs the fluid surface: the
            // submerged height is smooth in rotation. For a centrally-symmetric
            // cube the horizontal centroid of the submerged part is always its
            // center — no spurious yaw torque.
            final double fluidTop = fluidY + fluidHeight;
            final double minY = wy - halfY;
            final double maxY = wy + halfY;
            final double submerged = Math.min(maxY, fluidTop) - minY;
            if (submerged <= 0.0) {
                continue;
            }
            final double volume = Math.min(1.0, submerged / (halfY * 2.0)) * mult * sampleScale;
            final double midY = Math.min(maxY, fluidTop) - submerged * 0.5;

            // drag: F = -v * 1.7 * volume at the submerged midpoint — damps both
            // linear motion and rotation while submerged.
            final double rx = wx - comX, ry = midY - comY, rz = wz - comZ;
            final double vx = lvx + avy * rz - avz * ry;
            final double vy = lvy + avz * rx - avx * rz;
            final double vz = lvz + avx * ry - avy * rx;
            final double fdx = -vx * FLUID_DRAG * volume;
            final double fdy = -vy * FLUID_DRAG * volume;
            final double fdz = -vz * FLUID_DRAG * volume;
            accFx += fdx;
            accFy += fdy;
            accFz += fdz;
            accTx += ry * fdz - rz * fdy;
            accTy += rz * fdx - rx * fdz;
            accTz += rx * fdy - ry * fdx;

            // accumulate submerged volume for the single centroid float force
            floatVolume += volume;
            centroidX += wx * volume;
            centroidY += midY * volume;
            centroidZ += wz * volume;
        }

        if (floatVolume > 0.0) {
            // Float applied at the centroid of the submerged volume: zero torque
            // when level, smooth righting torque when tilted — stable flotation.
            final double fby = -this.gravityY * FLUID_DENSITY * floatVolume;
            final double pbx = centroidX / floatVolume;
            final double pbz = centroidZ / floatVolume;
            accFy += fby;
            // Buoyancy force is vertical (0, fby, 0): r x F = (-rz*fby, 0, rx*fby).
            final double brx = pbx - comX, brz = pbz - comZ;
            accTx += -brz * fby;
            accTz += brx * fby;
        }

        if (!(Double.isFinite(accFx) && Double.isFinite(accFy) && Double.isFinite(accFz)
                && Double.isFinite(accTx) && Double.isFinite(accTy) && Double.isFinite(accTz))) {
            return;
        }
        if (accFx != 0.0 || accFy != 0.0 || accFz != 0.0) {
            body.addForce((float) accFx, (float) accFy, (float) accFz);
            if (accTx != 0.0 || accTy != 0.0 || accTz != 0.0) {
                body.addTorque((float) accTx, (float) accTy, (float) accTz);
            }
        }
    }

    private static int floor(final double v) {
        return (int) Math.floor(v);
    }
    //endregion

    //region Contraption motion

    // Reusable objects for updateContraptionMotion (server thread only).
    private final RVec3 tmpMovePos = new RVec3();
    private final Quat tmpMoveRot = new Quat();

    /**
     * Drives the kinematic bodies of world contraptions to their target pose.
     * Jolt derives the body velocity from the motion, so contacts see the
     * contraption's real velocity. Mounted contraptions move with their mount's
     * compound instead (see {@link #assembleParent}).
     */
    private void updateContraptionMotion(final float dt) {
        for (final SableBody sb : this.contraptionList) {
            if (sb.body == null) {
                continue;
            }
            this.tmpMovePos.set(sb.relPos.x, sb.relPos.y, sb.relPos.z);
            this.tmpMoveRot.set((float) sb.relRot.x, (float) sb.relRot.y, (float) sb.relRot.z, (float) sb.relRot.w);
            this.bi.moveKinematic(sb.joltId, this.tmpMovePos, this.tmpMoveRot, dt);
        }
    }

    //endregion

    //region Collision reporting

    /**
     * Reused output buffer of {@link #clearCollisions} (server thread only).
     */
    private final double[] collisionOut = new double[100 * 15];
    private int collisionCount;

    /**
     * Reads & clears all reported collisions.
     * Each collision is formatted as:
     * [body_a, body_b, force_amount, local_normal_a, local_normal_b, local_point_a, local_point_b]
     */
    public double[] clearCollisions() {
        final int max = 100;
        synchronized (this.reportedLock) {
            if (this.reportedCollisions.size() > max) {
                this.reportedCollisions.subList(max, this.reportedCollisions.size()).clear();
            }
            this.collisionCount = this.reportedCollisions.size();
            int i = 0;
            for (final double[] rec : this.reportedCollisions) {
                System.arraycopy(rec, 0, this.collisionOut, i, 15);
                i += 15;
            }
            this.reportedCollisions.clear();
        }
        return this.collisionOut;
    }

    /**
     * Number of valid 15-double records in the last {@link #clearCollisions} result.
     */
    public int lastCollisionCount() {
        return this.collisionCount;
    }

    void reportCollision(final double[] rec) {
        synchronized (this.reportedLock) {
            if (this.reportedCollisions.size() < 100) {
                this.reportedCollisions.add(rec);
            }
        }
    }

    JoltVoxelColliderData.Registry registry() {
        return this.colliderRegistry;
    }


    //endregion

    //region Block contact callbacks

    /**
     * A block contact callback recorded during the step; invoked afterwards on
     * the server thread (callbacks read and edit the world, which must never
     * happen on a Jolt worker thread or inside the physics update).
     */
    private record PendingCallback(JoltVoxelColliderData entry, long blockKey, int x, int y, int z,
                                   int otherX, int otherY, int otherZ, boolean hasOther,
                                   double hitX, double hitY, double hitZ, double impactVelocity) {
    }

    /**
     * Callback result: tangent motion in the block's frame and whether the
     * contact is removed. Applied to the block's contacts in the next step.
     */
    private record CallbackResult(float tanX, float tanY, float tanZ, boolean remove) {
    }

    private final Object pendingCallbacksLock = new Object();
    private ObjectArrayList<PendingCallback> pendingCallbacks = new ObjectArrayList<>();
    /**
     * Results of the previous step's callbacks, keyed by block position (plot
     * and world coordinates never overlap). Replaced wholesale between steps;
     * read by contact callbacks during the update.
     */
    private volatile Long2ObjectOpenHashMap<CallbackResult> callbackResults = new Long2ObjectOpenHashMap<>();

    private void queueCallback(final PendingCallback callback) {
        synchronized (this.pendingCallbacksLock) {
            if (this.pendingCallbacks.size() < 4096) {
                this.pendingCallbacks.add(callback);
            }
        }
    }

    private long lastCallbackErrorNanos;

    /**
     * Invokes the block contact callbacks recorded during the last step. Must be
     * called on the server thread between steps (still inside Sable's physics
     * tick, as the callbacks expect).
     */
    public void processDeferredCallbacks() {
        final ObjectArrayList<PendingCallback> list;
        synchronized (this.pendingCallbacksLock) {
            if (this.pendingCallbacks.isEmpty()) {
                if (!this.callbackResults.isEmpty()) {
                    this.callbackResults = new Long2ObjectOpenHashMap<>();
                }
                return;
            }
            list = this.pendingCallbacks;
            this.pendingCallbacks = new ObjectArrayList<>();
        }
        final Long2ObjectOpenHashMap<CallbackResult> results = new Long2ObjectOpenHashMap<>();
        final LongOpenHashSet seen = new LongOpenHashSet();
        for (final PendingCallback p : list) {
            if (this.disposed) {
                return;
            }
            if (!seen.add(p.blockKey())) {
                continue; // one invocation per block per step
            }
            try {
                final double[] result = p.entry().contactEvents.onCollision(
                        p.x(), p.y(), p.z(), p.otherX(), p.otherY(), p.otherZ(),
                        p.hitX(), p.hitY(), p.hitZ(), p.impactVelocity(), p.hasOther());
                if (result != null && result.length >= 4) {
                    final boolean remove = result[3] > 0.0;
                    if (remove || result[0] != 0.0 || result[1] != 0.0 || result[2] != 0.0) {
                        results.put(p.blockKey(), new CallbackResult((float) result[0], (float) result[1], (float) result[2], remove));
                    }
                }
            } catch (final Throwable t) {
                final long now = System.nanoTime();
                if (now - this.lastCallbackErrorNanos > 5_000_000_000L) {
                    this.lastCallbackErrorNanos = now;
                    Sable.LOGGER.error("Block contact callback failed", t);
                }
            }
        }
        this.callbackResults = results;
    }

    private static long blockKey(final int x, final int y, final int z) {
        return (((long) x & 0x3FFFFFFL) << 38) | (((long) z & 0x3FFFFFFL) << 12) | ((long) y & 0xFFFL);
    }

    //endregion

    //region Constraints

    /**
     * Whether a constraint side can be built: the static world (no body), or a
     * body with a live native body. Native constraint creation with a missing
     * body would crash.
     */
    public boolean canConstrain(@Nullable final dev.ryanhcode.sable.api.physics.PhysicsPipelineBody body, @Nullable final SableBody sb) {
        if (this.disposed) {
            return false;
        }
        return body == null || (sb != null && sb.body != null && sb.joltId != 0);
    }

    public TwoBodyConstraint createConstraint(final TwoBodyConstraintSettings settings, final int joltIdA, final int joltIdB) {
        final TwoBodyConstraint constraint = this.bi.createConstraint(settings, joltIdA, joltIdB);
        this.system.addConstraint(constraint);
        return constraint;
    }

    public void removeConstraint(final Constraint constraint) {
        if (constraint != null && !this.disposed) {
            this.system.removeConstraint(constraint);
        }
    }

    public long registerJoint(final JointRecord record) {
        record.id = this.nextJointId.getAndIncrement();
        this.joints.put(record.id, record);
        this.countAttachment(record.joltIdA, 1);
        this.countAttachment(record.joltIdB, 1);
        return record.id;
    }

    @Nullable
    public JointRecord joint(final long id) {
        return this.joints.get(id);
    }

    public void removeJoint(final long id) {
        final JointRecord rec = this.joints.remove(id);
        if (rec == null) {
            return;
        }
        this.countAttachment(rec.joltIdA, -1);
        this.countAttachment(rec.joltIdB, -1);
        if (rec.damper) {
            this.damperJoints.remove(rec);
        }
        if (rec.constraint != null) {
            this.removeConstraint(rec.constraint);
            rec.constraint = null;
        }
    }

    private final ObjectArrayList<JointRecord> damperJoints = new ObjectArrayList<>();
    private final DamperSide damperA = new DamperSide();
    private final DamperSide damperB = new DamperSide();
    private final Vector3d damperAxis = new Vector3d();
    private final Vector3d damperArm = new Vector3d();

    /**
     * Makes a joint a viscous damper. Every caller passes one value for all linear
     * axes and one for all angular axes, so one coefficient per group is enough.
     */
    public void setJointDamping(final JointRecord rec, final boolean linear, final double damping) {
        if (!rec.damper) {
            rec.damper = true;
            this.damperJoints.add(rec);
        }
        if (linear) {
            rec.linearDamping = damping;
        } else {
            rec.angularDamping = damping;
        }
    }

    /**
     * Applies drag between the two anchors of each damper joint, as forces for the
     * coming substep: dv/dt = -c * v on the relative velocity, independent of mass
     * (the behaviour of Rapier's acceleration-based motors), integrated implicitly
     * so a large coefficient cannot overshoot. Sleeping bodies are not touched:
     * forces on them would pile up until they wake.
     */
    private void applyJointDampers(final double dt) {
        for (int i = 0; i < this.damperJoints.size(); i++) {
            final JointRecord rec = this.damperJoints.get(i);
            final SableBody sbA = this.bodiesByJoltId.get(rec.joltIdA);
            final SableBody sbB = this.bodiesByJoltId.get(rec.joltIdB);
            if (!(rec.genericSettings instanceof final SixDofConstraintSettings settings)) {
                continue;
            }
            final DamperSide a = this.damperA.load(sbA, settings.getPosition1());
            final DamperSide b = this.damperB.load(sbB, settings.getPosition2());
            if (!a.driven && !b.driven) {
                continue;
            }

            // Linear: relative velocity of the anchors, along its own direction.
            final double vx = b.vx + b.wy * b.rz - b.wz * b.ry - (a.vx + a.wy * a.rz - a.wz * a.ry);
            final double vy = b.vy + b.wz * b.rx - b.wx * b.rz - (a.vy + a.wz * a.rx - a.wx * a.rz);
            final double vz = b.vz + b.wx * b.ry - b.wy * b.rx - (a.vz + a.wx * a.ry - a.wy * a.rx);
            final double speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
            if (rec.linearDamping > 0.0 && speed > 1.0e-5) {
                final Vector3d n = this.damperAxis.set(vx / speed, vy / speed, vz / speed);
                final double k = a.inverseMassAt(n, this.damperArm) + b.inverseMassAt(n, this.damperArm);
                if (k > 1.0e-12) {
                    final double force = rec.linearDamping * dt / (1.0 + rec.linearDamping * dt) * speed / k / dt;
                    a.pushAtAnchor(force * n.x, force * n.y, force * n.z);
                    b.pushAtAnchor(-force * n.x, -force * n.y, -force * n.z);
                }
            }

            // Angular: relative spin, along its own axis.
            final double ox = b.wx - a.wx;
            final double oy = b.wy - a.wy;
            final double oz = b.wz - a.wz;
            final double spin = Math.sqrt(ox * ox + oy * oy + oz * oz);
            if (rec.angularDamping > 0.0 && spin > 1.0e-5) {
                final Vector3d n = this.damperAxis.set(ox / spin, oy / spin, oz / spin);
                final double k = a.inverseInertiaAlong(n) + b.inverseInertiaAlong(n);
                if (k > 1.0e-12) {
                    final double torque = rec.angularDamping * dt / (1.0 + rec.angularDamping * dt) * spin / k / dt;
                    a.twist(torque * n.x, torque * n.y, torque * n.z);
                    b.twist(-torque * n.x, -torque * n.y, -torque * n.z);
                }
            }
        }
    }

    /** One side of a damper joint, read fresh for each substep. The static world reads as all zeros. */
    private static final class DamperSide {
        private final Vec3 vec = new Vec3();
        private final Quat rotation = new Quat();
        private final Quat inertiaRotation = new Quat();
        private final Quaterniond toPrincipal = new Quaterniond();
        private final Vector3d principal = new Vector3d();
        private Body body;
        /** Awake and dynamic: forces apply. Otherwise it only contributes its velocity. */
        boolean driven;
        double rx, ry, rz, vx, vy, vz, wx, wy, wz;
        private double inverseMass, ix, iy, iz;

        DamperSide load(@Nullable final SableBody sb, final RVec3 localAnchor) {
            this.driven = false;
            this.rx = this.ry = this.rz = this.vx = this.vy = this.vz = this.wx = this.wy = this.wz = 0.0;
            this.body = sb == null ? null : sb.body;
            if (this.body == null) {
                return this;
            }
            this.body.getLinearVelocity(this.vec);
            this.vx = this.vec.getX();
            this.vy = this.vec.getY();
            this.vz = this.vec.getZ();
            this.body.getAngularVelocity(this.vec);
            this.wx = this.vec.getX();
            this.wy = this.vec.getY();
            this.wz = this.vec.getZ();
            this.body.getRotation(this.rotation);
            final Vector3d r = this.toPrincipal.set(this.rotation.getX(), this.rotation.getY(), this.rotation.getZ(), this.rotation.getW())
                    .transform(this.principal.set(localAnchor.xx(), localAnchor.yy(), localAnchor.zz()));
            this.rx = r.x;
            this.ry = r.y;
            this.rz = r.z;
            this.driven = this.body.isDynamic() && this.body.isActive();
            if (this.driven) {
                final var motion = this.body.getMotionProperties();
                this.inverseMass = motion.getInverseMass();
                motion.getInverseInertiaDiagonal(this.vec);
                this.ix = this.vec.getX();
                this.iy = this.vec.getY();
                this.iz = this.vec.getZ();
                motion.getInertiaRotation(this.inertiaRotation);
                // World to principal axes: (body rotation * inertia rotation)^-1.
                this.toPrincipal.mul(this.inertiaRotation.getX(), this.inertiaRotation.getY(), this.inertiaRotation.getZ(), this.inertiaRotation.getW()).conjugate();
            }
            return this;
        }

        double inverseInertiaAlong(final Vector3d axis) {
            if (!this.driven) {
                return 0.0;
            }
            final Vector3d p = this.toPrincipal.transform(this.principal.set(axis));
            return p.x * p.x * this.ix + p.y * p.y * this.iy + p.z * p.z * this.iz;
        }

        /** 1 / effective mass of the anchor along a direction. */
        double inverseMassAt(final Vector3d direction, final Vector3d arm) {
            if (!this.driven) {
                return 0.0;
            }
            arm.set(this.ry * direction.z - this.rz * direction.y, this.rz * direction.x - this.rx * direction.z, this.rx * direction.y - this.ry * direction.x);
            return this.inverseMass + this.inverseInertiaAlong(arm);
        }

        void pushAtAnchor(final double fx, final double fy, final double fz) {
            if (this.driven && Double.isFinite(fx + fy + fz)) {
                this.body.addForce((float) fx, (float) fy, (float) fz);
                this.body.addTorque((float) (this.ry * fz - this.rz * fy), (float) (this.rz * fx - this.rx * fz), (float) (this.rx * fy - this.ry * fx));
            }
        }

        void twist(final double tx, final double ty, final double tz) {
            if (this.driven && Double.isFinite(tx + ty + tz)) {
                this.body.addTorque((float) tx, (float) ty, (float) tz);
            }
        }
    }

    private void countAttachment(final int joltId, final int delta) {
        final SableBody sb = this.bodiesByJoltId.get(joltId);
        if (sb != null) {
            sb.attachmentCount = Math.max(0, sb.attachmentCount + delta);
        }
    }

    /**
     * Keeps joints anchored at the same block-space point when a body's center
     * of mass moves (Rapier re-derives anchors from the current center of mass
     * every tick; Jolt stores them relative to it).
     */
    private void notifyComShift(final int joltId, final float dx, final float dy, final float dz) {
        final Vec3 delta = this.tmpVec.get();
        delta.set(dx, dy, dz);
        for (final JointRecord rec : this.joints.values()) {
            if (rec.constraint == null || (rec.joltIdA != joltId && rec.joltIdB != joltId)) {
                continue;
            }
            rec.constraint.notifyShapeChanged(joltId, delta);
            // The servo math of six-DOF handles reads the anchors from the settings.
            if (rec.genericSettings instanceof final SixDofConstraintSettings settings) {
                if (rec.joltIdA == joltId) {
                    final RVec3 p = settings.getPosition1();
                    settings.setPosition1(new RVec3(p.xx() - dx, p.yy() - dy, p.zz() - dz));
                }
                if (rec.joltIdB == joltId) {
                    final RVec3 p = settings.getPosition2();
                    settings.setPosition2(new RVec3(p.xx() - dx, p.yy() - dy, p.zz() - dz));
                }
            }
        }
        for (final RopeStrand strand : this.ropes.values()) {
            if (strand.start != null && strand.start.mountJoltId == joltId && strand.start.constraint != null) {
                strand.start.constraint.notifyShapeChanged(joltId, delta);
            }
            if (strand.end != null && strand.end.mountJoltId == joltId && strand.end.constraint != null) {
                strand.end.constraint.notifyShapeChanged(joltId, delta);
            }
        }
    }

    //endregion

    //region Ropes

    private int createRopePoint(final double x, final double y, final double z, final double pointRadius) {
        final float radius = (float) Math.max(pointRadius, 0.01);
        final BodyCreationSettings bcs = new BodyCreationSettings(
                this.sharedBoxShape(radius, radius, radius),
                new RVec3(x, y, z), Quat.sIdentity(), EMotionType.Dynamic, LAYER_ROPE);
        bcs.setLinearDamping((float) (this.universalDrag + 6.0));
        bcs.setAngularDamping((float) this.universalDrag);
        bcs.setInertiaMultiplier(0.0f);
        // Rapier's rope friction; the scene combines frictions as Rapier does in
        // Rapier-compatible mode (average).
        bcs.setFriction(FRICTION_ROPE);
        final MassProperties mp = new MassProperties();
        mp.setMass(0.35f);
        bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
        bcs.setMassPropertiesOverride(mp);

        final Body body;
        try {
            body = this.bi.createBody(bcs);
        } catch (final IllegalStateException e) {
            this.warnCapacity("rope point was not created");
            return -1;
        }
        body.setUserData(-1L);
        this.bi.addBody(body, EActivation.Activate);
        this.ropePointBodies.put(body.getId(), body);
        return body.getId();
    }

    private final Int2ObjectOpenHashMap<Body> ropePointBodies = new Int2ObjectOpenHashMap<>();

    private void destroyRopePoint(final int bodyId) {
        if (this.ropePointBodies.remove(bodyId) == null) {
            return;
        }
        this.bi.removeBody(bodyId);
        this.bi.destroyBody(bodyId);
    }

    private DistanceConstraint createRopeJoint(final int joltIdA, final int joltIdB, final double length) {
        final DistanceConstraintSettings settings = new DistanceConstraintSettings();
        settings.setPoint1(RVec3.sZero());
        settings.setPoint2(RVec3.sZero());
        settings.setMinDistance(0.0f);
        settings.setMaxDistance((float) length);
        final SpringSettings spring = settings.getLimitsSpringSettings();
        spring.setMode(ESpringMode.FrequencyAndDamping);
        spring.setFrequency(30.0f);
        spring.setDamping(1.0f);
        return (DistanceConstraint) this.createConstraint(settings, joltIdA, joltIdB);
    }

    public long createRope(final double pointRadius, final double firstJointLength, final double[] points, final int pointCount) {
        final long id = this.nextRopeId.getAndIncrement();
        if (this.disposed) {
            return id;
        }
        final RopeStrand strand = new RopeStrand();
        strand.pointRadius = pointRadius;
        strand.firstJointLength = firstJointLength;

        for (int i = 0; i < pointCount; i++) {
            final int point = this.createRopePoint(points[i * 3], points[i * 3 + 1], points[i * 3 + 2], pointRadius);
            if (point != -1) {
                strand.points.add(point);
            }
        }
        for (int i = 0; i < strand.points.size() - 1; i++) {
            final double length = i == 0 ? firstJointLength : 1.0;
            strand.joints.add(this.createRopeJoint(strand.points.getInt(i), strand.points.getInt(i + 1), length));
        }

        this.ropes.put(id, strand);
        return id;
    }

    public void removeRope(final long ropeId) {
        final RopeStrand strand = this.ropes.remove(ropeId);
        if (strand == null) {
            return;
        }
        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
        }
        if (strand.end != null) {
            this.removeRopeAttachment(strand.end);
        }
        for (final DistanceConstraint joint : strand.joints) {
            this.removeConstraint(joint);
        }
        for (final int point : strand.points) {
            this.destroyRopePoint(point);
        }
    }

    private void removeRopeAttachment(final RopeAttachment attachment) {
        this.removeConstraint(attachment.constraint);
        attachment.constraint = null;
        this.countAttachment(attachment.mountJoltId, -1);
    }

    public double[] queryRope(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return new double[0];
        }
        final double[] out = new double[strand.points.size() * 3];
        final RVec3 p = this.tmpCom.get();
        for (int i = 0; i < strand.points.size(); i++) {
            final Body body = this.ropePointBodies.get(strand.points.getInt(i));
            if (body == null) {
                continue;
            }
            body.getPosition(p);
            out[i * 3] = p.xx();
            out[i * 3 + 1] = p.yy();
            out[i * 3 + 2] = p.zz();
        }
        return out;
    }

    public void setRopeFirstSegmentLength(final long ropeId, final double length) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.joints.isEmpty()) {
            return;
        }
        strand.firstJointLength = length;
        strand.joints.get(0).setDistance(0.0f, (float) length);
    }

    public void addRopePointAtStart(final long ropeId, final double x, final double y, final double z) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return;
        }
        final int bodyId = this.createRopePoint(x, y, z, strand.pointRadius);
        if (bodyId == -1) {
            return;
        }
        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, 1.0f);
        }
        if (!strand.points.isEmpty()) {
            strand.joints.add(0, this.createRopeJoint(bodyId, strand.points.getInt(0), strand.firstJointLength));
        }
        strand.points.add(0, bodyId);

        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
            strand.start = null;
        }
    }

    public void removeRopePointAtStart(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.points.isEmpty()) {
            return;
        }
        if (strand.start != null) {
            this.removeRopeAttachment(strand.start);
            strand.start = null;
        }
        final int point = strand.points.removeInt(0);
        if (!strand.joints.isEmpty()) {
            this.removeConstraint(strand.joints.remove(0));
        }
        this.destroyRopePoint(point);

        if (!strand.joints.isEmpty()) {
            strand.joints.get(0).setDistance(0.0f, (float) strand.firstJointLength);
        }
    }

    public void wakeUpRope(final long ropeId) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null) {
            return;
        }
        for (final int point : strand.points) {
            this.bi.activateBody(point);
        }
    }

    public void setRopeAttachment(final long ropeId, final int subLevelId, final double x, final double y, final double z, final boolean end) {
        final RopeStrand strand = this.ropes.get(ropeId);
        if (strand == null || strand.points.isEmpty()) {
            return;
        }

        final int ropeBody = end ? strand.points.getInt(strand.points.size() - 1) : strand.points.getInt(0);
        final SableBody mountSb = subLevelId != -1 ? this.bodies.get(subLevelId) : null;
        if (subLevelId != -1 && (mountSb == null || mountSb.body == null)) {
            return;
        }
        final int mountBody = mountSb != null ? mountSb.joltId : this.groundBodyId;

        // Block-space anchor relative to the mount's center of mass (Rapier:
        // pos - center_of_mass); later COM shifts are applied by notifyComShift.
        final double comX = mountSb != null ? mountSb.centerOfMass.x : 0.0;
        final double comY = mountSb != null ? mountSb.centerOfMass.y : 0.0;
        final double comZ = mountSb != null ? mountSb.centerOfMass.z : 0.0;

        final PointConstraintSettings settings = new PointConstraintSettings();
        settings.setSpace(com.github.stephengold.joltjni.enumerate.EConstraintSpace.LocalToBodyCom);
        settings.setPoint1(new RVec3(x - comX, y - comY, z - comZ));
        settings.setPoint2(RVec3.sZero());
        final Constraint constraint = this.createConstraint(settings, mountBody, ropeBody);

        final RopeAttachment attachment = new RopeAttachment();
        attachment.mountJoltId = mountBody;
        attachment.x = x;
        attachment.y = y;
        attachment.z = z;
        attachment.constraint = constraint;
        this.countAttachment(mountBody, 1);

        if (end) {
            if (strand.end != null) {
                this.removeRopeAttachment(strand.end);
            }
            strand.end = attachment;
        } else {
            if (strand.start != null) {
                this.removeRopeAttachment(strand.start);
            }
            strand.start = attachment;
        }
    }

    //endregion

    //region Config

    public void configSolverIterations(final int solverIterations, final int pgsIterations) {
        if (this.disposed) {
            return;
        }
        // clamped to the minimums that keep dragged bodies from sinking into walls
        this.configuredVelocitySteps = Math.max(10, solverIterations);
        this.configuredPositionSteps = Math.max(4, pgsIterations);
        final var settings = this.system.getPhysicsSettings();
        settings.setNumVelocitySteps(this.reducedSolverApplied ? Math.max(5, this.configuredVelocitySteps / 2) : this.configuredVelocitySteps);
        settings.setNumPositionSteps(this.reducedSolverApplied ? Math.max(2, this.configuredPositionSteps / 2) : this.configuredPositionSteps);
        this.system.setPhysicsSettings(settings);
    }

    //endregion

    private static int unpackChunkX(final long key) {
        return (int) (key >> 42);
    }

    private static int unpackChunkY(final long key) {
        return (int) ((key << 44) >> 44);
    }

    private static int unpackChunkZ(final long key) {
        return (int) ((key << 22) >> 42);
    }

    /**
     * Per-thread cache of native body addresses to Java wrappers for the contact
     * listener. A {@code Body} wrapper is stateless (every method re-reads through
     * the stored address) and the native address is stable for the body's lifetime,
     * so the cache is exact; each fresh wrapper otherwise costs 3 heap objects
     * per contact callback. Manifold/settings addresses are callback-scoped
     * temporaries and must NOT be cached.
     */
    private static final ThreadLocal<it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap<Body>> BODY_WRAPPER_CACHE =
            ThreadLocal.withInitial(it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap::new);

    private static Body wrapContactBody(final long bodyVa) {
        final var cache = BODY_WRAPPER_CACHE.get();
        Body wrapper = cache.get(bodyVa);
        if (wrapper == null) {
            wrapper = new Body(bodyVa);
            cache.put(bodyVa, wrapper);
            if (cache.size() > 4096) {
                // Bound entries churned by short-lived bodies (rope points);
                // repopulates lazily and stays correct (address-keyed).
                cache.clear();
            }
        }
        return wrapper;
    }

    private long lastListenerErrorNanos;

    private void logListenerFailure(final Throwable t) {
        final long now = System.nanoTime();
        if (now - this.lastListenerErrorNanos > 5_000_000_000L) {
            this.lastListenerErrorNanos = now;
            Sable.LOGGER.error("[SableJolt] contact listener failed", t);
        }
    }

    /**
     * The scene contact listener: reports collisions, records block contact
     * callbacks, applies friction / restitution overrides and the surface
     * velocities of mounted contraptions. Runs on Jolt worker threads: it only
     * reads scene state and never throws into native code.
     */
    private final class SceneContactListener extends FilteredContactListener {

        @Override
        public void onContactAdded(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        @Override
        public void onContactPersisted(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            this.handle(body1Va, body2Va, manifoldVa, settingsVa);
        }

        private void handle(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            // A Java exception must never propagate into the native solver thread.
            try {
                this.handleUnchecked(body1Va, body2Va, manifoldVa, settingsVa);
            } catch (final Throwable t) {
                JoltPhysicsScene.this.logListenerFailure(t);
            }
        }

        private void handleUnchecked(final long body1Va, final long body2Va, final long manifoldVa, final long settingsVa) {
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            if (scene.disposed) {
                return;
            }
            // Bodies are long-lived: wrappers come from the cache (no allocation).
            // Manifold/settings are callback-scoped and must wrap fresh.
            final Body body1 = JoltPhysicsScene.wrapContactBody(body1Va);
            final Body body2 = JoltPhysicsScene.wrapContactBody(body2Va);
            final int id1 = body1.getId();
            final int id2 = body2.getId();

            final SableBody sb1 = scene.bodiesByJoltId.get(id1);
            final SableBody sb2 = scene.bodiesByJoltId.get(id2);
            final RegionLayer gc1 = sb1 == null ? scene.terrainRegionsByBodyId.get(id1) : null;
            final RegionLayer gc2 = sb2 == null ? scene.terrainRegionsByBodyId.get(id2) : null;

            if (sb1 == null && sb2 == null && gc1 == null && gc2 == null) {
                return;
            }

            final ContactManifold manifold = new ContactManifold(manifoldVa);
            final ContactSettings settings = new ContactSettings(settingsVa);

            // Deepest-penetration tracking for creatively dragged bodies: the
            // end-of-step push-out keeps the wand from grinding them into the
            // terrain faster than the solver can recover.
            if (sb1 != null && scene.creativeDragBodies.contains(sb1.joltId)) {
                this.accumulateDragPenetration(sb1, body1, manifold, true);
            }
            if (sb2 != null && scene.creativeDragBodies.contains(sb2.joltId)) {
                this.accumulateDragPenetration(sb2, body2, manifold, false);
            }

            final Child c1 = sb1 != null ? sb1.childAt(manifold.getSubShapeId1()) : gc1 != null ? gc1.childAt(manifold.getSubShapeId1()) : null;
            final Child c2 = sb2 != null ? sb2.childAt(manifold.getSubShapeId2()) : gc2 != null ? gc2.childAt(manifold.getSubShapeId2()) : null;

            final JoltVoxelColliderData entry1 = c1 != null ? scene.colliderRegistry.get(c1.colliderId - 1) : null;
            final JoltVoxelColliderData entry2 = c2 != null ? scene.colliderRegistry.get(c2.colliderId - 1) : null;

            // friction / restitution overrides (block-level)
            float mult = (entry1 == null ? 1.0f : entry1.frictionMultiplier) * (entry2 == null ? 1.0f : entry2.frictionMultiplier);
            float restitution = Math.max(entry1 == null ? 0.0f : entry1.restitution, entry2 == null ? 0.0f : entry2.restitution);
            // One manifold here can span blocks of different kinds (Jolt merges
            // the contacts of a body pair that share a normal): sample the block
            // pairs instead when either side may mix them.
            final boolean special1 = gc1 != null && gc1.layer == TERRAIN_SPECIAL, special2 = gc2 != null && gc2.layer == TERRAIN_SPECIAL;
            if (!special1 && !special2) {
                final SableBody driver = sb1 != null && sb1.body != null && sb1.kind != SableBody.Kind.BOX ? sb1
                        : sb2 != null && sb2.body != null && sb2.kind != SableBody.Kind.BOX ? sb2 : null;
                final SableBody other = driver == sb1 ? sb2 : sb1;
                final boolean terrain = gc1 != null || gc2 != null;
                if (driver != null && (driver.layerSpecial || (terrain && driver.nearSpecial) || (other != null && other.layerSpecial))) {
                    final Vec3 n = manifold.getWorldSpaceNormal();
                    final float sign = driver == sb1 ? 1.0f : -1.0f;
                    final float[] sampled = this.sampleScratch.get();
                    if (this.sampleBlockPairs(driver, terrain ? null : other, terrain, sign * n.getX(), sign * n.getY(), sign * n.getZ(), sampled)) {
                        mult = sampled[0];
                        restitution = sampled[1];
                    }
                }
            }
            if (scene.rapierMode) {
                // Rapier: average of the collider frictions times both multipliers.
                final float f1 = frictionOf(sb1, gc1, c1);
                final float f2 = frictionOf(sb2, gc2, c2);
                settings.setCombinedFriction((f1 + f2) * 0.5f * mult);
            } else if (mult != 1.0f) {
                settings.setCombinedFriction(settings.getCombinedFriction() * mult);
            }
            if (restitution > 0.0f) {
                settings.setCombinedRestitution(restitution);
            }

            // Mounted contraption blocks move relative to their mount's body:
            // their motion becomes contact surface velocity (Rapier's fake velocities).
            this.applyContraptionSurfaceVelocity(c1, sb1, settings, true);
            this.applyContraptionSurfaceVelocity(c2, sb2, settings, false);

            final boolean callback1 = entry1 != null && entry1.contactEvents != null && !isContraptionBlock(c1);
            final boolean callback2 = entry2 != null && entry2.contactEvents != null && !isContraptionBlock(c2);
            final boolean report = c1 != null || c2 != null;
            if (!callback1 && !callback2 && !report) {
                return;
            }

            // Approximate contact point: the block of the callback side (or of
            // body 1) moved half a block towards the other body along the normal.
            final Vec3 n = manifold.getWorldSpaceNormal();
            final double nx = n.getX(), ny = n.getY(), nz = n.getZ();

            if (callback1) {
                this.recordCallback(entry1, c1, c2, sb1, gc2 != null, nx, ny, nz, sb1, sb2, true, settings);
            }
            if (callback2) {
                this.recordCallback(entry2, c2, c1, sb2, gc1 != null, -nx, -ny, -nz, sb1, sb2, false, settings);
            }

            if (report) {
                this.report(sb1, sb2, c1, c2, body1, body2, nx, ny, nz);
            }
        }

        private static boolean isContraptionBlock(@Nullable final Child c) {
            return c != null && c.owner != null && c.owner.kind == SableBody.Kind.CONTRAPTION;
        }

        private final ThreadLocal<float[]> sampleScratch = ThreadLocal.withInitial(() -> new float[2]);

        /**
         * Support blocks of a contact: within this distance (blocks) of the
         * deepest corner block along the normal.
         */
        private static final double SUPPORT_TOLERANCE = 0.05;
        private static final int MAX_SAMPLES = 32;

        /**
         * Friction multiplier and restitution of a contact taken per block pair,
         * as Rapier's hooks do per voxel pair (hooks.rs): the corner blocks of
         * {@code driver}'s leaves deepest along the contact normal (on a flat
         * contact the corners of its contact face, the only blocks Rapier's
         * neighbourhood culling keeps there) against the solid block right
         * beyond each of them on the other side.
         *
         * @param other  the other voxel body, null for the terrain
         * @param dx     contact normal from {@code driver} towards the other side
         * @param out    receives the mean of the friction multiplier products and
         *               the mean of the restitutions
         * @return false when no pair of solid blocks was found
         */
        private boolean sampleBlockPairs(final SableBody driver, @Nullable final SableBody other, final boolean terrain,
                                         final float dx, final float dy, final float dz, final float[] out) {
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            final ObjectArrayList<Child> leaves = driver.children;
            final int count = leaves.size();
            if (count == 0 || count > 4096 || (!terrain && (other == null || other.body == null))) {
                return false;
            }
            final Vector3f dl = scene.rotateInto(dx, dy, dz, -driver.snapQx, -driver.snapQy, -driver.snapQz, driver.snapQw);
            final double lx = dl.x, ly = dl.y, lz = dl.z;
            final double comX = driver.centerOfMass.x, comY = driver.centerOfMass.y, comZ = driver.centerOfMass.z;
            final Object[] elements = leaves.elements();
            double deepest = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < count; i++) {
                final Child c = (Child) elements[i];
                final double px = (lx >= 0.0 ? c.ex : c.bx) + 0.5 - comX;
                final double py = (ly >= 0.0 ? c.ey : c.by) + 0.5 - comY;
                final double pz = (lz >= 0.0 ? c.ez : c.bz) + 0.5 - comZ;
                deepest = Math.max(deepest, px * lx + py * ly + pz * lz);
            }
            final double limit = deepest - SUPPORT_TOLERANCE;
            double multSum = 0.0, restSum = 0.0;
            int samples = 0;
            for (int i = 0; i < count && samples < MAX_SAMPLES; i++) {
                final Child c = (Child) elements[i];
                final JoltVoxelColliderData own = scene.colliderRegistry.get(c.colliderId - 1);
                for (int corner = 0; corner < 8 && samples < MAX_SAMPLES; corner++) {
                    final int bx = (corner & 1) == 0 ? c.bx : c.ex;
                    final int by = (corner & 2) == 0 ? c.by : c.ey;
                    final int bz = (corner & 4) == 0 ? c.bz : c.ez;
                    // the same block twice when the leaf is one block thick along an axis
                    if (((corner & 1) != 0 && c.ex == c.bx) || ((corner & 2) != 0 && c.ey == c.by) || ((corner & 4) != 0 && c.ez == c.bz)) {
                        continue;
                    }
                    final double px = bx + 0.5 - comX, py = by + 0.5 - comY, pz = bz + 0.5 - comZ;
                    if (px * lx + py * ly + pz * lz < limit) {
                        continue;
                    }
                    final Vector3f w = scene.rotateInto((float) px, (float) py, (float) pz, driver.snapQx, driver.snapQy, driver.snapQz, driver.snapQw);
                    // From the block center past its face into whatever it touches.
                    final double wx = driver.snapComX + w.x + dx * 0.8, wy = driver.snapComY + w.y + dy * 0.8, wz = driver.snapComZ + w.z + dz * 0.8;
                    final JoltVoxelColliderData beyond = terrain ? scene.terrainEntryAt(floor(wx), floor(wy), floor(wz)) : scene.bodyEntryAt(other, wx, wy, wz);
                    if (beyond == null) {
                        continue;
                    }
                    multSum += (own == null ? 1.0f : own.frictionMultiplier) * beyond.frictionMultiplier;
                    restSum += Math.max(own == null ? 0.0f : own.restitution, beyond.restitution);
                    samples++;
                }
            }
            if (samples == 0) {
                return false;
            }
            out[0] = (float) (multSum / samples);
            out[1] = (float) (restSum / samples);
            return true;
        }

        private static float frictionOf(@Nullable final SableBody sb, @Nullable final RegionLayer gc, @Nullable final Child c) {
            if (gc != null) {
                return FRICTION_TERRAIN_RAPIER;
            }
            if (sb == null) {
                return FRICTION_ROPE;
            }
            if (isContraptionBlock(c) || sb.kind == SableBody.Kind.CONTRAPTION) {
                return FRICTION_CONTRAPTION;
            }
            return sb.kind == SableBody.Kind.BOX ? FRICTION_BOX : FRICTION_SUB_LEVEL;
        }

        /**
         * Records the deepest contact penetration of a dragged body this step.
         * The manifold normal moves body 2 out of collision, so it is flipped
         * for body 1; the result is additionally verified against the
         * contact-to-COM direction so a wrong convention can never push the
         * body deeper.
         */
        private void accumulateDragPenetration(final SableBody dragged, final Body self,
                                               final ContactManifold manifold, final boolean isFirst) {
            final float depth = manifold.getPenetrationDepth();
            if (depth <= 0.0f || depth <= dragged.dragPenDepth) {
                return;
            }
            final Vec3 n = manifold.getWorldSpaceNormal();
            double nx = n.getX(), ny = n.getY(), nz = n.getZ();
            if (isFirst) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            final RVec3 com = self.getCenterOfMassPosition();
            final RVec3 base = manifold.getBaseOffset();
            final double ox = com.xx() - base.xx(), oy = com.yy() - base.yy(), oz = com.zz() - base.zz();
            if (nx * ox + ny * oy + nz * oz < 0.0) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            dragged.dragPenDepth = depth;
            dragged.dragPenNx = nx;
            dragged.dragPenNy = ny;
            dragged.dragPenNz = nz;
        }

        /**
         * World-space velocity of a body (snapshot of the last step) at a point.
         */
        private static void velocityAt(@Nullable final SableBody sb, final double px, final double py, final double pz,
                                       final double[] out, final int offset) {
            if (sb == null || sb.body == null) {
                out[offset] = out[offset + 1] = out[offset + 2] = 0.0;
                return;
            }
            final double rx = px - sb.snapComX, ry = py - sb.snapComY, rz = pz - sb.snapComZ;
            out[offset] = sb.snapVx + sb.snapAy * rz - sb.snapAz * ry;
            out[offset + 1] = sb.snapVy + sb.snapAz * rx - sb.snapAx * rz;
            out[offset + 2] = sb.snapVz + sb.snapAx * ry - sb.snapAy * rx;
        }

        private final ThreadLocal<double[]> velScratch = ThreadLocal.withInitial(() -> new double[6]);

        /**
         * Records a block contact callback for the server thread and applies the
         * result of the previous step's invocation for this block.
         *
         * @param dirX direction from the block towards the other body (world)
         */
        private void recordCallback(final JoltVoxelColliderData entry, final Child c, @Nullable final Child other,
                                    @Nullable final SableBody self, final boolean otherIsTerrain,
                                    final double dirX, final double dirY, final double dirZ,
                                    @Nullable final SableBody sb1, @Nullable final SableBody sb2,
                                    final boolean isFirst, final ContactSettings settings) {
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            final long key = blockKey(c.bx, c.by, c.bz);

            // Apply last step's result (tangent motion in the block's frame).
            final CallbackResult result = scene.callbackResults.get(key);
            if (result != null) {
                if (result.remove()) {
                    settings.setIsSensor(true);
                }
                if (result.tanX() != 0.0f || result.tanY() != 0.0f || result.tanZ() != 0.0f) {
                    float tx = result.tanX(), ty = result.tanY(), tz = result.tanZ();
                    if (self != null) {
                        final Vector3f r = scene.rotateInto(tx, ty, tz, self.snapQx, self.snapQy, self.snapQz, self.snapQw);
                        tx = r.x;
                        ty = r.y;
                        tz = r.z;
                    }
                    // remove the normal component (Rapier projects onto the tangent plane)
                    final float dot = (float) (tx * dirX + ty * dirY + tz * dirZ);
                    tx -= (float) (dirX * dot);
                    ty -= (float) (dirY * dot);
                    tz -= (float) (dirZ * dot);
                    final float sign = isFirst ? -1.0f : 1.0f;
                    final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
                    final Vec3 combined = scene.tmpVec.get();
                    combined.set(existing.getX() + sign * tx, existing.getY() + sign * ty, existing.getZ() + sign * tz);
                    settings.setRelativeLinearSurfaceVelocity(combined);
                }
            }

            // Contact point estimate in world space.
            double cxw, cyw, czw;
            if (self != null) {
                final float lx = (float) (c.bx + 0.5 - self.centerOfMass.x);
                final float ly = (float) (c.by + 0.5 - self.centerOfMass.y);
                final float lz = (float) (c.bz + 0.5 - self.centerOfMass.z);
                final Vector3f r = scene.rotateInto(lx, ly, lz, self.snapQx, self.snapQy, self.snapQz, self.snapQw);
                cxw = self.snapComX + r.x;
                cyw = self.snapComY + r.y;
                czw = self.snapComZ + r.z;
            } else {
                cxw = c.bx + 0.5;
                cyw = c.by + 0.5;
                czw = c.bz + 0.5;
            }
            final double px = cxw + dirX * 0.5;
            final double py = cyw + dirY * 0.5;
            final double pz = czw + dirZ * 0.5;

            // Impact velocity: relative normal speed at the contact point.
            final double[] v = this.velScratch.get();
            velocityAt(sb1, px, py, pz, v, 0);
            velocityAt(sb2, px, py, pz, v, 3);
            final double impact = Math.abs((v[0] - v[3]) * dirX + (v[1] - v[4]) * dirY + (v[2] - v[5]) * dirZ);

            // Hit position in the block's own frame (plot coordinates for sub-levels).
            double hx = px, hy = py, hz = pz;
            if (self != null) {
                final Vector3f r = scene.rotateInto((float) (px - self.snapComX), (float) (py - self.snapComY), (float) (pz - self.snapComZ),
                        -self.snapQx, -self.snapQy, -self.snapQz, self.snapQw);
                hx = self.centerOfMass.x + r.x;
                hy = self.centerOfMass.y + r.y;
                hz = self.centerOfMass.z + r.z;
            }

            int ox = 0, oy = 0, oz = 0;
            final boolean hasOther = other != null;
            if (other != null) {
                if (otherIsTerrain) {
                    // merged terrain boxes: derive the touched block from the point
                    ox = floor(px + dirX * 0.25);
                    oy = floor(py + dirY * 0.25);
                    oz = floor(pz + dirZ * 0.25);
                } else if (other.merged() && other.owner != null && other.owner.kind == SableBody.Kind.SUB_LEVEL) {
                    // the block of a merged box closest to the contact point
                    final SableBody owner = other.owner;
                    final Vector3f r = scene.rotateInto((float) (px + dirX * 0.25 - owner.snapComX), (float) (py + dirY * 0.25 - owner.snapComY),
                            (float) (pz + dirZ * 0.25 - owner.snapComZ), -owner.snapQx, -owner.snapQy, -owner.snapQz, owner.snapQw);
                    ox = other.nearestX(owner.centerOfMass.x + r.x);
                    oy = other.nearestY(owner.centerOfMass.y + r.y);
                    oz = other.nearestZ(owner.centerOfMass.z + r.z);
                } else {
                    ox = other.bx;
                    oy = other.by;
                    oz = other.bz;
                }
            }
            scene.queueCallback(new PendingCallback(entry, key, c.bx, c.by, c.bz, ox, oy, oz, hasOther, hx, hy, hz, impact));
        }

        /**
         * Surface velocity of a mounted contraption block (its motion relative
         * to the mount body), evaluated at the block center.
         */
        private void applyContraptionSurfaceVelocity(@Nullable final Child c, @Nullable final SableBody body,
                                                     final ContactSettings settings, final boolean isFirst) {
            if (c == null || body == null || c.owner == null || c.owner == body || c.owner.kind != SableBody.Kind.CONTRAPTION) {
                return;
            }
            final SableBody contraption = c.owner;
            if (contraption.linVel.lengthSquared() == 0.0 && contraption.angVel.lengthSquared() == 0.0) {
                return;
            }
            // local velocity at the block (center of a merged box): v + w x r (contraption frame)
            final double rx = (c.bx + c.ex) * 0.5 + 0.5 - contraption.centerOfMass.x;
            final double ry = (c.by + c.ey) * 0.5 + 0.5 - contraption.centerOfMass.y;
            final double rz = (c.bz + c.ez) * 0.5 + 0.5 - contraption.centerOfMass.z;
            final Vector3d w = contraption.angVel;
            final double lvx = contraption.linVel.x + w.y * rz - w.z * ry;
            final double lvy = contraption.linVel.y + w.z * rx - w.x * rz;
            final double lvz = contraption.linVel.z + w.x * ry - w.y * rx;
            // contraption frame -> mount block space -> world
            final Quaterniond rel = contraption.relRot;
            final Vector3d mountLocal = new Vector3d(lvx, lvy, lvz);
            rel.transform(mountLocal);
            final Vector3f r = JoltPhysicsScene.this.rotateInto((float) mountLocal.x, (float) mountLocal.y, (float) mountLocal.z,
                    body.snapQx, body.snapQy, body.snapQz, body.snapQw);

            // relative surface velocity: body2 relative to body1
            final float sign = isFirst ? -1.0f : 1.0f;
            final Vec3 existing = settings.getRelativeLinearSurfaceVelocity();
            final Vec3 combined = JoltPhysicsScene.this.tmpVec.get();
            combined.set(existing.getX() + sign * r.x,
                    existing.getY() + sign * r.y,
                    existing.getZ() + sign * r.z);
            settings.setRelativeLinearSurfaceVelocity(combined);
        }

        private final ThreadLocal<double[]> blockScratch = ThreadLocal.withInitial(() -> new double[3]);

        /**
         * The block of a leaf touching the other body: the leaf's own block, or
         * for a merged box the block closest to a point one body radius from the
         * center of mass towards the other body.
         *
         * @param dirX direction from {@code self} towards the other body (world)
         */
        private double[] contactBlock(final Child c, final SableBody self, @Nullable final SableBody other,
                                      final double dirX, final double dirY, final double dirZ) {
            final double[] out = this.blockScratch.get();
            if (!c.merged()) {
                out[0] = c.bx;
                out[1] = c.by;
                out[2] = c.bz;
                return out;
            }
            final JoltPhysicsScene scene = JoltPhysicsScene.this;
            double tx, ty, tz;
            if (other != null && other.body != null) {
                tx = other.snapComX - self.snapComX;
                ty = other.snapComY - self.snapComY;
                tz = other.snapComZ - self.snapComZ;
            } else {
                tx = dirX * self.boundRadius;
                ty = dirY * self.boundRadius;
                tz = dirZ * self.boundRadius;
            }
            final Vector3f r = scene.rotateInto((float) tx, (float) ty, (float) tz, -self.snapQx, -self.snapQy, -self.snapQz, self.snapQw);
            out[0] = c.nearestX(self.centerOfMass.x + r.x);
            out[1] = c.nearestY(self.centerOfMass.y + r.y);
            out[2] = c.nearestZ(self.centerOfMass.z + r.z);
            return out;
        }

        private void report(@Nullable final SableBody sb1, @Nullable final SableBody sb2, @Nullable final Child c1, @Nullable final Child c2,
                            final Body body1, final Body body2, final double nx, final double ny, final double nz) {
            // No collision damage while god-dragging.
            if ((sb1 != null && JoltPhysicsScene.this.creativeDragBodies.contains(sb1.joltId))
                    || (sb2 != null && JoltPhysicsScene.this.creativeDragBodies.contains(sb2.joltId))) {
                return;
            }
            final JoltPhysicsScene scene = JoltPhysicsScene.this;

            // contact point approximated by the involved block center (body-local,
            // relative to the COM); for a merged box, its block facing the other body
            double px1 = 0, py1 = 0, pz1 = 0;
            double px2 = 0, py2 = 0, pz2 = 0;
            double wx1, wy1, wz1;
            if (c1 != null && sb1 != null && c1.owner == sb1) {
                final double[] b = this.contactBlock(c1, sb1, sb2, nx, ny, nz);
                px1 = b[0] + 0.5 - sb1.centerOfMass.x;
                py1 = b[1] + 0.5 - sb1.centerOfMass.y;
                pz1 = b[2] + 0.5 - sb1.centerOfMass.z;
                final Vector3f r = scene.rotateInto((float) px1, (float) py1, (float) pz1, sb1.snapQx, sb1.snapQy, sb1.snapQz, sb1.snapQw);
                wx1 = sb1.snapComX + r.x;
                wy1 = sb1.snapComY + r.y;
                wz1 = sb1.snapComZ + r.z;
            } else if (sb1 != null) {
                wx1 = sb1.snapComX;
                wy1 = sb1.snapComY;
                wz1 = sb1.snapComZ;
            } else {
                return; // body 1 is always the moving one
            }
            if (c2 != null && sb2 != null && c2.owner == sb2) {
                final double[] b = this.contactBlock(c2, sb2, sb1, -nx, -ny, -nz);
                px2 = b[0] + 0.5 - sb2.centerOfMass.x;
                py2 = b[1] + 0.5 - sb2.centerOfMass.y;
                pz2 = b[2] + 0.5 - sb2.centerOfMass.z;
            }

            final double[] v = this.velScratch.get();
            velocityAt(sb1, wx1, wy1, wz1, v, 0);
            velocityAt(sb2, wx1, wy1, wz1, v, 3);
            final double rvx = v[0] - v[3], rvy = v[1] - v[4], rvz = v[2] - v[5];
            // Rapier reports the contact force of the substep (event_handler.rs):
            // the impulse that stops the approach divided by the substep. The
            // pipeline's 25 * minMass threshold then needs a normal speed of
            // 25 * dt (0.625 m/s at 2 substeps); slower contacts are rejected here.
            final double dt = scene.lastTimeStep;
            final double minSpeed = 25.0 * dt;
            if (rvx * rvx + rvy * rvy + rvz * rvz < minSpeed * minSpeed) {
                return;
            }
            final double relSpeed = Math.abs(rvx * nx + rvy * ny + rvz * nz);
            final float invM1 = body1.isDynamic() ? body1.getMotionProperties().getInverseMass() : 0.0f;
            final float invM2 = body2.isDynamic() ? body2.getMotionProperties().getInverseMass() : 0.0f;
            final double effMass = (invM1 + invM2) > 1.0e-9f ? 1.0 / (invM1 + invM2) : 0.0;
            final double force = relSpeed * effMass / dt;
            final double m1 = invM1 > 0.0f ? 1.0 / invM1 : Double.MAX_VALUE;
            final double m2 = invM2 > 0.0f ? 1.0 / invM2 : Double.MAX_VALUE;
            if (!(force > 25.0 * Math.min(m1, m2))) {
                return;
            }
            final int id1 = sb1.kind == SableBody.Kind.SUB_LEVEL ? sb1.runtimeId : -1;
            final int id2 = sb2 != null && sb2.kind == SableBody.Kind.SUB_LEVEL ? sb2.runtimeId : -1;
            final double[] rec = new double[15];
            rec[0] = id1;
            rec[1] = id2;
            rec[2] = force;
            rec[3] = nx;
            rec[4] = ny;
            rec[5] = nz;
            rec[6] = -nx;
            rec[7] = -ny;
            rec[8] = -nz;
            rec[9] = px1;
            rec[10] = py1;
            rec[11] = pz1;
            rec[12] = px2;
            rec[13] = py2;
            rec[14] = pz2;

            scene.reportCollision(rec);
        }
    }
}
