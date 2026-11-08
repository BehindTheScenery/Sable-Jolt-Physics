package dev.behindthescenery.sablejolt;

import org.joml.Quaterniond;
import org.joml.Vector3d;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * <b>HEADLESS TERRAIN LOAD PROBE.</b> Streams hilly terrain with caves into a scene the way a player
 * flying around loads it (a few chunk columns per tick, in random order) and reports the cost of the
 * terrain rebuilds per tick (mean and worst) and in total, against loading everything at once.
 * With {@code bodies=N}, plates fall onto the terrain while it loads (their surroundings get
 * surface meshes).
 * <p>
 * Run with {@code gradle :common:parity -PparityMain=dev.behindthescenery.sablejolt.TerrainLoadProbe}
 * ({@code -PparityArgs="columns=16 perTick=4 bodies=4"}).
 */
public final class TerrainLoadProbe {
    private TerrainLoadProbe() {
    }

    public static void main(final String[] args) {
        int columns = 16;
        int perTick = 4;
        int bodies = 0;
        for (final String arg : args) {
            if (arg.startsWith("columns=")) {
                columns = Integer.parseInt(arg.substring(8));
            } else if (arg.startsWith("perTick=")) {
                perTick = Integer.parseInt(arg.substring(8));
            } else if (arg.startsWith("bodies=")) {
                bodies = Integer.parseInt(arg.substring(7));
            }
        }
        BenchSupport.boot();
        final ParityHarness.Grid terrain = terrain(columns, new Random(1));
        for (int round = 0; round < 2; round++) {
            stream(terrain, columns, perTick, bodies);
            stream(terrain, columns, Integer.MAX_VALUE, bodies);
        }
        System.exit(0);
    }

    /**
     * Rolling hills 4..20 blocks above y = 64 over stone with caves, dirt on top, a few slabs.
     */
    private static ParityHarness.Grid terrain(final int columns, final Random random) {
        final ParityHarness.Grid grid = new ParityHarness.Grid();
        final int size = columns * 16;
        for (int x = 0; x < size; x++) {
            for (int z = 0; z < size; z++) {
                final int top = 64 + 4 + (int) (8 + 8 * Math.sin(x * 0.07) * Math.cos(z * 0.05) + random.nextInt(2));
                for (int y = 0; y < top; y++) {
                    final boolean cave = y > 20 && y < 50 && Math.sin(x * 0.2 + y * 0.3) * Math.cos(z * 0.2) > 0.85;
                    if (!cave) {
                        grid.set(x, y, z, y == top - 1 ? ParityHarness.DIRT : ParityHarness.STONE);
                    }
                }
                if (random.nextInt(40) == 0) {
                    grid.set(x, top, z, ParityHarness.SLAB);
                }
            }
        }
        return grid;
    }

    private static void stream(final ParityHarness.Grid terrain, final int columns, final int perTick, final int bodies) {
        final ParityHarness.Scenario s = new ParityHarness.Scenario("load", "terrain load");
        s.terrain.blocks.putAll(terrain.blocks);
        if (bodies > 0) {
            s.body(9999).fill(0, 0, 0, 0, 0, 0, ParityHarness.PLANKS); // registers the material only
        }
        final JoltPhysicsScene scene = new JoltPhysicsScene(0.0, ParityHarness.GRAVITY, 0.0, ParityHarness.DRAG);
        final Map<ParityHarness.Material, Integer> ids = new IdentityHashMap<>();
        for (final ParityHarness.Material m : ParityHarness.materials(s)) {
            final var data = scene.colliderRegistry.create(m.friction(), m.volume(), m.restitution(), m.fluid(), null, null);
            for (final double[] box : m.boxes()) {
                data.addBox(box[0], box[1], box[2], box[3], box[4], box[5]);
            }
            ids.put(m, data.colliderHandle + 1);
        }
        final var packed = s.terrain.sections(ids, true);
        // plates above the terrain, spread over it (they fall onto it as it loads)
        for (int i = 0; i < bodies; i++) {
            final ParityHarness.BodySpec b = new ParityHarness.BodySpec(i).fill(0, 0, 0, 4, 0, 4, ParityHarness.PLANKS);
            b.computeMass();
            final double bx = (i * 37 % columns) * 16 + 8, bz = (i * 53 % columns) * 16 + 8;
            scene.createSubLevel(i, new Vector3d(bx, 100.0, bz), new Quaterniond());
            scene.setCenterOfMass(i, b.com.x, b.com.y, b.com.z);
            scene.setMassProperties(i, b.mass, b.inertia);
            for (final var e : b.blocks.sections(ids, true).long2ObjectEntrySet()) {
                scene.addChunk(ParityHarness.unpackX(e.getLongKey()), ParityHarness.unpackY(e.getLongKey()), ParityHarness.unpackZ(e.getLongKey()),
                        e.getValue(), false, i);
            }
            scene.setLocalBounds(i, b.blocks.minX, b.blocks.minY, b.blocks.minZ, b.blocks.maxX, b.blocks.maxY, b.blocks.maxZ);
        }
        // stream whole chunk columns in random order
        final Map<Long, List<Long>> byColumn = new HashMap<>();
        for (final long key : packed.keySet()) {
            final long column = ((long) ParityHarness.unpackX(key) << 32) | (ParityHarness.unpackZ(key) & 0xFFFFFFFFL);
            byColumn.computeIfAbsent(column, k -> new ArrayList<>()).add(key);
        }
        final List<Long> order = new ArrayList<>(byColumn.keySet());
        Collections.shuffle(order, new Random(2));
        long total = 0;
        long worst = 0;
        int ticks = 0;
        for (int i = 0; i < order.size(); i += perTick) {
            final long start = System.nanoTime();
            for (int j = i; j < Math.min(order.size(), i + perTick); j++) {
                for (final long key : byColumn.get(order.get(j))) {
                    scene.addChunk(ParityHarness.unpackX(key), ParityHarness.unpackY(key), ParityHarness.unpackZ(key), packed.get(key), true, -1);
                }
            }
            scene.step(1.0 / 40.0);
            final long took = System.nanoTime() - start;
            total += took;
            worst = Math.max(worst, took);
            ticks++;
        }
        final int[] stats = scene.terrainStats();
        System.out.printf(Locale.ROOT, "%s: %d sections in %d ticks: total %.1f ms, mean %.2f ms/tick, worst %.2f ms (%d surface meshes)%n",
                perTick == Integer.MAX_VALUE ? "all at once " : perTick + " columns/tick", packed.size(), ticks, total / 1e6, total / 1e6 / ticks,
                worst / 1e6, stats[2]);
        scene.dispose();
    }
}
