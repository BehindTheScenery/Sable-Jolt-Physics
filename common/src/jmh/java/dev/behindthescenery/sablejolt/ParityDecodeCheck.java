package dev.behindthescenery.sablejolt;

import java.util.Locale;

/**
 * Checks that contact decoding of terrain leaves agrees with Jolt's own sub-shape id composition, for
 * every parity scene and through a sequence of terrain edits (blocks removed under a resting body,
 * sections removed and re-added, all terrain removed).
 * Run with {@code gradle :common:parity -PparityMain=dev.behindthescenery.sablejolt.ParityDecodeCheck}.
 */
public final class ParityDecodeCheck {
    private ParityDecodeCheck() {
    }

    public static void main(final String[] args) {
        int failures = 0;
        for (final ParityHarness.Scenario s : ParityScenarios.all()) {
            for (final ParityHarness.BodySpec b : s.bodies) {
                b.computeMass();
            }
            try (final ParityHarness.JoltSim sim = new ParityHarness.JoltSim(s)) {
                sim.tick();
                final int wrong = sim.scene.verifyTerrainDecoding();
                final int[] stats = sim.scene.terrainStats();
                System.out.printf("%s: %d plain / %d special region bodies, %d surface meshes%n", s.name, stats[0], stats[1], stats[2]);
                failures += wrong;
                if (wrong != 0) {
                    System.out.printf("%s: %d terrain leaves decoded wrongly%n", s.name, wrong);
                }
            }
        }
        failures += editSequence();
        System.out.println(failures == 0 ? "decode check: OK" : "decode check: " + failures + " FAILURES");
        System.exit(failures == 0 ? 0 : 1);
    }

    private static int editSequence() {
        final ParityHarness.Scenario s = ParityScenarios.all().stream().filter(x -> x.name.equals("drop-plate")).findFirst().orElseThrow();
        for (final ParityHarness.BodySpec b : s.bodies) {
            b.computeMass();
        }
        int failures = 0;
        try (final ParityHarness.JoltSim sim = new ParityHarness.JoltSim(s)) {
            final double[] pose = new double[7];
            for (int t = 0; t < 40; t++) {
                sim.tick();
            }
            sim.pose(0, pose);
            System.out.printf(Locale.ROOT, "edit: plate resting at y=%.4f%n", pose[1]);

            // Dig a 7x7 pit under the 5x5 plate: it must drop into it.
            final int stone = ChunkSectionData.colliderIdOf(s.terrain.sections(idsOf(sim, s), true).values().iterator().next()[0]);
            final int n = 49 * 3;
            final int[] xs = new int[n], ys = new int[n], zs = new int[n], packed = new int[n];
            int i = 0;
            for (int dx = -3; dx <= 3; dx++) {
                for (int dz = -3; dz <= 3; dz++) {
                    for (int y = ParityScenarios.GROUND_TOP - 3; y < ParityScenarios.GROUND_TOP; y++) {
                        xs[i] = dx;
                        ys[i] = y;
                        zs[i] = dz;
                        packed[i] = 0;
                        i++;
                    }
                }
            }
            sim.scene.changeBlocks(xs, ys, zs, packed, null, n);
            sim.scene.wakeUpObject(0);
            for (int t = 0; t < 40; t++) {
                sim.tick();
            }
            failures += report(sim, "pit dug");
            sim.pose(0, pose);
            System.out.printf(Locale.ROOT, "edit: plate after the pit y=%.4f (dropped into it: %s)%n", pose[1], pose[1] < 62.0);
            if (pose[1] >= 62.0) {
                failures++;
            }

            // Remove and re-add a terrain section away from the plate.
            final int[] data = new int[ChunkSectionData.BLOCKS];
            java.util.Arrays.fill(data, (stone << 16) | 1);
            sim.scene.removeChunk(2, 3, 2, true);
            sim.tick();
            failures += report(sim, "section removed");
            sim.scene.addChunk(2, 3, 2, data, true, -1);
            sim.tick();
            failures += report(sim, "section re-added");

            // Remove all terrain: region bodies go away and the plate falls.
            for (int cx = -2; cx <= 2; cx++) {
                for (int cz = -2; cz <= 2; cz++) {
                    for (int cy = 2; cy <= 4; cy++) {
                        sim.scene.removeChunk(cx, cy, cz, true);
                    }
                }
            }
            sim.scene.wakeUpObject(0);
            for (int t = 0; t < 20; t++) {
                sim.tick();
            }
            failures += report(sim, "all terrain removed");
            sim.pose(0, pose);
            System.out.printf(Locale.ROOT, "edit: plate after removing the terrain y=%.4f (falling: %s)%n", pose[1], pose[1] < 60.0);
            if (pose[1] >= 60.0) {
                failures++;
            }
        }
        return failures;
    }

    private static java.util.Map<ParityHarness.Material, Integer> idsOf(final ParityHarness.JoltSim sim, final ParityHarness.Scenario s) {
        final java.util.Map<ParityHarness.Material, Integer> ids = new java.util.IdentityHashMap<>();
        int next = 1;
        for (final ParityHarness.Material m : ParityHarness.materials(s)) {
            ids.put(m, next++);
        }
        return ids;
    }

    private static int report(final ParityHarness.JoltSim sim, final String what) {
        final int wrong = sim.scene.verifyTerrainDecoding();
        System.out.printf("edit: %s, %d leaves decoded wrongly%n", what, wrong);
        return wrong;
    }
}
