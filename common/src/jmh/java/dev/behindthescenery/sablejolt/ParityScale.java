package dev.behindthescenery.sablejolt;

import java.util.Locale;

/**
 * <b>HEADLESS SCALE COMPARISON.</b> Many sub-levels sliding over mixed terrain at once, stepped by
 * Sable's Rapier natives and by {@link JoltPhysicsScene}: wall time per game tick of each engine.
 * <p>
 * Run with {@code gradle :common:parity -PparityMain=dev.behindthescenery.sablejolt.ParityScale}
 * (optionally {@code -PparityArgs="bodies=400"}).
 */
public final class ParityScale {
    private ParityScale() {
    }

    public static void main(final String[] args) {
        int bodies = 200;
        int size = 5;
        for (final String arg : args) {
            if (arg.startsWith("bodies=")) {
                bodies = Integer.parseInt(arg.substring(7));
            } else if (arg.startsWith("size=")) {
                size = Integer.parseInt(arg.substring(5));
            } else if (arg.indexOf('=') > 0) {
                System.setProperty(arg.substring(0, arg.indexOf('=')), arg.substring(arg.indexOf('=') + 1));
            }
        }
        final ParityHarness.Scenario s = new ParityHarness.Scenario("scale", bodies + " sliding bodies of " + size + "x2x" + size);
        ParityScenarios.ground(s.terrain, -160, -160, 160, 160, ParityHarness.STONE, ParityHarness.DIRT);
        final int spacing = size + 10;
        final int perRow = Math.max(1, 300 / spacing);
        for (int i = 0; i < bodies; i++) {
            final ParityHarness.BodySpec b = s.body(i)
                    .fill(0, 0, 0, size - 1, 0, size - 1, ParityHarness.PLANKS)
                    .fill(0, 1, 0, size - 1, 1, 0, ParityHarness.PLANKS);
            ParityScenarios.placeAbove(b, -150 + (i % perRow) * spacing, -150 + (i / perRow % perRow) * spacing, ParityScenarios.GROUND_TOP, 0.0);
        }
        for (final ParityHarness.BodySpec b : s.bodies) {
            b.computeMass();
        }
        for (int round = 0; round < 2; round++) {
            try (final ParityHarness.RapierSim sim = new ParityHarness.RapierSim(s)) {
                report("rapier", sim, bodies);
            }
            try (final ParityHarness.JoltSim sim = new ParityHarness.JoltSim(s)) {
                report("jolt", sim, bodies);
            }
        }
        System.exit(0);
    }

    /**
     * Keeps every body sliding (re-applies its velocity each tick) and times the ticks.
     */
    private static void report(final String engine, final ParityHarness.Sim sim, final int bodies) {
        final double[] v = new double[3];
        for (int t = 0; t < 120; t++) {
            if (t == 20) {
                final long start = System.nanoTime();
                for (int m = 0; m < 100; m++) {
                    push(sim, bodies, v);
                    sim.tick();
                }
                System.out.printf(Locale.ROOT, "%s: %.2f ms/tick with %d sliding bodies%n", engine, (System.nanoTime() - start) / 1e6 / 100, bodies);
                return;
            }
            push(sim, bodies, v);
            sim.tick();
        }
    }

    private static void push(final ParityHarness.Sim sim, final int bodies, final double[] v) {
        for (int i = 0; i < bodies; i++) {
            sim.linearVelocity(i, v);
            sim.addVelocity(i, 3.0 - v[0], 0.0, 1.5 - v[2], 0.0, 0.0, 0.0);
        }
    }
}
