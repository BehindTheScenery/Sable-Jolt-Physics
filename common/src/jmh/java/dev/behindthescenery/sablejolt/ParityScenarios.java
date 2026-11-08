package dev.behindthescenery.sablejolt;

import dev.behindthescenery.sablejolt.ParityHarness.BodySpec;
import dev.behindthescenery.sablejolt.ParityHarness.Grid;
import dev.behindthescenery.sablejolt.ParityHarness.Material;
import dev.behindthescenery.sablejolt.ParityHarness.Run;
import dev.behindthescenery.sablejolt.ParityHarness.Scenario;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static dev.behindthescenery.sablejolt.ParityHarness.BOUNCY;
import static dev.behindthescenery.sablejolt.ParityHarness.DIRT;
import static dev.behindthescenery.sablejolt.ParityHarness.ICE;
import static dev.behindthescenery.sablejolt.ParityHarness.PLANKS;
import static dev.behindthescenery.sablejolt.ParityHarness.SLAB;
import static dev.behindthescenery.sablejolt.ParityHarness.STONE;
import static dev.behindthescenery.sablejolt.ParityHarness.TOP_SLAB;
import static dev.behindthescenery.sablejolt.ParityHarness.WATER;
import static dev.behindthescenery.sablejolt.ParityHarness.speed;
import static dev.behindthescenery.sablejolt.ParityHarness.spin;
import static dev.behindthescenery.sablejolt.ParityHarness.tilt;

/**
 * The scenes compared by {@link ParityHarness}. Ground tops are at y = 64 unless noted.
 */
final class ParityScenarios {
    static final int GROUND_TOP = 64;

    private ParityScenarios() {
    }

    /** Flat ground three blocks deep spanning several sections, top face at y = 64. */
    static void ground(final Grid g, final int x0, final int z0, final int x1, final int z1, final Material... pattern) {
        for (int x = x0; x <= x1; x++) {
            for (int z = z0; z <= z1; z++) {
                final Material m = pattern[Math.floorMod(x + z, pattern.length)];
                g.fill(x, GROUND_TOP - 3, z, x, GROUND_TOP - 1, z, m);
            }
        }
    }

    /** Places the body level, its lowest block face {@code gap} above {@code top}. */
    static void placeAbove(final BodySpec b, final double x, final double z, final double top, final double gap) {
        b.computeMass();
        b.position.set(x, b.restingHeight(top) + gap, z);
    }

    static String fmt(final String format, final Object... args) {
        return String.format(Locale.ROOT, format, args);
    }

    /** Rest error (+ = floating, - = sunk), settle tick, deepest dip and tilt. */
    static String restMetrics(final Scenario s, final Run r, final double top) {
        final BodySpec b = s.bodies.get(0);
        final double ideal = b.restingHeight(top);
        final double[] last = r.last(0);
        return fmt("rest err %+.4f m, deepest %+.4f m, settled @%d, tilt %.2f deg, still %.3f m/s",
                last[1] - ideal, r.min(0, a -> a[1]) - ideal, r.settleTick(0, 0.02), tilt(last), speed(last) + spin(last));
    }

    static List<Scenario> all() {
        final List<Scenario> list = new ArrayList<>();

        // --- Resting and impacts ---------------------------------------------------------------

        {
            final Scenario s = new Scenario("drop-cube", "single block falls 3 m onto flat stone");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 0, 0, 0, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 3.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> restMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("drop-plate", "5x1x5 plate falls 2 m flat onto stone");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 2.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> restMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("drop-heavy", "9x3x9 solid block of stone falls 1 m (729 blocks)");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 8, 2, 8, STONE);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 1.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> restMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("tilted-drop", "3x3x3 cube, tilted 30/20 deg, falls 4 m");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 2, 2, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 4.0);
            b.orientation.rotateZ(Math.toRadians(30)).rotateX(Math.toRadians(20));
            s.ticks = 200;
            s.metrics = (sc, r) -> {
                final double[] last = r.last(0);
                // A cube settles on a face: tilt is a multiple of 90 degrees.
                final double t = tilt(last);
                final double offFace = Math.abs(t - 90.0 * Math.round(t / 90.0));
                return fmt("settled @%d, off-face %.2f deg, y %.3f (ideal %.3f), still %.3f",
                        r.settleTick(0, 0.02), offFace, last[1], sc.bodies.get(0).restingHeight(GROUND_TOP), speed(last) + spin(last));
            };
            list.add(s);
        }
        {
            final Scenario s = new Scenario("fast-fall", "single block falls 60 m (~36 m/s impact)");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 0, 0, 0, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 60.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> restMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("bouncy-drop", "block of bouncy material falls 3 m");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 0, 0, 0, BOUNCY);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 3.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> {
                final double ideal = sc.bodies.get(0).restingHeight(GROUND_TOP);
                // apex after the first impact
                int impact = -1;
                for (int t = 1; t <= r.ticks; t++) {
                    if (r.at(t - 1, 0)[8] < -1.0 && r.at(t, 0)[8] > r.at(t - 1, 0)[8] + 1.0) {
                        impact = t;
                        break;
                    }
                }
                double apex = Double.NaN;
                if (impact > 0) {
                    apex = Double.NEGATIVE_INFINITY;
                    for (int t = impact; t <= Math.min(r.ticks, impact + 30); t++) {
                        apex = Math.max(apex, r.at(t, 0)[1] - ideal);
                    }
                }
                return fmt("first impact @%d, bounce apex %.3f m, %s", impact, apex, restMetrics(sc, r, GROUND_TOP));
            };
            list.add(s);
        }
        {
            final Scenario s = new Scenario("stack", "3x1x3 plate dropped 1 m onto another resting plate");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec bottom = s.body(0).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(bottom, 0.5, 0.5, GROUND_TOP, 0.0);
            final BodySpec top = s.body(1).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(top, 0.5, 0.5, GROUND_TOP + 1, 1.0);
            s.ticks = 160;
            s.metrics = (sc, r) -> fmt("bottom err %+.4f, top err %+.4f, top settled @%d, top tilt %.2f",
                    r.last(0)[1] - sc.bodies.get(0).restingHeight(GROUND_TOP),
                    r.last(1)[1] - sc.bodies.get(1).restingHeight(GROUND_TOP + 1),
                    r.settleTick(1, 0.02), tilt(r.last(1)));
            list.add(s);
        }

        // --- Sliding and friction --------------------------------------------------------------

        list.add(slide("slide-plate", "5x1x5 plate slides at 6 m/s over uniform stone", 4, 0, 4, 6.0, STONE));
        list.add(slide("slide-long", "5x1x5 plate slides at 12 m/s over stone across a section boundary", 4, 0, 4, 12.0, STONE));
        {
            // Jolt terrain regions end at 512 + 1024 * k: the one place left where
            // two terrain bodies meet.
            final Scenario s = new Scenario("slide-region", "5x1x5 plate slides at 12 m/s across a terrain region boundary (x = 512)");
            ground(s.terrain, 470, -24, 560, 39, STONE, DIRT);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, 505.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(12.0, 0.0, 0.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = slide("slide-fast", "single block slides at 25 m/s over checkered stone/dirt (CCD speeds)", 0, 0, 0, 25.0, STONE, DIRT);
            s.terrain.blocks.clear();
            ground(s.terrain, -24, -24, 159, 39, STONE, DIRT);
            s.ticks = 140;
            list.add(s);
        }
        list.add(slide("slide-seams", "5x1x5 plate slides at 6 m/s over checkered stone/dirt (seams)", 4, 0, 4, 6.0, STONE, DIRT));
        list.add(slide("slide-cube", "single block slides at 6 m/s over checkered stone/dirt", 0, 0, 0, 6.0, STONE, DIRT));
        list.add(slide("slide-box", "3x3x3 cube slides at 5 m/s over checkered stone/dirt", 2, 2, 2, 5.0, STONE, DIRT));
        list.add(slide("slide-ice", "5x1x5 plate slides at 4 m/s over slippery blocks", 4, 0, 4, 4.0, ICE));
        {
            // Flat path; holes, slabs and slippery blocks right beside it split
            // Jolt's merged ground boxes into irregular pieces next to the path.
            final Scenario s = new Scenario("slide-rough", "5x1x3 plate slides at 8 m/s along a flat path between irregular blocks");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            final java.util.Random random = new java.util.Random(7);
            for (int x = -24; x <= 63; x++) {
                for (final int z : new int[]{-3, -2, 3, 4}) {
                    final int pick = random.nextInt(6);
                    if (pick == 0) {
                        s.terrain.set(x, GROUND_TOP - 1, z, null);
                    } else if (pick == 1) {
                        s.terrain.set(x, GROUND_TOP - 1, z, SLAB);
                    } else if (pick == 2) {
                        s.terrain.set(x, GROUND_TOP - 1, z, ICE);
                    }
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 2, PLANKS);
            placeAbove(b, -10.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(8.0, 0.0, 0.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            // Sliding across the irregular rows instead of along them.
            final Scenario s = new Scenario("slide-rough-cross", "3x1x3 plate slides at 6 m/s across rows of irregular blocks");
            ground(s.terrain, -24, -24, 39, 63, STONE, DIRT);
            final java.util.Random random = new java.util.Random(11);
            for (int z = -24; z <= 63; z++) {
                if (Math.floorMod(z, 3) != 0) {
                    continue;
                }
                for (int x = -24; x <= 39; x++) {
                    if (x >= -2 && x <= 2) {
                        continue; // the plate's path stays flat
                    }
                    final int pick = random.nextInt(5);
                    if (pick == 0) {
                        s.terrain.set(x, GROUND_TOP - 1, z, null);
                    } else if (pick == 1) {
                        s.terrain.set(x, GROUND_TOP - 1, z, ICE);
                    }
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(b, 0.5, -10.0, GROUND_TOP, 0.0);
            b.linearVelocity.set(0.0, 0.0, 6.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> fmt("distance %.3f m, max |vy| %.3f, max spin %.3f, max lift %+.4f, final tilt %.2f, lateral %.3f",
                    r.last(0)[2] - r.at(0, 0)[2], Math.max(r.max(0, a -> a[8]), -r.min(0, a -> a[8])), r.max(0, ParityHarness::spin),
                    r.max(0, a -> a[1]) - sc.bodies.get(0).restingHeight(GROUND_TOP), tilt(r.last(0)), r.last(0)[0] - r.at(0, 0)[0]);
            list.add(s);
        }
        {
            // A small body sliding over a big resting deck sub-level (sub-level on sub-level).
            final Scenario s = new Scenario("deck-slide", "single block slides at 6 m/s across a 15x1x7 deck lying on stone");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec deck = s.body(0).fill(0, 0, 0, 14, 0, 6, STONE);
            placeAbove(deck, 0.5, 0.5, GROUND_TOP, 0.0);
            final BodySpec cube = s.body(1).fill(0, 0, 0, 0, 0, 0, PLANKS);
            placeAbove(cube, -5.5, 0.5, GROUND_TOP + 1, 0.0);
            cube.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("cube distance %.3f m, max |vy| %.3f, max spin %.3f, cube lift %+.4f, deck moved %.4f m, cube tilt %.2f",
                    r.last(1)[0] - r.at(0, 1)[0], Math.max(r.max(1, a -> a[8]), -r.min(1, a -> a[8])), r.max(1, ParityHarness::spin),
                    r.max(1, a -> a[1]) - sc.bodies.get(1).restingHeight(GROUND_TOP + 1),
                    Math.hypot(r.last(0)[0] - r.at(0, 0)[0], r.last(0)[2] - r.at(0, 0)[2]), tilt(r.last(1)));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("slide-slabs", "5x1x5 plate slides at 5 m/s over a floor of slabs");
            for (int x = -24; x <= 39; x++) {
                for (int z = -24; z <= 39; z++) {
                    s.terrain.fill(x, GROUND_TOP - 3, z, x, GROUND_TOP - 2, z, STONE);
                    s.terrain.set(x, GROUND_TOP - 1, z, SLAB);
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -10.0, 0.5, GROUND_TOP - 0.5, 0.0);
            b.linearVelocity.set(5.0, 0.0, 0.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP - 0.5);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("spin", "3x1x3 plate spinning at 6 rad/s on stone");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 0.0);
            b.angularVelocity.set(0.0, 6.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> {
                int stop = -1;
                for (int t = 0; t <= r.ticks; t++) {
                    if (spin(r.at(t, 0)) < 0.1) {
                        stop = t;
                        break;
                    }
                }
                return fmt("spin @10 %.2f, @20 %.2f rad/s, stops @%d, drift %.3f m, tilt %.2f",
                        spin(r.at(10, 0)), spin(r.at(20, 0)), stop,
                        Math.hypot(r.last(0)[0] - r.at(0, 0)[0], r.last(0)[2] - r.at(0, 0)[2]), tilt(r.last(0)));
            };
            list.add(s);
        }

        // --- Walls, ledges, bodies -------------------------------------------------------------

        {
            final Scenario s = new Scenario("wall-hit", "2x2x2 cube moving at 10 m/s hits a 3-high wall");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            s.terrain.fill(8, GROUND_TOP, -6, 9, GROUND_TOP + 2, 6, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 1, 1, 1, PLANKS);
            placeAbove(b, 0.0, 0.5, GROUND_TOP, 0.02);
            b.linearVelocity.set(10.0, 0.0, 0.0);
            s.ticks = 80;
            s.metrics = (sc, r) -> fmt("max x %.3f (wall face 8.0, body half 1.0 -> 7.000), final x %.3f, rebound vx %.2f, tilt %.2f",
                    r.max(0, a -> a[0]), r.last(0)[0], r.min(0, a -> a[7]), tilt(r.last(0)));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("ledge", "5x1x5 plate overhanging a 4 m cliff slides off at 4 m/s");
            ground(s.terrain, -24, -24, 9, 39, STONE);
            for (int x = 10; x <= 39; x++) {
                for (int z = -24; z <= 39; z++) {
                    s.terrain.fill(x, GROUND_TOP - 7, z, x, GROUND_TOP - 5, z, STONE);
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, 9.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(4.0, 0.0, 0.0);
            s.ticks = 160;
            s.metrics = (sc, r) -> fmt("final (%.3f, %.3f), max tilt %.1f, final tilt %.1f, settled @%d",
                    r.last(0)[0], r.last(0)[1], r.max(0, ParityHarness::tilt), tilt(r.last(0)), r.settleTick(0, 0.02));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("bodies-collide", "two 2x2x2 cubes thrown at each other at 6 m/s");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec a = s.body(0).fill(0, 0, 0, 1, 1, 1, PLANKS);
            placeAbove(a, -2.5, 0.5, GROUND_TOP, 0.02);
            a.linearVelocity.set(6.0, 0.0, 0.0);
            final BodySpec c = s.body(1).fill(0, 0, 0, 1, 1, 1, STONE);
            placeAbove(c, 2.5, 0.5, GROUND_TOP, 0.02);
            c.linearVelocity.set(-6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("final x %.3f / %.3f (gap %.3f), min gap %.3f, tilts %.1f / %.1f",
                    r.last(0)[0], r.last(1)[0], r.last(1)[0] - r.last(0)[0] - 2.0,
                    minGap(r), tilt(r.last(0)), tilt(r.last(1)));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("pillar", "1x5x1 pillar standing upright, pushed at the top");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 0, 4, 0, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 0.0);
            s.ticks = 120;
            s.action = (sim, t) -> {
                if (t == 5) {
                    sim.applyImpulse(0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0);
                }
            };
            s.metrics = (sc, r) -> fmt("final tilt %.1f, final (%.3f, %.3f), settled @%d",
                    tilt(r.last(0)), r.last(0)[0], r.last(0)[1], r.settleTick(0, 0.02));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("l-shape", "L-shaped body (overhanging arm) dropped 1 m");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 1, PLANKS).fill(0, 1, 0, 0, 3, 1, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP, 1.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> restMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("water-float", "5x1x5 plate dropped into a 4-deep pool");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            s.terrain.fill(-8, GROUND_TOP, -8, 8, GROUND_TOP + 3, 8, WATER);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP + 4, 1.0);
            s.ticks = 200;
            s.metrics = (sc, r) -> fmt("final y %.3f (surface %d, bottom at %.3f), min y %.3f, tilt %.2f, still %.3f",
                    r.last(0)[1], GROUND_TOP + 4, r.last(0)[1] - (sc.bodies.get(0).com.y - sc.bodies.get(0).blocks.minY),
                    r.min(0, a -> a[1]), tilt(r.last(0)), speed(r.last(0)) + spin(r.last(0)));
            list.add(s);
        }
        lines(list);
        return list;
    }

    /**
     * Bodies meeting lines of blocks: curbs, walls, rails, trenches, steps, regular patterns of
     * separate blocks, and lines of special or partial blocks in a flat floor.
     */
    static void lines(final List<Scenario> list) {
        {
            final Scenario s = new Scenario("curb-hit", "5x1x5 plate slides at 6 m/s into a 1-high curb");
            ground(s.terrain, -24, -24, 39, 39, STONE, DIRT);
            s.terrain.fill(0, GROUND_TOP, -8, 0, GROUND_TOP, 8, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -4.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 80;
            s.metrics = (sc, r) -> fmt("max x %.3f (curb face 0.0, half 2.5 -> -2.500), final x %.3f, rebound vx %.2f, %s",
                    r.max(0, a -> a[0]), r.last(0)[0], r.min(0, a -> a[7]), motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("curb-hit-box", "3x3x3 cube slides at 6 m/s into a 1-high curb");
            ground(s.terrain, -24, -24, 39, 39, STONE, DIRT);
            s.terrain.fill(0, GROUND_TOP, -8, 0, GROUND_TOP, 8, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 2, 2, PLANKS);
            placeAbove(b, -3.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("max x %.3f, final (%.3f, %.3f), max tilt %.1f, %s",
                    r.max(0, a -> a[0]), r.last(0)[0], r.last(0)[1], r.max(0, ParityHarness::tilt), motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("curb-glance", "3x1x3 plate hits a curb at 30 degrees and slides along it");
            ground(s.terrain, -24, -24, 39, 39, STONE, DIRT);
            s.terrain.fill(-24, GROUND_TOP, 4, 39, GROUND_TOP, 4, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(b, -10.0, 2.0, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0 * Math.cos(Math.toRadians(30)), 0.0, 6.0 * Math.sin(Math.toRadians(30)));
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("final (%.3f, %.3f), max z %.3f (curb face 4.0, half 1.5 -> 2.500), yaw %.1f, %s",
                    r.last(0)[0], r.last(0)[2], r.max(0, a -> a[2]), yaw(r.last(0)), motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("wall-slide", "3x2x3 box slides at 6 m/s along a 2-high wall, pressed into it");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            for (int x = -24; x <= 63; x++) {
                for (int y = GROUND_TOP; y <= GROUND_TOP + 1; y++) {
                    s.terrain.set(x, y, 3, ((x + y) & 1) == 0 ? STONE : DIRT);
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 1, 2, PLANKS);
            placeAbove(b, -10.0, 1.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.5);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("distance %.3f m, final z %.3f (wall 3.0 -> 1.500), yaw %.1f, %s",
                    r.last(0)[0] - r.at(0, 0)[0], r.last(0)[2], yaw(r.last(0)), motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("wall-slide-rough", "3x2x3 box slides along a wall with random slippery/missing blocks");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            final java.util.Random random = new java.util.Random(5);
            for (int x = -24; x <= 63; x++) {
                for (int y = GROUND_TOP; y <= GROUND_TOP + 1; y++) {
                    final int pick = random.nextInt(4);
                    s.terrain.set(x, y, 3, pick == 0 ? ICE : STONE);
                }
                if (random.nextInt(5) == 0) {
                    s.terrain.set(x, GROUND_TOP + 1, 3, null);
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 1, 2, PLANKS);
            placeAbove(b, -10.0, 1.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.5);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("distance %.3f m, final z %.3f (wall 3.0 -> 1.500), yaw %.1f, %s",
                    r.last(0)[0] - r.at(0, 0)[0], r.last(0)[2], yaw(r.last(0)), motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = slide("beam-cross", "1x1x7 beam slides sideways at 6 m/s over checkered stone/dirt", 0, 0, 6, 6.0, STONE, DIRT);
            list.add(s);
        }
        {
            final Scenario s = slide("beam-along", "9x1x1 beam slides lengthwise at 6 m/s over checkered stone/dirt", 8, 0, 0, 6.0, STONE, DIRT);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("rail-slide", "5x1x5 plate slides at 6 m/s on two 1-high rails");
            ground(s.terrain, -24, -24, 63, 39, STONE);
            for (int x = -24; x <= 63; x++) {
                s.terrain.set(x, GROUND_TOP, -2, (x & 1) == 0 ? STONE : DIRT);
                s.terrain.set(x, GROUND_TOP, 2, (x & 1) == 0 ? DIRT : STONE);
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -10.0, 0.5, GROUND_TOP + 1, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP + 1);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("trench-cross", "5x1x5 plate slides at 6 m/s across a 1-wide trench");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            s.terrain.fill(-5, GROUND_TOP - 1, -24, -5, GROUND_TOP - 1, 39, null);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -8.5, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 80;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("trench-cross-cube", "single block slides at 6 m/s across a 1-wide, 1-deep trench");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            s.terrain.fill(-5, GROUND_TOP - 1, -24, -5, GROUND_TOP - 1, 39, null);
            final BodySpec b = s.body(0).fill(0, 0, 0, 0, 0, 0, PLANKS);
            placeAbove(b, -7.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 80;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("comb-slide", "7x1x7 plate slides at 6 m/s over a checkerboard of raised blocks");
            ground(s.terrain, -24, -24, 63, 39, STONE);
            for (int x = -24; x <= 63; x++) {
                for (int z = -10; z <= 10; z++) {
                    if (((x + z) & 1) == 0) {
                        s.terrain.set(x, GROUND_TOP, z, STONE);
                    }
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 6, 0, 6, PLANKS);
            placeAbove(b, -10.5, 0.5, GROUND_TOP + 1, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP + 1);
            list.add(s);
        }
        {
            // Same comb, the plate starting half a block further and dropped onto it.
            final Scenario s = new Scenario("comb-slide-2", "7x1x7 plate dropped onto a checkerboard of raised blocks, sliding at 6 m/s");
            ground(s.terrain, -24, -24, 63, 39, STONE);
            for (int x = -24; x <= 63; x++) {
                for (int z = -10; z <= 10; z++) {
                    if (((x + z) & 1) == 0) {
                        s.terrain.set(x, GROUND_TOP, z, STONE);
                    }
                }
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 6, 0, 6, PLANKS);
            placeAbove(b, -10.0, 0.5, GROUND_TOP + 1, 0.05);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP + 1);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("slide-diagonal", "5x1x5 plate turned 45 degrees slides at 6 m/s over checkered stone/dirt");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -10.0, 0.5, GROUND_TOP, 0.0);
            b.orientation.rotateY(Math.toRadians(45));
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
        // Block wheels: upright discs rolling along x (7x7 at 4 m/s, then other sizes and speeds).
        list.add(wheel("wheel-roll", 7, 10.0, 4.0, 120));
        list.add(wheel("wheel-roll-slow", 7, 10.0, 2.5, 120));
        list.add(wheel("wheel-roll-fast", 7, 10.0, 7.0, 120));
        list.add(wheel("wheel-roll-big", 11, 26.0, 6.0, 140));
        list.add(wheel("wheel-roll-small", 5, 5.0, 3.0, 120));
        {
            final Scenario s = new Scenario("stripe-slide", "5x1x5 plate slides at 6 m/s over raised lines every other block");
            ground(s.terrain, -24, -24, 63, 39, STONE);
            for (int x = -24; x <= 63; x += 2) {
                s.terrain.fill(x, GROUND_TOP, -10, x, GROUND_TOP, 10, (x & 2) == 0 ? STONE : DIRT);
            }
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -9.5, 0.5, GROUND_TOP + 1, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP + 1);
            list.add(s);
        }
        {
            final Scenario s = new Scenario("step-down", "5x1x5 plate slides at 6 m/s off a 1-high step");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            s.terrain.fill(-24, GROUND_TOP, -24, -1, GROUND_TOP, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, -4.0, 0.5, GROUND_TOP + 1, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("final (%.3f, %.3f), max tilt %.1f, final tilt %.2f, settled @%d, %s",
                    r.last(0)[0], r.last(0)[1], r.max(0, ParityHarness::tilt), tilt(r.last(0)), r.settleTick(0, 0.02),
                    motion(sc, r, 0, GROUND_TOP));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("step-up-box", "3x3x3 cube slides at 7 m/s into a 1-high step");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            s.terrain.fill(0, GROUND_TOP, -24, 63, GROUND_TOP, 39, STONE);
            final BodySpec b = s.body(0).fill(0, 0, 0, 2, 2, 2, PLANKS);
            placeAbove(b, -3.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(7.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("final (%.3f, %.3f), max tilt %.1f, final tilt %.1f, settled @%d",
                    r.last(0)[0], r.last(0)[1], r.max(0, ParityHarness::tilt), tilt(r.last(0)), r.settleTick(0, 0.02));
            list.add(s);
        }
        {
            final Scenario s = new Scenario("drop-on-rail", "5x1x5 plate dropped 1 m onto a single 1-high line of blocks");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            s.terrain.fill(0, GROUND_TOP, -8, 0, GROUND_TOP, 8, DIRT);
            final BodySpec b = s.body(0).fill(0, 0, 0, 4, 0, 4, PLANKS);
            placeAbove(b, 0.5, 0.5, GROUND_TOP + 1, 1.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> fmt("final (%.3f, %.3f), final tilt %.2f, max tilt %.2f, settled @%d",
                    r.last(0)[0], r.last(0)[1], tilt(r.last(0)), r.max(0, ParityHarness::tilt), r.settleTick(0, 0.02));
            list.add(s);
        }
        {
            // A line of blocks that is itself a sub-level.
            final Scenario s = new Scenario("beam-hit", "3x1x3 plate slides at 6 m/s into a 1x1x9 beam lying on the ground");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec plate = s.body(0).fill(0, 0, 0, 2, 0, 2, PLANKS);
            placeAbove(plate, -4.0, 0.5, GROUND_TOP, 0.0);
            plate.linearVelocity.set(6.0, 0.0, 0.0);
            final BodySpec beam = s.body(1).fill(0, 0, 0, 0, 0, 8, PLANKS);
            placeAbove(beam, 0.5, 0.5, GROUND_TOP, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("plate final x %.3f, beam final (%.3f, %.3f) yaw %.1f, beam lift %+.4f, plate lift %+.4f",
                    r.last(0)[0], r.last(1)[0], r.last(1)[2], yaw(r.last(1)),
                    r.max(1, a -> a[1]) - sc.bodies.get(1).restingHeight(GROUND_TOP),
                    r.max(0, a -> a[1]) - sc.bodies.get(0).restingHeight(GROUND_TOP));
            list.add(s);
        }
        {
            // Lines of special blocks across the path (merged ground splits around them).
            final Scenario s = slide("ice-lines", "5x1x5 plate slides at 6 m/s across full lines of slippery blocks", 4, 0, 4, 6.0, STONE, DIRT);
            for (int x = -24; x <= 63; x += 4) {
                s.terrain.fill(x, GROUND_TOP - 1, -24, x, GROUND_TOP - 1, 39, ICE);
            }
            list.add(s);
        }
        {
            final Scenario s = slide("ice-dashes", "5x1x5 plate slides at 6 m/s across dashed lines of slippery blocks", 4, 0, 4, 6.0, STONE, DIRT);
            for (int x = -24; x <= 63; x += 3) {
                for (int z = -24; z <= 39; z++) {
                    if (Math.floorMod(z, 3) != 2) {
                        s.terrain.set(x, GROUND_TOP - 1, z, ICE);
                    }
                }
            }
            list.add(s);
        }
        {
            final Scenario s = slide("slab-lines", "5x1x5 plate slides at 6 m/s across lines of slabs sunk into the floor", 4, 0, 4, 6.0, STONE, DIRT);
            for (int x = -24; x <= 63; x += 4) {
                s.terrain.fill(x, GROUND_TOP - 1, -24, x, GROUND_TOP - 1, 39, SLAB);
            }
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid", "5x1x5 plate slides at 6 m/s over a floor with a regular grid of holes", 4, 0, 4, 6.0, STONE, DIRT);
            holeGrid(s);
            list.add(s);
        }

        // --- Irregular floors: more shapes, speeds and directions -----------------------------------
        {
            final Scenario s = slide("hole-grid-fast", "5x1x5 plate slides at 14 m/s over a floor with a grid of holes", 4, 0, 4, 14.0, STONE, DIRT);
            s.terrain.blocks.clear();
            ground(s.terrain, -24, -24, 95, 39, STONE, DIRT);
            holeGrid(s);
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid-big", "9x1x9 plate slides at 6 m/s over a floor with a grid of holes", 8, 0, 8, 6.0, STONE, DIRT);
            holeGrid(s);
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid-small", "2x1x2 plate slides at 6 m/s over a floor with a grid of holes", 1, 0, 1, 6.0, STONE, DIRT);
            holeGrid(s);
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid-box", "3x3x3 cube slides at 6 m/s over a floor with a grid of holes", 2, 2, 2, 6.0, STONE, DIRT);
            holeGrid(s);
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid-diag", "5x1x5 plate turned 30 degrees slides at 6 m/s over a grid of holes", 4, 0, 4, 6.0, STONE, DIRT);
            holeGrid(s);
            s.bodies.get(0).orientation.rotateY(Math.toRadians(30));
            list.add(s);
        }
        {
            final Scenario s = slide("hole-grid-cross", "5x1x5 plate slides at 6 m/s diagonally (x and z) over a grid of holes", 4, 0, 4, 6.0, STONE, DIRT);
            holeGrid(s);
            s.bodies.get(0).linearVelocity.set(6.0 * Math.cos(Math.toRadians(35)), 0.0, 6.0 * Math.sin(Math.toRadians(35)));
            list.add(s);
        }
        {
            final Scenario s = slide("random-floor", "5x1x5 plate slides at 8 m/s over a floor of random holes, slabs and slippery blocks", 4, 0, 4, 8.0, STONE, DIRT);
            final java.util.Random random = new java.util.Random(21);
            for (int x = -24; x <= 63; x++) {
                for (int z = -24; z <= 39; z++) {
                    final int pick = random.nextInt(12);
                    if (pick == 0) {
                        s.terrain.set(x, GROUND_TOP - 1, z, null);
                    } else if (pick == 1) {
                        s.terrain.set(x, GROUND_TOP - 1, z, TOP_SLAB);
                    } else if (pick == 2) {
                        s.terrain.set(x, GROUND_TOP - 1, z, ICE);
                    }
                }
            }
            list.add(s);
        }
        {
            final Scenario s = slide("top-slab-floor", "5x1x5 plate slides at 6 m/s over stone with random flush top slabs", 4, 0, 4, 6.0, STONE, DIRT);
            final java.util.Random random = new java.util.Random(3);
            for (int x = -24; x <= 63; x++) {
                for (int z = -24; z <= 39; z++) {
                    if (random.nextInt(4) == 0) {
                        s.terrain.set(x, GROUND_TOP - 1, z, TOP_SLAB);
                    }
                }
            }
            list.add(s);
        }
        {
            // Sub-level on sub-level: a cube sliding over a deck with holes and slippery blocks.
            final Scenario s = new Scenario("deck-rough", "2x1x2 plate slides at 6 m/s across a 21x1x9 deck with holes and slippery blocks");
            ground(s.terrain, -24, -24, 39, 39, STONE);
            final BodySpec deck = s.body(0).fill(0, 0, 0, 20, 0, 8, STONE);
            final java.util.Random random = new java.util.Random(9);
            for (int x = 1; x < 20; x++) {
                for (int z = 0; z <= 8; z++) {
                    if (z >= 3 && z <= 5) {
                        continue; // the plate's lane stays flat
                    }
                    final int pick = random.nextInt(5);
                    if (pick == 0) {
                        deck.set(x, 0, z, null);
                    } else if (pick == 1) {
                        deck.set(x, 0, z, ICE);
                    }
                }
            }
            placeAbove(deck, 0.5, 0.5, GROUND_TOP, 0.0);
            final BodySpec plate = s.body(1).fill(0, 0, 0, 1, 0, 1, PLANKS);
            placeAbove(plate, -8.0, 0.5, GROUND_TOP + 1, 0.0);
            plate.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 100;
            s.metrics = (sc, r) -> fmt("plate distance %.3f m, max |vy| %.3f, max spin %.3f, plate lift %+.4f, deck moved %.4f m, plate tilt %.2f",
                    r.last(1)[0] - r.at(0, 1)[0], Math.max(r.max(1, a -> a[8]), -r.min(1, a -> a[8])), r.max(1, ParityHarness::spin),
                    r.max(1, a -> a[1]) - sc.bodies.get(1).restingHeight(GROUND_TOP + 1),
                    Math.hypot(r.last(0)[0] - r.at(0, 0)[0], r.last(0)[2] - r.at(0, 0)[2]), tilt(r.last(1)));
            list.add(s);
        }
        {
            // A plate with an irregular underside (holes, slabs) sliding over uniform stone.
            final Scenario s = new Scenario("rough-plate", "7x1x7 plate with holes and slabs in it slides at 6 m/s over checkered stone/dirt");
            ground(s.terrain, -24, -24, 63, 39, STONE, DIRT);
            final BodySpec b = s.body(0).fill(0, 0, 0, 6, 0, 6, PLANKS);
            final java.util.Random random = new java.util.Random(13);
            for (int x = 1; x < 6; x++) {
                for (int z = 1; z < 6; z++) {
                    final int pick = random.nextInt(4);
                    if (pick == 0) {
                        b.set(x, 0, z, null);
                    } else if (pick == 1) {
                        b.set(x, 0, z, TOP_SLAB);
                    }
                }
            }
            placeAbove(b, -10.0, 0.5, GROUND_TOP, 0.0);
            b.linearVelocity.set(6.0, 0.0, 0.0);
            s.ticks = 120;
            s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
            list.add(s);
        }
    }

    /** Removes every third ground block in both directions (1x1 holes, one block deep). */
    static void holeGrid(final Scenario s) {
        for (int x = -24; x <= 95; x += 3) {
            for (int z = -24; z <= 39; z += 3) {
                s.terrain.set(x, GROUND_TOP - 1, z, null);
            }
        }
    }

    /**
     * An upright block disc of {@code size x size} blocks (cells within {@code radiusSq} of the center)
     * rolling along x without slipping at {@code speed} over checkered ground.
     */
    static Scenario wheel(final String name, final int size, final double radiusSq, final double speed, final int ticks) {
        final Scenario s = new Scenario(name, "upright " + size + "x" + size + " block disc rolls at " + speed + " m/s over checkered stone/dirt");
        ground(s.terrain, -24, -24, 95, 39, STONE, DIRT);
        final BodySpec b = s.body(0);
        final double c = (size - 1) * 0.5;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                final double dx = x - c, dy = y - c;
                if (dx * dx + dy * dy <= radiusSq) {
                    b.set(x, y, 0, PLANKS);
                }
            }
        }
        placeAbove(b, -10.0, 0.5, GROUND_TOP, 0.0);
        b.linearVelocity.set(speed, 0.0, 0.0);
        b.angularVelocity.set(0.0, 0.0, -speed / (size * 0.5));
        s.ticks = ticks;
        s.metrics = (sc, r) -> {
            // distance where the wheel first nearly stops (it then rocks back and forth)
            double stopX = Double.NaN;
            for (int t = 1; t <= r.ticks; t++) {
                if (Math.abs(r.at(t, 0)[7]) < 0.15) {
                    stopX = r.at(t, 0)[0] - r.at(0, 0)[0];
                    break;
                }
            }
            return fmt("first stop at %.3f m, final distance %.3f m, max |vy| %.3f, max y %.3f, lateral %.3f, lean %.1f",
                    stopX, r.last(0)[0] - r.at(0, 0)[0], Math.max(r.max(0, a -> a[8]), -r.min(0, a -> a[8])), r.max(0, a -> a[1]),
                    r.last(0)[2] - r.at(0, 0)[2], lean(r.last(0)));
        };
        return s;
    }

    /** Lift, vertical speed and spin while moving, final tilt and lateral drift of one body. */
    static String motion(final Scenario s, final Run r, final int body, final double top) {
        final double ideal = s.bodies.get(body).restingHeight(top);
        return fmt("max lift %+.4f, max |vy| %.3f, max spin %.3f, final tilt %.2f, lateral %.3f",
                r.max(body, a -> a[1]) - ideal, Math.max(r.max(body, a -> a[8]), -r.min(body, a -> a[8])),
                r.max(body, ParityHarness::spin), tilt(r.last(body)), r.last(body)[2] - r.at(0, body)[2]);
    }

    /** Angle (degrees) between the body's z axis and world z: how far an upright disc leans over. */
    static double lean(final double[] s) {
        final org.joml.Quaterniond q = new org.joml.Quaterniond(s[3], s[4], s[5], s[6]);
        final org.joml.Vector3d axis = q.transform(new org.joml.Vector3d(0, 0, 1));
        return Math.toDegrees(Math.acos(Math.min(1.0, Math.abs(axis.z))));
    }

    /** Heading (degrees) of the body's x axis around world y. */
    static double yaw(final double[] s) {
        final org.joml.Quaterniond q = new org.joml.Quaterniond(s[3], s[4], s[5], s[6]);
        final org.joml.Vector3d fwd = q.transform(new org.joml.Vector3d(1, 0, 0));
        return Math.toDegrees(Math.atan2(-fwd.z, fwd.x));
    }

    static double minGap(final Run r) {
        double m = Double.POSITIVE_INFINITY;
        for (int t = 0; t <= r.ticks; t++) {
            m = Math.min(m, r.at(t, 1)[0] - r.at(t, 0)[0] - 2.0);
        }
        return m;
    }

    static Scenario slide(final String name, final String description, final int sx, final int sy, final int sz,
                          final double speed, final Material... ground) {
        final Scenario s = new Scenario(name, description);
        ground(s.terrain, -24, -24, 63, 39, ground);
        final BodySpec b = s.body(0).fill(0, 0, 0, sx, sy, sz, PLANKS);
        placeAbove(b, -10.0, 0.5, GROUND_TOP, 0.0);
        b.linearVelocity.set(speed, 0.0, 0.0);
        s.ticks = 120;
        s.metrics = (sc, r) -> slideMetrics(sc, r, GROUND_TOP);
        return s;
    }

    /** Distance, stop tick, vertical hops and spin while sliding (seam snags show up as hops/spin). */
    static String slideMetrics(final Scenario s, final Run r, final double top) {
        int stop = -1;
        for (int t = 1; t <= r.ticks; t++) {
            if (speed(r.at(t, 0)) < 0.05) {
                stop = t;
                break;
            }
        }
        final double ideal = s.bodies.get(0).restingHeight(top);
        return fmt("distance %.3f m, stops @%d, max |vy| %.3f, max spin %.3f, max lift %+.4f, sink %+.4f, final tilt %.2f, lateral %.3f",
                r.last(0)[0] - r.at(0, 0)[0], stop, Math.max(r.max(0, a -> a[8]), -r.min(0, a -> a[8])),
                r.max(0, ParityHarness::spin), r.max(0, a -> a[1]) - ideal, r.last(0)[1] - ideal, tilt(r.last(0)),
                r.last(0)[2] - r.at(0, 0)[2]);
    }
}
