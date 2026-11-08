package dev.behindthescenery.sablejolt.constraint.free;

import com.github.stephengold.joltjni.Body;
import com.github.stephengold.joltjni.MotorSettings;
import com.github.stephengold.joltjni.Quat;
import com.github.stephengold.joltjni.RVec3;
import com.github.stephengold.joltjni.SixDofConstraint;
import com.github.stephengold.joltjni.SixDofConstraintSettings;
import com.github.stephengold.joltjni.Vec3;
import com.github.stephengold.joltjni.enumerate.EAxis;
import com.github.stephengold.joltjni.enumerate.EConstraintSpace;
import com.github.stephengold.joltjni.enumerate.EMotorState;
import dev.behindthescenery.sablejolt.JoltDebugLogging;
import dev.behindthescenery.sablejolt.JoltPhysicsScene;
import dev.behindthescenery.sablejolt.constraint.JoltConstraintHandle;
import dev.behindthescenery.sablejolt.constraint.SixDofMotors;
import dev.behindthescenery.sablejolt.constraint.StaffDebug;
import dev.behindthescenery.sablejolt.constraint.fixed.JoltFixedConstraintHandle;
import dev.ryanhcode.sable.api.physics.PhysicsPipelineBody;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintConfiguration;
import dev.ryanhcode.sable.api.physics.constraint.FreeConstraintHandle;
import dev.ryanhcode.sable.sublevel.ServerSubLevel;
import org.jetbrains.annotations.ApiStatus;
import org.jetbrains.annotations.Nullable;
import org.joml.Quaterniond;
import org.joml.Vector3d;

@ApiStatus.Internal
public class JoltFreeConstraintHandle extends JoltConstraintHandle implements FreeConstraintHandle {

    private final SixDofMotors.MotorParams[] motors = new SixDofMotors.MotorParams[SixDofMotors.AXIS_COUNT];
    private com.github.stephengold.joltjni.SixDofConstraintSettings settings;
    private org.joml.Quaterniond frameQuat = new org.joml.Quaterniond();

    /**
     * The world-space target of the world-side anchor in the scene (plot) frame.
     * Only used when {@link #renderFrameServo} is false (the anchor geometry
     * carries the target, motor targets are zero — the iron handle pattern).
     */
    private final Vector3d targetWorld = new Vector3d();

    /**
     * True when the caller drives the motors with per-tick joint-space targets
     * expressed in the render frame (the physics staff pattern: pos1 is the
     * ZERO sentinel). The servo then measures the body anchor through the
     * sub-level's logical pose instead of the constraint geometry.
     */
    private boolean renderFrameServo;

    @Nullable
    private ServerSubLevel subLevel;

    public static JoltFreeConstraintHandle create(final JoltPhysicsScene scene, @Nullable final PhysicsPipelineBody bodyA, @Nullable final PhysicsPipelineBody bodyB, final FreeConstraintConfiguration config, final boolean renderFrameServo) {
        // The servo drives body 2. Ropes and plungers may pass the static world as
        // body B, and the ground body has no motion properties: its addForce is a
        // null read in native code, which takes the JVM down. Keep the world on side 1.
        if (bodyB == null && bodyA != null) {
            return create(scene, null, bodyA, new FreeConstraintConfiguration(config.pos2(), config.pos1(), config.orientation()), renderFrameServo);
        }

        final JoltPhysicsScene.SableBody sbA = bodyA == null ? null : scene.body(bodyA.getRuntimeId());
        final JoltPhysicsScene.SableBody sbB = bodyB == null ? null : scene.body(bodyB.getRuntimeId());
        if ((sbA == null && sbB == null) || !scene.canConstrain(bodyA, sbA) || !scene.canConstrain(bodyB, sbB)) {
            return new JoltFreeConstraintHandle(scene, -1);
        }

        final int joltA = sbA != null ? sbA.joltId : scene.groundBodyId();
        final int joltB = sbB != null ? sbB.joltId : scene.groundBodyId();

        final SixDofConstraintSettings settings = new SixDofConstraintSettings();
        settings.setSpace(EConstraintSpace.LocalToBodyCom);

        // Sable's contract: each anchor arrives in the block space of its own body
        // (the plot of a sub-level, world coordinates for the static world) while
        // the Jolt scene lives in the render frame next to the player. Project each
        // one with its own sub-level's logical pose (the same transform Sable's
        // projectOutOfSubLevel applies), then express it relative to the LIVE center
        // of mass so forces at the anchor produce realistic torque. All axes are
        // free, so Jolt itself never pulls on these anchors; keep them small to stay
        // away from float precision loss.
        final ServerSubLevel ssl = bodyB instanceof final ServerSubLevel s ? s : null;
        final org.joml.Vector3dc p1World = bodyA instanceof final ServerSubLevel sa
                ? sa.logicalPose().transformPosition(config.pos1(), new org.joml.Vector3d())
                : config.pos1();
        final org.joml.Vector3dc p2World = ssl != null
                ? ssl.logicalPose().transformPosition(config.pos2(), new org.joml.Vector3d())
                : config.pos2();
        settings.setPosition1(sbA != null ? comRelative(sbA, p1World) : RVec3.sZero());
        settings.setPosition2(sbB != null ? comRelative(sbB, p2World)
                : new RVec3((float) p2World.x(), (float) p2World.y(), (float) p2World.z()));

        final var q = config.orientation();
        // NOTE: do NOT compose this with the sub-level's live orientation —
        // logicalPose tracks the body itself, so the servo target would move
        // with the body (positive feedback → perpetual self-spin). The gun
        // orientation is used as-is; it is stable across ticks.
        final Quaterniond frameQ = new Quaterniond(q.x(), q.y(), q.z(), q.w());
        settings.setAxisX1(JoltFixedConstraintHandle.rotate((float) frameQ.x, (float) frameQ.y, (float) frameQ.z, (float) frameQ.w, Vec3.sAxisX()));
        settings.setAxisY1(JoltFixedConstraintHandle.rotate((float) frameQ.x, (float) frameQ.y, (float) frameQ.z, (float) frameQ.w, Vec3.sAxisY()));

        for (final EAxis axis : SixDofMotors.AXES) {
            if (axis == EAxis.Num) {
                continue;
            }
            settings.makeFreeAxis(axis);
        }

        final SixDofConstraint constraint = (SixDofConstraint) scene.createConstraint(settings, joltA, joltB);

        final var record = new JoltPhysicsScene.JointRecord();
        record.joltIdA = joltA;
        record.joltIdB = joltB;
        record.constraint = constraint;
        record.contactsEnabled = true;
        record.fixedContacts = true;
        record.genericSettings = settings;
        final long id = scene.registerJoint(record);

        if (JoltDebugLogging.STAFF) {
            StaffDebug.constraintCreated(sbB != null ? sbB.runtimeId : -1,
                    config.pos2().x(), config.pos2().y(), config.pos2().z(),
                    q.x(), q.y(), q.z(), q.w());
        }

        final JoltFreeConstraintHandle handle = new JoltFreeConstraintHandle(scene, id);
        handle.settings = settings;
        handle.frameQuat.set(frameQ);
        handle.renderFrameServo = renderFrameServo;
        if (!renderFrameServo) {
            handle.targetWorld.set(p1World.x(), p1World.y(), p1World.z());
        }
        handle.subLevel = ssl;
        return handle;
    }

    private static RVec3 comRelative(final JoltPhysicsScene.SableBody sb, final org.joml.Vector3dc world) {
        final RVec3 com = sb.body.getCenterOfMassPosition();
        final Quat inv = sb.body.getRotation().conjugated();
        final Vec3 off = JoltFixedConstraintHandle.rotate(inv.getX(), inv.getY(), inv.getZ(), inv.getW(),
                (float) (world.x() - com.xx()), (float) (world.y() - com.yy()), (float) (world.z() - com.zz()));
        return new RVec3(off.getX(), off.getY(), off.getZ());
    }

    private JoltFreeConstraintHandle(final JoltPhysicsScene scene, final long handle) {
        super(scene);
        this.attach(handle);
    }

    @Override
    public void remove() {
        // The angular servo can leave significant residual spin (it drives the
        // body towards up to MAX_ANGULAR_SPEED while following the grab). On
        // release that reads as the object wildly spinning for several turns.
        // Damp it here — both for the per-tick constraint recreation and for the
        // final release.
        JoltPhysicsScene.JointRecord record = null;
        try {
            record = this.record();
            // A rope or plunger letting go must not stop the ship's rotation.
            if (record != null && !record.damper && record.constraint instanceof final SixDofConstraint constraint) {
                final Body body = constraint.getBody2();
                if (body != null && body.isDynamic()) {
                    // Fully stop the residual spin: the servo re-drives omega within
                    // the same tick during the per-tick constraint recreation, so
                    // only the final release ever keeps the zeroed value.
                    body.setAngularVelocity(0.0f, 0.0f, 0.0f);
                }
            }
        } catch (final Throwable ignored) {
            // never block removal on diagnostics
        }
        // End of a creative drag (or a per-tick recreation — the restore is
        // deferred to the next step start, so recreation is harmless).
        try {
            if (record == null) {
                record = this.record();
            }
            if (record != null) {
                this.scene.endCreativeDrag(record.joltIdB);
            }
        } catch (final Throwable ignored) {
            // never block removal on bookkeeping
        }
        super.remove();
    }

    @Override
    protected void setMotorImpl(final int axisOrdinal, final double target, final double stiffness, final double damping, final boolean hasForceLimit, final double maxForce) {
        final JoltPhysicsScene.JointRecord record = this.record();
        if (record == null || !(record.constraint instanceof final SixDofConstraint constraint)) {
            return;
        }

        // No spring and no force limit: plain drag between the anchors, set once and
        // active for the constraint's whole life (ropes, plunger launcher). Not the
        // creative grip: that one would switch the ship's gravity off and snap its
        // orientation. Measured on Rapier: dv/dt = -damping * v, independent of mass.
        if (!hasForceLimit && stiffness == 0.0) {
            this.scene.setJointDamping(record, axisOrdinal < 3, damping);
            return;
        }

        this.motors[axisOrdinal] = new SixDofMotors.MotorParams(target, stiffness, damping, hasForceLimit, maxForce);
        // Creative grip (no force limit = physics staff): gravity is switched
        // off for the body while held, so weight can neither pull nor tip it.
        if (!hasForceLimit && axisOrdinal < 3) {
            this.scene.beginCreativeDrag(record.joltIdB);
        }
        SixDofMotors.applyMotorFree(constraint, this.settings, axisOrdinal, this.motors, this.scene,
                record.joltIdA, record.joltIdB, this.frameQuat, this.targetWorld, this.renderFrameServo, this.subLevel);
    }
}
