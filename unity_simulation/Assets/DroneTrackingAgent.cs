using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(LockedAltitudeDroneController))]
[RequireComponent(typeof(Rigidbody))]
public class DroneTrackingAgent : Agent
{
    private const int ExpectedVectorObservationSize = 19;

    public enum RuntimeTrackingOutcome
    {
        LostTarget,
        OutOfBounds,
        Collision,
        TargetUnavailable
    }

    [Header("References")]
    [SerializeField] private LockedAltitudeDroneController droneController;
    [SerializeField] private Rigidbody rb;
    [SerializeField] private DroneTrackingTrainingManager trainingManager;
    [SerializeField] private TruckTarget trackedTruck;
    [SerializeField] private Transform sensorOrigin;
    [SerializeField] private DroneTrackingSensorFootprint sensorFootprint;

    [Header("Tracking")]
    [SerializeField, Min(10f)] private float maxTrackingDistance = 180f;
    [SerializeField, Min(5f)] private float idealTrackingDistance = 55f;
    [SerializeField, Min(1f)] private float distanceTolerance = 18f;
    [SerializeField, Range(10f, 89f)] private float viewHalfAngle = 65f;
    [SerializeField, Min(0.1f)] private float maxTimeWithoutSight = 6f;
    [SerializeField, Min(0f)] private float runtimeClaimReleaseGraceTime = 2.5f;
    [SerializeField, Min(0f)] private float outOfBoundsGraceDistance = 6f;
    [SerializeField, Min(1f)] private float maxExpectedTargetSpeed = 16f;
    [SerializeField, Min(1f)] private float maxExpectedOwnSpeed = 18f;
    [SerializeField] private bool endEpisodeAfterStableAcquire = false;
    [SerializeField, Min(0.1f)] private float stableAcquireDuration = 8f;
    [SerializeField, Range(0f, 1f)] private float stableAcquireCenterednessThreshold = 0.72f;
    [SerializeField] private bool forwardYawOnly = true;

    [Header("Rewards")]
    [SerializeField] private float stepPenalty = 0.0005f;
    [SerializeField] private float visibleReward = 0.015f;
    [SerializeField] private float inRangeReward = 0.01f;
    [SerializeField] private float distanceRewardScale = 0.03f;
    [SerializeField] private float frontAlignmentReward = 0.02f;
    [SerializeField] private float keepUpRewardScale = 0.02f;
    [SerializeField, Range(0f, 1f)] private float lostSightFrontAlignmentMultiplier = 0.7f;
    [SerializeField] private float yawPenalty = 0.0002f;
    [SerializeField] private float stableAcquireReward = 1f;
    [SerializeField] private float lostTargetPenalty = 1.25f;
    [SerializeField] private float outOfBoundsPenalty = 1f;
    [SerializeField] private float collisionPenalty = 1f;
    [SerializeField] private float timeoutPenalty = 0.25f;
    [SerializeField, Range(0f, 1f)] private float preferredViewportX = 0.5f;
    [SerializeField, Range(0f, 1f)] private float preferredViewportY = 0.52f;
    [SerializeField, Min(0f)] private float viewportHorizontalRewardScale = 0.03f;
    [SerializeField, Min(0f)] private float viewportVerticalRewardScale = 0.02f;
    [SerializeField, Min(0f)] private float viewportCenterRewardScale = 0.045f;
    [SerializeField, Min(0f)] private float viewportOffCenterPenaltyScale = 0.03f;
    [SerializeField, Min(0f)] private float viewportEdgePenaltyScale = 0.015f;
    [SerializeField, Range(0.01f, 0.49f)] private float viewportCenterComfortRadius = 0.18f;
    [SerializeField, Range(0.01f, 0.49f)] private float edgeComfortMargin = 0.14f;

    [Header("Action Settings")]
    [SerializeField, Range(0f, 1f)] private float actionSmoothing = 0.8f;
    [SerializeField, Range(0f, 0.5f)] private float actionDeadzone = 0.05f;
    [SerializeField, Min(0f)] private float runtimeInitialAssistDuration = 2f;
    [SerializeField, Range(0f, 1f)] private float runtimeInitialAssistForward = 0.65f;
    [SerializeField, Range(0f, 1f)] private float runtimeLostSightAssistForward = 0.5f;
    [SerializeField, Range(0f, 1f)] private float runtimeLostSightMinForward = 0.28f;
    [SerializeField, Range(0f, 1f)] private float runtimeVisiblePursuitForward = 0.8f;
    [SerializeField, Range(0f, 1f)] private float runtimeVisibleMinForward = 0.22f;
    [SerializeField, Range(0f, 1f)] private float runtimeTurnAssistBlend = 0.7f;
    [SerializeField, Range(0f, 1f)] private float runtimeVisibleTurnAssistBlend = 0.55f;
    [SerializeField, Range(10f, 180f)] private float runtimeAssistFullTurnAngle = 60f;

    private bool collisionDetected;
    private bool hasTrackedTarget;
    private bool runtimeHasDirectSight;
    private bool originalManualInputState;
    private bool runtimeTrackingMode;
    private bool runtimeTrackingActive;
    private bool hasRuntimeMissionBounds;
    private Vector3 lastKnownTrackingPoint;
    private Vector3 lastKnownTargetVelocity;
    private float runtimeTrackingElapsedTime;
    private float timeSinceLastVisible;
    private float stableAcquireTimer;
    private Vector3 smoothedAction;
    private bool debugLastVisible;
    private float debugLastDistanceToTarget;
    private float debugLastObservedDistance;
    private float debugLastForwardCommand;
    private float debugLastTurnCommand;
    private float debugLastStrafeCommand;
    private Vector2 runtimeMissionCenterXZ;
    private Vector2 runtimeMissionSizeXZ;

    public event System.Action<DroneTrackingAgent, RuntimeTrackingOutcome> RuntimeTrackingEnded;

    public float MaxTrackingDistance => maxTrackingDistance;
    public bool RuntimeTrackingActive => runtimeTrackingActive;
    public TruckTarget TrackedTruck => trackedTruck;
    public bool DebugLastVisible => debugLastVisible;
    public bool DebugCollisionDetected => collisionDetected;
    public bool DebugHasTrackedTarget => hasTrackedTarget;
    public bool DebugHasDirectSight => runtimeHasDirectSight;
    public float DebugTimeSinceLastVisible => timeSinceLastVisible;
    public float DebugLastDistanceToTarget => debugLastDistanceToTarget;
    public float DebugLastObservedDistance => debugLastObservedDistance;
    public float DebugLastForwardCommand => debugLastForwardCommand;
    public float DebugLastTurnCommand => debugLastTurnCommand;
    public float DebugLastStrafeCommand => debugLastStrafeCommand;
    public float DebugCurrentSpeed => ProjectToXZ(rb != null ? rb.linearVelocity : Vector3.zero).magnitude;
    public float DebugCurrentMaxSpeed => droneController != null ? droneController.maxSpeed : 0f;
    public float DebugCurrentTurnSpeed => droneController != null ? droneController.turnSpeed : 0f;

    public override void Initialize()
    {
        ResolveReferences();

        if (droneController != null)
        {
            originalManualInputState = droneController.manualInputEnabled;
            droneController.manualInputEnabled = false;
        }

        if (MaxStep <= 0 && !runtimeTrackingMode)
        {
            MaxStep = 5000;
        }

        ValidateObservationSizeConfiguration();
    }

    public override void OnEpisodeBegin()
    {
        ResolveReferences();

        ResetTrackingState(clearTrackedTruck: false);

        if (runtimeTrackingMode)
        {
            return;
        }

        if (trainingManager != null)
        {
            trainingManager.ResetEpisode();
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        ResolveReferences();

        if (runtimeTrackingMode && !runtimeTrackingActive)
        {
            AddZeroObservations(sensor);
            return;
        }

        if (trackedTruck == null)
        {
            AddZeroObservations(sensor);
            return;
        }

        Vector3 trackingPoint = trackedTruck.GetTrackingPoint();
        Vector3 targetVelocity = ProjectToXZ(trackedTruck.CurrentVelocity);
        Vector3 flatTrackingPoint = new Vector3(trackingPoint.x, transform.position.y, trackingPoint.z);
        Vector3 localOwnVelocity = transform.InverseTransformDirection(ProjectToXZ(rb != null ? rb.linearVelocity : Vector3.zero));

        Vector3 toTargetFlat = flatTrackingPoint - transform.position;
        float distance = toTargetFlat.magnitude;
        float forwardDot = distance > 0.001f
            ? Vector3.Dot(transform.forward, toTargetFlat / distance)
            : 1f;

        bool inRange = distance <= maxTrackingDistance;
        bool inFront = forwardDot > 0f;
        bool inViewCone = IsTargetInsideSensorView(trackingPoint, forwardDot);
        bool hasLineOfSight = HasLineOfSight(trackingPoint);
        bool visible = inRange && inViewCone && hasLineOfSight;
        UpdateTargetMemory(visible, trackingPoint, targetVelocity);

        bool hasObservedTarget = TryGetObservedTargetState(
            visible,
            trackingPoint,
            targetVelocity,
            out Vector3 observedTrackingPoint,
            out Vector3 observedTargetVelocity);

        Vector3 observedFlatTrackingPoint = hasObservedTarget
            ? new Vector3(observedTrackingPoint.x, transform.position.y, observedTrackingPoint.z)
            : transform.position;
        Vector3 localTargetPosition = hasObservedTarget
            ? transform.InverseTransformPoint(observedFlatTrackingPoint)
            : Vector3.zero;
        Vector3 localTargetVelocity = hasObservedTarget
            ? transform.InverseTransformDirection(observedTargetVelocity)
            : Vector3.zero;
        Vector3 toObservedTargetFlat = observedFlatTrackingPoint - transform.position;
        float observedDistance = hasObservedTarget ? toObservedTargetFlat.magnitude : 0f;
        float observedForwardDot = hasObservedTarget && observedDistance > 0.001f
            ? Vector3.Dot(transform.forward, toObservedTargetFlat / observedDistance)
            : 0f;
        bool observedInRange = hasObservedTarget && observedDistance <= maxTrackingDistance;
        bool observedInFront = hasObservedTarget && observedForwardDot > 0f;

        sensor.AddObservation(Mathf.Clamp(localTargetPosition.x / maxTrackingDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localTargetPosition.z / maxTrackingDistance, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localTargetVelocity.x / maxExpectedTargetSpeed, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localTargetVelocity.z / maxExpectedTargetSpeed, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localOwnVelocity.x / maxExpectedOwnSpeed, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(localOwnVelocity.z / maxExpectedOwnSpeed, -1f, 1f));
        sensor.AddObservation(hasObservedTarget ? Mathf.Clamp01(observedDistance / maxTrackingDistance) : 1f);
        sensor.AddObservation(hasObservedTarget
            ? Mathf.Clamp((observedDistance - idealTrackingDistance) / Mathf.Max(1f, distanceTolerance * 2f), -1f, 1f)
            : 1f);
        sensor.AddObservation(Mathf.Clamp(observedForwardDot, -1f, 1f));
        sensor.AddObservation(observedInRange ? 1f : 0f);
        sensor.AddObservation(observedInFront ? 1f : 0f);
        sensor.AddObservation(visible ? 1f : 0f);
        sensor.AddObservation(hasTrackedTarget ? 1f : 0f);
        sensor.AddObservation(visible ? 1f : 0f);
        sensor.AddObservation(GetNormalizedTimeSinceLastVisibleObservation());

        Vector4 boundsDistances = GetNormalizedDistancesToBounds(transform.position);

        sensor.AddObservation(boundsDistances.x);
        sensor.AddObservation(boundsDistances.y);
        sensor.AddObservation(boundsDistances.z);
        sensor.AddObservation(boundsDistances.w);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        ResolveReferences();

        if (runtimeTrackingMode && !runtimeTrackingActive)
        {
            return;
        }

        if (droneController == null)
        {
            return;
        }

        if (trackedTruck == null)
        {
            if (runtimeTrackingMode && runtimeTrackingActive)
            {
                NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome.TargetUnavailable);
            }

            return;
        }

        var continuousActions = actions.ContinuousActions;
        Vector3 rawAction = new Vector3(
            Mathf.Clamp(continuousActions[0], -1f, 1f),
            Mathf.Clamp(continuousActions[1], -1f, 1f),
            Mathf.Clamp(continuousActions[2], -1f, 1f));

        smoothedAction = Vector3.Lerp(smoothedAction, rawAction, actionSmoothing);

        float strafe = ApplyActionDeadzone(smoothedAction.x);
        float forward = ApplyActionDeadzone(smoothedAction.y);
        float turn = ApplyActionDeadzone(smoothedAction.z);

        if (forwardYawOnly)
        {
            strafe = 0f;
            forward = Mathf.Max(0f, forward);
        }

        AddReward(-stepPenalty);

        if (rb != null)
        {
            AddReward(-Mathf.Clamp01(Mathf.Abs(rb.angularVelocity.y) / 6f) * yawPenalty);
        }

        if (runtimeTrackingMode)
        {
            runtimeTrackingElapsedTime += Time.fixedDeltaTime;
        }

        Vector3 trackingPoint = trackedTruck.GetTrackingPoint();
        Vector3 targetVelocity = ProjectToXZ(trackedTruck.CurrentVelocity);
        Vector3 flatTrackingPoint = new Vector3(trackingPoint.x, transform.position.y, trackingPoint.z);
        Vector3 toTargetFlat = flatTrackingPoint - transform.position;
        float liveDistance = toTargetFlat.magnitude;
        debugLastDistanceToTarget = liveDistance;
        float liveForwardDot = liveDistance > 0.001f
            ? Vector3.Dot(transform.forward, toTargetFlat / liveDistance)
            : 1f;

        bool inRange = liveDistance <= maxTrackingDistance;
        bool inViewCone = IsTargetInsideSensorView(trackingPoint, liveForwardDot);
        bool hasLineOfSight = HasLineOfSight(trackingPoint);
        bool visible = inRange && inViewCone && hasLineOfSight;
        debugLastVisible = visible;
        float viewportCentering01 = 0f;

        if (visible)
        {
            if (runtimeTrackingMode)
            {
                runtimeHasDirectSight = true;
            }

            UpdateTargetMemory(true, trackingPoint, targetVelocity);
            timeSinceLastVisible = 0f;
            AddReward(visibleReward);
            viewportCentering01 = AddViewportCenteringReward(trackingPoint);

            if (viewportCentering01 >= stableAcquireCenterednessThreshold)
            {
                stableAcquireTimer += Time.fixedDeltaTime;
            }
            else
            {
                stableAcquireTimer = 0f;
            }
        }
        else
        {
            timeSinceLastVisible = hasTrackedTarget
                ? timeSinceLastVisible + Time.fixedDeltaTime
                : 0f;
            stableAcquireTimer = 0f;
        }

        bool hasObservedTarget = TryGetObservedTargetState(
            visible,
            trackingPoint,
            targetVelocity,
            out Vector3 observedTrackingPoint,
            out Vector3 observedTargetVelocity);

        if (runtimeTrackingMode && hasObservedTarget)
        {
            ApplyRuntimeTrackingAssist(ref forward, ref turn, observedTrackingPoint, observedTargetVelocity, visible);
        }

        debugLastStrafeCommand = strafe;
        debugLastForwardCommand = forward;
        debugLastTurnCommand = turn;
        droneController.SetControlInputs(strafe, forward, turn);

        if (hasObservedTarget)
        {
            Vector3 observedFlatTrackingPoint = new Vector3(observedTrackingPoint.x, transform.position.y, observedTrackingPoint.z);
            Vector3 toObservedTargetFlat = observedFlatTrackingPoint - transform.position;
            float observedDistance = toObservedTargetFlat.magnitude;
            debugLastObservedDistance = observedDistance;
            float observedForwardDot = observedDistance > 0.001f
                ? Vector3.Dot(transform.forward, toObservedTargetFlat / observedDistance)
                : 0f;
            bool observedInRange = observedDistance <= maxTrackingDistance;

            if (observedInRange)
            {
                AddReward(inRangeReward);
            }

            float distanceReward01 = 1f - Mathf.Clamp01(Mathf.Abs(observedDistance - idealTrackingDistance) / Mathf.Max(1f, distanceTolerance));
            AddReward(distanceReward01 * distanceRewardScale);

            float frontAlignmentScale;
            if (visible)
            {
                frontAlignmentScale = Mathf.Lerp(0.15f, 1f, viewportCentering01);
            }
            else
            {
                float reacquireWindow01 = 1f - Mathf.Clamp01(timeSinceLastVisible / Mathf.Max(0.01f, maxTimeWithoutSight));
                frontAlignmentScale = reacquireWindow01 * lostSightFrontAlignmentMultiplier;
            }

            AddReward(Mathf.Max(0f, observedForwardDot) * frontAlignmentReward * frontAlignmentScale);

            float targetSpeed = observedTargetVelocity.magnitude;
            float ownSpeed = ProjectToXZ(rb != null ? rb.linearVelocity : Vector3.zero).magnitude;
            float speedMatch01 = 1f - Mathf.Clamp01(Mathf.Abs(ownSpeed - targetSpeed) / Mathf.Max(1f, maxExpectedOwnSpeed));
            AddReward(speedMatch01 * keepUpRewardScale);
        }

        if (collisionDetected)
        {
            AddReward(-collisionPenalty);
            if (runtimeTrackingMode)
            {
                NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome.Collision);
            }
            else
            {
                EndEpisode();
            }

            return;
        }

        float outsideDistance = DistanceOutsideBounds(transform.position);
        if (outsideDistance > outOfBoundsGraceDistance)
        {
            AddReward(-outOfBoundsPenalty);
            if (runtimeTrackingMode)
            {
                NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome.OutOfBounds);
            }
            else
            {
                EndEpisode();
            }

            return;
        }

        if (hasTrackedTarget && visible && liveDistance > maxTrackingDistance + Mathf.Max(distanceTolerance, 10f))
        {
            AddReward(-lostTargetPenalty);
            if (runtimeTrackingMode)
            {
                NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome.LostTarget);
            }
            else
            {
                EndEpisode();
            }

            return;
        }

        if (hasTrackedTarget && timeSinceLastVisible > GetLostTargetReleaseThreshold())
        {
            AddReward(-lostTargetPenalty);
            if (runtimeTrackingMode)
            {
                NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome.LostTarget);
            }
            else
            {
                EndEpisode();
            }

            return;
        }

        if (!runtimeTrackingMode && endEpisodeAfterStableAcquire && stableAcquireTimer >= stableAcquireDuration)
        {
            AddReward(stableAcquireReward);
            EndEpisode();
            return;
        }

        if (!runtimeTrackingMode && MaxStep > 0 && StepCount >= MaxStep)
        {
            AddReward(-timeoutPenalty);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;

        float strafe = Input.GetAxisRaw("Horizontal");
        float forward = Input.GetAxisRaw("Vertical");
        float turn = Input.GetKey(KeyCode.Q) ? -1f : (Input.GetKey(KeyCode.E) ? 1f : 0f);

        if (forwardYawOnly)
        {
            strafe = 0f;
            forward = Mathf.Max(0f, forward);
        }

        continuousActions[0] = Mathf.Clamp(strafe, -1f, 1f);
        continuousActions[1] = Mathf.Clamp(forward, -1f, 1f);
        continuousActions[2] = Mathf.Clamp(turn, -1f, 1f);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (!collision.collider.isTrigger)
        {
            collisionDetected = true;
        }
    }

    private void FixedUpdate()
    {
        if (runtimeTrackingMode && runtimeTrackingActive && isActiveAndEnabled)
        {
            RequestDecision();
        }
    }

    protected override void OnDisable()
    {
        base.OnDisable();

        if (droneController != null)
        {
            droneController.ClearControlInputs();
            droneController.manualInputEnabled = originalManualInputState;
        }
    }

    public void ConfigureRuntimeTracking(Vector2 missionCenterXZ, Vector2 missionSizeXZ)
    {
        runtimeTrackingMode = true;
        ResolveReferences();

        runtimeTrackingActive = false;
        hasRuntimeMissionBounds = missionSizeXZ.x > 0.001f && missionSizeXZ.y > 0.001f;
        runtimeMissionCenterXZ = missionCenterXZ;
        runtimeMissionSizeXZ = missionSizeXZ;
        trainingManager = null;
        MaxStep = 0;

        ResetTrackingState(clearTrackedTruck: true);

        if (droneController != null)
        {
            droneController.manualInputEnabled = false;
            droneController.ClearControlInputs();
        }
    }

    public void BeginRuntimeTracking(TruckTarget target)
    {
        runtimeTrackingMode = true;
        ResolveReferences();

        trackedTruck = target;
        runtimeTrackingActive = trackedTruck != null;
        collisionDetected = false;
        hasTrackedTarget = trackedTruck != null;
        runtimeHasDirectSight = false;
        runtimeTrackingElapsedTime = 0f;
        timeSinceLastVisible = 0f;
        stableAcquireTimer = 0f;
        smoothedAction = Vector3.zero;

        if (trackedTruck != null)
        {
            // Seed target memory from the claim frame so edge-of-footprint claims can
            // still predict forward briefly and recover back to search if tracking fails.
            lastKnownTrackingPoint = trackedTruck.GetTrackingPoint();
            lastKnownTargetVelocity = ProjectToXZ(trackedTruck.CurrentVelocity);
        }
        else
        {
            lastKnownTrackingPoint = Vector3.zero;
            lastKnownTargetVelocity = Vector3.zero;
        }

        if (droneController != null)
        {
            droneController.manualInputEnabled = false;
            droneController.ClearControlInputs();
        }

        if (runtimeTrackingActive && isActiveAndEnabled)
        {
            RequestDecision();
        }
    }

    public void StopRuntimeTracking()
    {
        runtimeTrackingActive = false;
        trackedTruck = null;
        ResetTrackingState(clearTrackedTruck: false);

        if (droneController != null)
        {
            droneController.ClearControlInputs();
        }
    }

    private void ResolveReferences()
    {
        if (droneController == null)
        {
            droneController = GetComponent<LockedAltitudeDroneController>();
        }

        if (rb == null)
        {
            rb = GetComponent<Rigidbody>();
        }

        if (trainingManager == null && !runtimeTrackingMode)
        {
            trainingManager = FindFirstObjectByType<DroneTrackingTrainingManager>();
        }

        if (sensorFootprint == null)
        {
            sensorFootprint = GetComponent<DroneTrackingSensorFootprint>();
        }

        if (trackedTruck == null && !runtimeTrackingMode)
        {
            trackedTruck = trainingManager != null
                ? trainingManager.TruckTarget
                : FindFirstObjectByType<TruckTarget>();
        }

        if (sensorOrigin == null || (sensorOrigin != transform && !sensorOrigin.IsChildOf(transform)))
        {
            sensorOrigin = null;
            if (sensorFootprint != null && sensorFootprint.CoverageCamera != null)
            {
                sensorOrigin = sensorFootprint.CoverageCamera.transform;
            }
            else
            {
                Camera childCamera = GetComponentInChildren<Camera>(true);
                sensorOrigin = childCamera != null ? childCamera.transform : transform;
            }
        }
    }

    public void RefreshSensorReferences()
    {
        sensorOrigin = null;
        ResolveReferences();
    }

    private bool IsTargetInsideSensorView(Vector3 targetPoint, float fallbackForwardDot)
    {
        if (sensorFootprint != null)
        {
            return sensorFootprint.IsPointInView(targetPoint);
        }

        return fallbackForwardDot >= Mathf.Cos(viewHalfAngle * Mathf.Deg2Rad);
    }

    private bool HasLineOfSight(Vector3 targetPoint)
    {
        if (trackedTruck == null)
        {
            return false;
        }

        Vector3 origin = GetSensorOrigin();
        Vector3 toTarget = targetPoint - origin;
        float distance = toTarget.magnitude;
        if (distance <= 0.001f)
        {
            return true;
        }

        Vector3 direction = toTarget / distance;
        origin += direction * 0.5f;

        if (!Physics.Raycast(origin, direction, out RaycastHit hitInfo, distance + 0.1f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
        {
            return false;
        }

        return hitInfo.transform == trackedTruck.transform || hitInfo.transform.IsChildOf(trackedTruck.transform);
    }

    private float AddViewportCenteringReward(Vector3 targetPoint)
    {
        if (!TryGetTargetViewportPoint(targetPoint, out Vector3 viewportPoint))
        {
            return 0f;
        }

        float horizontalOffset01 = Mathf.Clamp01(Mathf.Abs(viewportPoint.x - preferredViewportX) / 0.5f);
        float verticalOffset01 = Mathf.Clamp01(Mathf.Abs(viewportPoint.y - preferredViewportY) / 0.5f);

        float horizontalReward01 = 1f - horizontalOffset01;
        float verticalReward01 = 1f - verticalOffset01;
        float centeredness01 = 1f - Mathf.Clamp01(
            Mathf.Sqrt((horizontalOffset01 * horizontalOffset01) + (verticalOffset01 * verticalOffset01)) / Mathf.Sqrt(2f));

        AddReward(horizontalReward01 * horizontalReward01 * viewportHorizontalRewardScale);
        AddReward(verticalReward01 * verticalReward01 * viewportVerticalRewardScale);
        AddReward(centeredness01 * centeredness01 * viewportCenterRewardScale);

        float comfortRadius01 = Mathf.Clamp01(viewportCenterComfortRadius / 0.5f);
        if (centeredness01 < 1f - comfortRadius01)
        {
            float offCenterPenalty01 = Mathf.InverseLerp(1f - comfortRadius01, 0f, centeredness01);
            AddReward(-offCenterPenalty01 * viewportOffCenterPenaltyScale);
        }

        float minEdgeDistance = Mathf.Min(
            Mathf.Min(viewportPoint.x, 1f - viewportPoint.x),
            Mathf.Min(viewportPoint.y, 1f - viewportPoint.y));

        if (minEdgeDistance < edgeComfortMargin)
        {
            float edgePenalty01 = 1f - Mathf.Clamp01(minEdgeDistance / edgeComfortMargin);
            AddReward(-edgePenalty01 * viewportEdgePenaltyScale);
        }

        return centeredness01;
    }

    private float GetNormalizedTimeSinceLastVisibleObservation()
    {
        return hasTrackedTarget
            ? Mathf.Clamp01(timeSinceLastVisible / Mathf.Max(0.01f, maxTimeWithoutSight))
            : 1f;
    }

    private float GetLostTargetReleaseThreshold()
    {
        return runtimeTrackingMode
            ? maxTimeWithoutSight + Mathf.Max(0f, runtimeClaimReleaseGraceTime)
            : maxTimeWithoutSight;
    }

    private bool TryGetObservedTargetState(
        bool visible,
        Vector3 liveTrackingPoint,
        Vector3 liveTargetVelocity,
        out Vector3 observedTrackingPoint,
        out Vector3 observedTargetVelocity)
    {
        if (visible)
        {
            observedTrackingPoint = liveTrackingPoint;
            observedTargetVelocity = ProjectToXZ(liveTargetVelocity);
            return true;
        }

        if (!hasTrackedTarget)
        {
            observedTrackingPoint = default;
            observedTargetVelocity = Vector3.zero;
            return false;
        }

        float predictionTime = Mathf.Min(timeSinceLastVisible, maxTimeWithoutSight);
        observedTargetVelocity = lastKnownTargetVelocity;
        observedTrackingPoint = lastKnownTrackingPoint + (observedTargetVelocity * predictionTime);
        return true;
    }

    private void UpdateTargetMemory(bool visible, Vector3 trackingPoint, Vector3 targetVelocity)
    {
        if (!visible)
        {
            return;
        }

        hasTrackedTarget = true;
        lastKnownTrackingPoint = trackingPoint;
        lastKnownTargetVelocity = ProjectToXZ(targetVelocity);
    }

    private bool TryGetTargetViewportPoint(Vector3 targetPoint, out Vector3 viewportPoint)
    {
        Camera sensorCamera = sensorFootprint != null ? sensorFootprint.CoverageCamera : null;
        if (sensorCamera == null && sensorOrigin != null)
        {
            sensorCamera = sensorOrigin.GetComponent<Camera>();
        }

        if (sensorCamera == null)
        {
            viewportPoint = default;
            return false;
        }

        viewportPoint = sensorCamera.WorldToViewportPoint(targetPoint);
        return viewportPoint.z > 0f;
    }

    private Vector3 GetSensorOrigin()
    {
        if (sensorFootprint != null)
        {
            return sensorFootprint.SensorWorldPosition;
        }

        return sensorOrigin != null
            ? sensorOrigin.position
            : transform.position + Vector3.up;
    }

    private static Vector3 ProjectToXZ(Vector3 value)
    {
        return new Vector3(value.x, 0f, value.z);
    }

    private float ApplyActionDeadzone(float value)
    {
        return Mathf.Abs(value) < actionDeadzone ? 0f : value;
    }

    private void ApplyRuntimeTrackingAssist(
        ref float forward,
        ref float turn,
        Vector3 observedTrackingPoint,
        Vector3 observedTargetVelocity,
        bool visible)
    {
        if (!runtimeTrackingMode)
        {
            return;
        }

        Vector3 flatToObservedTarget = new Vector3(
            observedTrackingPoint.x - transform.position.x,
            0f,
            observedTrackingPoint.z - transform.position.z);
        if (flatToObservedTarget.sqrMagnitude <= 0.0001f)
        {
            return;
        }

        Vector3 flatForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
        if (flatForward.sqrMagnitude < 0.0001f)
        {
            flatForward = Vector3.forward;
        }

        float signedAngle = Vector3.SignedAngle(flatForward, flatToObservedTarget.normalized, Vector3.up);
        float assistTurn = Mathf.Clamp(signedAngle / Mathf.Max(10f, runtimeAssistFullTurnAngle), -1f, 1f);
        float observedDistance = flatToObservedTarget.magnitude;
        float distanceGap = observedDistance - idealTrackingDistance;

        if (!runtimeHasDirectSight && runtimeTrackingElapsedTime <= runtimeInitialAssistDuration)
        {
            turn = Mathf.Abs(turn) > Mathf.Abs(assistTurn) ? turn : assistTurn;

            float assistedForwardFloor = Mathf.Abs(signedAngle) > 100f
                ? 0.15f
                : runtimeInitialAssistForward;
            forward = Mathf.Max(forward, assistedForwardFloor);
        }
        else if (!visible && hasTrackedTarget)
        {
            float reacquireWindow01 = 1f - Mathf.Clamp01(timeSinceLastVisible / Mathf.Max(0.01f, maxTimeWithoutSight));
            float distanceRecovery01 = Mathf.Clamp01(
                (observedDistance - (idealTrackingDistance * 0.5f)) /
                Mathf.Max(10f, maxTrackingDistance - (idealTrackingDistance * 0.5f)));
            float assistedForwardFloor = Mathf.Max(
                runtimeLostSightMinForward,
                runtimeLostSightAssistForward * Mathf.Max(0.35f, reacquireWindow01));
            assistedForwardFloor = Mathf.Lerp(assistedForwardFloor, runtimeLostSightAssistForward, distanceRecovery01);
            if (Mathf.Abs(signedAngle) > 100f)
            {
                assistedForwardFloor *= 0.55f;
            }

            turn = Mathf.Lerp(turn, assistTurn, runtimeTurnAssistBlend);
            forward = Mathf.Max(forward, assistedForwardFloor);
        }
        else if (visible && distanceGap > 0f)
        {
            float distanceGap01 = Mathf.Clamp01(distanceGap / Mathf.Max(10f, maxTrackingDistance - idealTrackingDistance));
            float visibleForwardFloor = runtimeVisiblePursuitForward * distanceGap01;
            if (Mathf.Abs(signedAngle) > 100f)
            {
                visibleForwardFloor *= 0.3f;
            }

            float visibleTurnBlend = runtimeVisibleTurnAssistBlend * Mathf.Clamp01(Mathf.Abs(signedAngle) / 60f);
            turn = Mathf.Lerp(turn, assistTurn, visibleTurnBlend);
            forward = Mathf.Max(forward, visibleForwardFloor);
        }
        else if (visible)
        {
            float targetSpeed01 = Mathf.Clamp01(observedTargetVelocity.magnitude / Mathf.Max(1f, maxExpectedTargetSpeed));
            float visibleForwardFloor = Mathf.Lerp(runtimeVisibleMinForward * 0.75f, runtimeVisibleMinForward, targetSpeed01);

            if (observedDistance > Mathf.Max(12f, idealTrackingDistance * 0.32f) && Mathf.Abs(signedAngle) < 75f)
            {
                forward = Mathf.Max(forward, visibleForwardFloor);
            }
        }

        if (forwardYawOnly)
        {
            forward = Mathf.Clamp01(forward);
        }
        else
        {
            forward = Mathf.Clamp(forward, -1f, 1f);
        }

        turn = Mathf.Clamp(turn, -1f, 1f);
    }

    private void ResetTrackingState(bool clearTrackedTruck)
    {
        collisionDetected = false;
        hasTrackedTarget = false;
        runtimeHasDirectSight = false;
        lastKnownTrackingPoint = Vector3.zero;
        lastKnownTargetVelocity = Vector3.zero;
        runtimeTrackingElapsedTime = 0f;
        timeSinceLastVisible = 0f;
        stableAcquireTimer = 0f;
        smoothedAction = Vector3.zero;
        debugLastVisible = false;
        debugLastDistanceToTarget = 0f;
        debugLastObservedDistance = 0f;
        debugLastForwardCommand = 0f;
        debugLastTurnCommand = 0f;
        debugLastStrafeCommand = 0f;

        if (clearTrackedTruck)
        {
            trackedTruck = null;
        }
    }

    private float DistanceOutsideBounds(Vector3 worldPosition)
    {
        if (runtimeTrackingMode)
        {
            if (!hasRuntimeMissionBounds)
            {
                return 0f;
            }

            Bounds bounds = BuildRuntimeMissionBounds();
            float dx = Mathf.Max(Mathf.Max(bounds.min.x - worldPosition.x, 0f), worldPosition.x - bounds.max.x);
            float dz = Mathf.Max(Mathf.Max(bounds.min.z - worldPosition.z, 0f), worldPosition.z - bounds.max.z);
            return Mathf.Sqrt((dx * dx) + (dz * dz));
        }

        return trainingManager != null
            ? trainingManager.DistanceOutsideEpisodeBounds(worldPosition)
            : 0f;
    }

    private Vector4 GetNormalizedDistancesToBounds(Vector3 worldPosition)
    {
        if (runtimeTrackingMode)
        {
            if (!hasRuntimeMissionBounds)
            {
                return Vector4.one * 0.5f;
            }

            Bounds bounds = BuildRuntimeMissionBounds();
            float width = Mathf.Max(0.001f, bounds.size.x);
            float depth = Mathf.Max(0.001f, bounds.size.z);

            float toMinX = Mathf.Clamp01((worldPosition.x - bounds.min.x) / width);
            float toMaxX = Mathf.Clamp01((bounds.max.x - worldPosition.x) / width);
            float toMinZ = Mathf.Clamp01((worldPosition.z - bounds.min.z) / depth);
            float toMaxZ = Mathf.Clamp01((bounds.max.z - worldPosition.z) / depth);

            return new Vector4(toMinX, toMaxX, toMinZ, toMaxZ);
        }

        return trainingManager != null
            ? trainingManager.GetNormalizedDistancesToBounds(worldPosition)
            : Vector4.one * 0.5f;
    }

    private Bounds BuildRuntimeMissionBounds()
    {
        return new Bounds(
            new Vector3(runtimeMissionCenterXZ.x, transform.position.y, runtimeMissionCenterXZ.y),
            new Vector3(runtimeMissionSizeXZ.x, 0f, runtimeMissionSizeXZ.y));
    }

    private void NotifyRuntimeTrackingEnded(RuntimeTrackingOutcome outcome)
    {
        if (!runtimeTrackingMode || !runtimeTrackingActive)
        {
            return;
        }

        runtimeTrackingActive = false;
        RuntimeTrackingEnded?.Invoke(this, outcome);
    }

    private void AddZeroObservations(VectorSensor sensor)
    {
        for (int i = 0; i < ExpectedVectorObservationSize; i++)
        {
            sensor.AddObservation(0f);
        }
    }

    private void ValidateObservationSizeConfiguration()
    {
        var behaviorParameters = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (behaviorParameters == null)
        {
            return;
        }

        int configuredSize = behaviorParameters.BrainParameters.VectorObservationSize;
        if (configuredSize != ExpectedVectorObservationSize)
        {
            Debug.LogWarning(
                $"DroneTrackingAgent expects VectorObservationSize {ExpectedVectorObservationSize}, but BehaviorParameters is {configuredSize}.",
                this);
        }
    }

    private void OnDrawGizmosSelected()
    {
        if (sensorFootprint != null && sensorFootprint.TryGetGroundFootprint(out _, out _))
        {
            return;
        }

        Gizmos.color = new Color(1f, 0.8f, 0.2f, 0.8f);
        Gizmos.DrawWireSphere(transform.position, maxTrackingDistance);

        Vector3 origin = sensorOrigin != null ? sensorOrigin.position : transform.position;
        Quaternion leftRotation = Quaternion.AngleAxis(-viewHalfAngle, Vector3.up);
        Quaternion rightRotation = Quaternion.AngleAxis(viewHalfAngle, Vector3.up);

        Gizmos.color = new Color(0.3f, 1f, 0.4f, 0.9f);
        Gizmos.DrawRay(origin, leftRotation * transform.forward * maxTrackingDistance);
        Gizmos.DrawRay(origin, rightRotation * transform.forward * maxTrackingDistance);
    }
}
