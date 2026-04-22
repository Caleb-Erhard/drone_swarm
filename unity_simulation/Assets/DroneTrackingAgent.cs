using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(LockedAltitudeDroneController))]
[RequireComponent(typeof(Rigidbody))]
public class DroneTrackingAgent : Agent
{
    private const int ExpectedVectorObservationSize = 19;

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
    [SerializeField, Min(0f)] private float outOfBoundsGraceDistance = 6f;
    [SerializeField, Min(1f)] private float maxExpectedTargetSpeed = 16f;
    [SerializeField, Min(1f)] private float maxExpectedOwnSpeed = 18f;
    [SerializeField] private bool endEpisodeAfterStableAcquire = true;
    [SerializeField, Min(0.1f)] private float stableAcquireDuration = 3.5f;
    [SerializeField, Range(0f, 1f)] private float stableAcquireCenterednessThreshold = 0.72f;
    [SerializeField] private bool forwardYawOnly = true;

    [Header("Rewards")]
    [SerializeField] private float stepPenalty = 0.0005f;
    [SerializeField] private float visibleReward = 0.015f;
    [SerializeField] private float inRangeReward = 0.01f;
    [SerializeField] private float distanceRewardScale = 0.03f;
    [SerializeField] private float frontAlignmentReward = 0.02f;
    [SerializeField] private float keepUpRewardScale = 0.01f;
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

    private bool collisionDetected;
    private bool hasTrackedTarget;
    private bool originalManualInputState;
    private Vector3 lastKnownTrackingPoint;
    private Vector3 lastKnownTargetVelocity;
    private float timeSinceLastVisible;
    private float stableAcquireTimer;
    private Vector3 smoothedAction;
    private bool loggedEpisodeBegin;
    private bool loggedFirstObservation;
    private bool loggedFirstAction;

    public override void Initialize()
    {
        ResolveReferences();

        if (droneController != null)
        {
            originalManualInputState = droneController.manualInputEnabled;
            droneController.manualInputEnabled = false;
        }

        if (MaxStep <= 0)
        {
            MaxStep = 2000;
        }

        ValidateObservationSizeConfiguration();
    }

    public override void OnEpisodeBegin()
    {
        ResolveReferences();

        collisionDetected = false;
        hasTrackedTarget = false;
        lastKnownTrackingPoint = Vector3.zero;
        lastKnownTargetVelocity = Vector3.zero;
        timeSinceLastVisible = 0f;
        stableAcquireTimer = 0f;
        smoothedAction = Vector3.zero;

        if (trainingManager != null)
        {
            trainingManager.ResetEpisode();
        }

        if (!loggedEpisodeBegin)
        {
            Debug.LogWarning("DroneTrackingAgent.OnEpisodeBegin reached.", this);
            loggedEpisodeBegin = true;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        ResolveReferences();

        if (!loggedFirstObservation)
        {
            Debug.LogWarning("DroneTrackingAgent.CollectObservations reached.", this);
            loggedFirstObservation = true;
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

        Vector4 boundsDistances = trainingManager != null
            ? trainingManager.GetNormalizedDistancesToBounds(transform.position)
            : Vector4.one * 0.5f;

        sensor.AddObservation(boundsDistances.x);
        sensor.AddObservation(boundsDistances.y);
        sensor.AddObservation(boundsDistances.z);
        sensor.AddObservation(boundsDistances.w);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        ResolveReferences();

        if (!loggedFirstAction)
        {
            Debug.LogWarning("DroneTrackingAgent.OnActionReceived reached.", this);
            loggedFirstAction = true;
        }

        if (droneController == null || trackedTruck == null)
        {
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

        droneController.SetControlInputs(strafe, forward, turn);

        AddReward(-stepPenalty);

        if (rb != null)
        {
            AddReward(-Mathf.Clamp01(Mathf.Abs(rb.angularVelocity.y) / 6f) * yawPenalty);
        }

        Vector3 trackingPoint = trackedTruck.GetTrackingPoint();
        Vector3 targetVelocity = ProjectToXZ(trackedTruck.CurrentVelocity);
        Vector3 flatTrackingPoint = new Vector3(trackingPoint.x, transform.position.y, trackingPoint.z);
        Vector3 toTargetFlat = flatTrackingPoint - transform.position;
        float liveDistance = toTargetFlat.magnitude;
        float liveForwardDot = liveDistance > 0.001f
            ? Vector3.Dot(transform.forward, toTargetFlat / liveDistance)
            : 1f;

        bool inRange = liveDistance <= maxTrackingDistance;
        bool inViewCone = IsTargetInsideSensorView(trackingPoint, liveForwardDot);
        bool hasLineOfSight = HasLineOfSight(trackingPoint);
        bool visible = inRange && inViewCone && hasLineOfSight;
        float viewportCentering01 = 0f;

        if (visible)
        {
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

        if (hasObservedTarget)
        {
            Vector3 observedFlatTrackingPoint = new Vector3(observedTrackingPoint.x, transform.position.y, observedTrackingPoint.z);
            Vector3 toObservedTargetFlat = observedFlatTrackingPoint - transform.position;
            float observedDistance = toObservedTargetFlat.magnitude;
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
            EndEpisode();
            return;
        }

        if (trainingManager != null)
        {
            float outsideDistance = trainingManager.DistanceOutsideEpisodeBounds(transform.position);
            if (outsideDistance > outOfBoundsGraceDistance)
            {
                AddReward(-outOfBoundsPenalty);
                EndEpisode();
                return;
            }
        }

        if (hasTrackedTarget && visible && liveDistance > maxTrackingDistance + Mathf.Max(distanceTolerance, 10f))
        {
            AddReward(-lostTargetPenalty);
            EndEpisode();
            return;
        }

        if (hasTrackedTarget && timeSinceLastVisible > maxTimeWithoutSight)
        {
            AddReward(-lostTargetPenalty);
            EndEpisode();
            return;
        }

        if (endEpisodeAfterStableAcquire && stableAcquireTimer >= stableAcquireDuration)
        {
            AddReward(stableAcquireReward);
            EndEpisode();
            return;
        }

        if (MaxStep > 0 && StepCount >= MaxStep)
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

    protected override void OnDisable()
    {
        base.OnDisable();

        if (droneController != null)
        {
            droneController.ClearControlInputs();
            droneController.manualInputEnabled = originalManualInputState;
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

        if (trainingManager == null)
        {
            trainingManager = FindFirstObjectByType<DroneTrackingTrainingManager>();
        }

        if (sensorFootprint == null)
        {
            sensorFootprint = GetComponent<DroneTrackingSensorFootprint>();
        }

        if (trackedTruck == null)
        {
            trackedTruck = trainingManager != null
                ? trainingManager.TruckTarget
                : FindFirstObjectByType<TruckTarget>();
        }

        if (sensorOrigin == null)
        {
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
