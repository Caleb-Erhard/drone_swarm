using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

[RequireComponent(typeof(LockedAltitudeDroneController))]
[RequireComponent(typeof(Rigidbody))]
public class DroneCoverageAgent : Agent
{
    private const int LocalCoverageSamplesPerAxis = 5;
    private const int ExpectedVectorObservationSize = 57;

    [Header("References")]
    [SerializeField] private LockedAltitudeDroneController droneController;
    [SerializeField] private AreaCoverageTracker coverageTracker;

    [Header("Episode Settings")]
    [SerializeField] private float requiredCoverage = 1f;
    [SerializeField] private float maxDistanceOutsideZone = 3f;
    [SerializeField] private float spawnAltitudeOffset = 0f;
    [SerializeField, Min(200)] private int episodeStepLimit = 4000;
    [SerializeField] private bool logEpisodeEndReason = true;
    [SerializeField] private bool useFixedZoneSize = true;
    [SerializeField] private Vector2 fixedZoneSize = new Vector2(250f, 200f);
    [SerializeField] private bool randomizeZoneSizeEachEpisode = true;
    [SerializeField] private Vector2 zoneWidthRange = new Vector2(50f, 130f);
    [SerializeField] private Vector2 zoneDepthRange = new Vector2(50f, 130f);

    [Header("Curriculum")]
    [SerializeField] private bool useZoneSizeCurriculum = true;
    [SerializeField, Min(1)] private int curriculumRampEpisodes = 2500;
    [SerializeField] private Vector2 curriculumStartZoneWidthRange = new Vector2(50f, 95f);
    [SerializeField] private Vector2 curriculumStartZoneDepthRange = new Vector2(50f, 95f);
    [SerializeField] private Vector2 curriculumFinalZoneWidthRange = new Vector2(80f, 150f);
    [SerializeField] private Vector2 curriculumFinalZoneDepthRange = new Vector2(80f, 150f);

    [Header("Episode Budget Scaling")]
    [SerializeField] private bool scaleEpisodeStepLimitByZoneArea = true;
    [SerializeField, Min(1f)] private float referenceZoneWidth = 250f;
    [SerializeField, Min(1f)] private float referenceZoneDepth = 200f;
    [SerializeField, Min(200)] private int referenceEpisodeStepLimit = 7000;
    [SerializeField, Min(200)] private int minEpisodeStepLimit = 5000;
    [SerializeField, Min(200)] private int maxEpisodeStepLimit = 60000;
    [SerializeField, Range(0.5f, 1.5f)] private float areaStepScalingExponent = 1f;

    [Header("Spawn Settings")]
    [SerializeField] private bool spawnAtZoneEdge = true;
    [SerializeField, Min(0f)] private float edgeSpawnInset = 0.2f;
    [SerializeField, Min(0f)] private float edgeSpawnOutsideOffset = 0f;
    [SerializeField] private bool faceZoneCenterOnEdgeSpawn = true;

    [Header("Rewards")]
    [SerializeField] private float stepPenalty = 0.0001f;
    [SerializeField] private float newCoverageReward = 6f;
    [SerializeField] private float outsideZonePenalty = 0.01f;
    [SerializeField] private float collisionPenalty = 0.75f;
    [SerializeField] private float completionReward = 3f;
    [SerializeField] private float completionEfficiencyBonus = 10f;
    [SerializeField] private float timeoutPenalty = 0.5f;
    [SerializeField] private float revisitingPenalty = 0f;
    [SerializeField] private float overlapPenalty = 0.0012f;
    [SerializeField] private float turnInputPenalty = 0f;
    [SerializeField] private float yawRatePenalty = 0.0005f;
    [SerializeField] private float overlapTurnPenalty = 0f;
    [SerializeField] private float diagonalMotionPenalty = 0.001f;
    [SerializeField] private float frontierAlignmentRewardScale = 0f;
    [SerializeField] private float frontierDistanceRewardScale = 0f;
    [SerializeField, Min(0.1f)] private float targetExplorationSpeed = 8f;
    [SerializeField] private float explorationSpeedReward = 0.1f;
    [SerializeField] private float lowSpeedOverlapPenalty = 0f;
    [SerializeField, Range(0f, 1f)] private float actionSmoothing = 1f;
    [SerializeField, Range(0f, 0.5f)] private float actionDeadzone = 0.08f;
    [SerializeField] private float stationaryPenalty = 0f;
    [SerializeField, Min(0f)] private float stationaryDistanceThreshold = 0.08f;
    [SerializeField] private float movementReward = 0.001f;

    [Header("Observation Settings")]
    [SerializeField, Min(10f)] private float maxExpectedZoneDimension = 160f;
    [SerializeField, Min(0.1f)] private float localCoverageSampleSpacing = 0.9f;

    [Header("Heuristic Controls")]
    [SerializeField, Range(0.1f, 1f)] private float heuristicMoveScale = 0.65f;
    [SerializeField, Range(0.1f, 1f)] private float heuristicTurnScale = 0.45f;
    [SerializeField] private bool useHeuristicPrecisionModifier = true;
    [SerializeField] private KeyCode heuristicPrecisionKey = KeyCode.LeftShift;
    [SerializeField, Range(0.1f, 1f)] private float heuristicPrecisionScale = 0.4f;

    [Header("Action Constraints")]
    [SerializeField] private bool forwardYawOnly = true;

    private Rigidbody rb;
    private EnvironmentParameters environmentParameters;
    private bool collisionDetected;
    private bool originalManualInputState;
    private bool hasFrontierDistance;
    private float previousFrontierDistance01;
    private int episodeCounter;
    private Vector2 activeZoneWidthRange;
    private Vector2 activeZoneDepthRange;
    private float currentCurriculumProgress01;
    private Vector3 smoothedAction;
    private Vector2 previousPlanarPosition;

    private void OnValidate()
    {
        requiredCoverage = Mathf.Clamp01(requiredCoverage);
        maxExpectedZoneDimension = Mathf.Max(10f, maxExpectedZoneDimension);
        localCoverageSampleSpacing = Mathf.Max(0.1f, localCoverageSampleSpacing);
        episodeStepLimit = Mathf.Max(200, episodeStepLimit);
        edgeSpawnInset = Mathf.Max(0f, edgeSpawnInset);
        edgeSpawnOutsideOffset = Mathf.Max(0f, edgeSpawnOutsideOffset);
        completionEfficiencyBonus = Mathf.Max(0f, completionEfficiencyBonus);
        overlapPenalty = Mathf.Max(0f, overlapPenalty);
        turnInputPenalty = Mathf.Max(0f, turnInputPenalty);
        yawRatePenalty = Mathf.Max(0f, yawRatePenalty);
        overlapTurnPenalty = Mathf.Max(0f, overlapTurnPenalty);
        diagonalMotionPenalty = Mathf.Max(0f, diagonalMotionPenalty);
        frontierAlignmentRewardScale = Mathf.Max(0f, frontierAlignmentRewardScale);
        frontierDistanceRewardScale = Mathf.Max(0f, frontierDistanceRewardScale);
        targetExplorationSpeed = Mathf.Max(0.1f, targetExplorationSpeed);
        explorationSpeedReward = Mathf.Max(0f, explorationSpeedReward);
        lowSpeedOverlapPenalty = Mathf.Max(0f, lowSpeedOverlapPenalty);
        actionSmoothing = Mathf.Clamp01(actionSmoothing);
        actionDeadzone = Mathf.Clamp(actionDeadzone, 0f, 0.5f);
        stationaryPenalty = Mathf.Max(0f, stationaryPenalty);
        stationaryDistanceThreshold = Mathf.Max(0f, stationaryDistanceThreshold);
        movementReward = Mathf.Max(0f, movementReward);
        heuristicMoveScale = Mathf.Clamp(heuristicMoveScale, 0.1f, 1f);
        heuristicTurnScale = Mathf.Clamp(heuristicTurnScale, 0.1f, 1f);
        heuristicPrecisionScale = Mathf.Clamp(heuristicPrecisionScale, 0.1f, 1f);

        fixedZoneSize.x = Mathf.Max(2f, fixedZoneSize.x);
        fixedZoneSize.y = Mathf.Max(2f, fixedZoneSize.y);
        zoneWidthRange.x = Mathf.Max(2f, zoneWidthRange.x);
        zoneWidthRange.y = Mathf.Max(zoneWidthRange.x, zoneWidthRange.y);
        zoneDepthRange.x = Mathf.Max(2f, zoneDepthRange.x);
        zoneDepthRange.y = Mathf.Max(zoneDepthRange.x, zoneDepthRange.y);
        curriculumRampEpisodes = Mathf.Max(1, curriculumRampEpisodes);
        curriculumStartZoneWidthRange.x = Mathf.Max(2f, curriculumStartZoneWidthRange.x);
        curriculumStartZoneWidthRange.y = Mathf.Max(curriculumStartZoneWidthRange.x, curriculumStartZoneWidthRange.y);
        curriculumStartZoneDepthRange.x = Mathf.Max(2f, curriculumStartZoneDepthRange.x);
        curriculumStartZoneDepthRange.y = Mathf.Max(curriculumStartZoneDepthRange.x, curriculumStartZoneDepthRange.y);
        curriculumFinalZoneWidthRange.x = Mathf.Max(2f, curriculumFinalZoneWidthRange.x);
        curriculumFinalZoneWidthRange.y = Mathf.Max(curriculumFinalZoneWidthRange.x, curriculumFinalZoneWidthRange.y);
        curriculumFinalZoneDepthRange.x = Mathf.Max(2f, curriculumFinalZoneDepthRange.x);
        curriculumFinalZoneDepthRange.y = Mathf.Max(curriculumFinalZoneDepthRange.x, curriculumFinalZoneDepthRange.y);
        referenceZoneWidth = Mathf.Max(1f, referenceZoneWidth);
        referenceZoneDepth = Mathf.Max(1f, referenceZoneDepth);
        referenceEpisodeStepLimit = Mathf.Max(200, referenceEpisodeStepLimit);
        minEpisodeStepLimit = Mathf.Max(200, minEpisodeStepLimit);
        maxEpisodeStepLimit = Mathf.Max(minEpisodeStepLimit, maxEpisodeStepLimit);
        areaStepScalingExponent = Mathf.Clamp(areaStepScalingExponent, 0.5f, 1.5f);
    }

    public override void Initialize()
    {
        if (droneController == null)
        {
            droneController = GetComponent<LockedAltitudeDroneController>();
        }

        if (coverageTracker == null)
        {
            coverageTracker = FindFirstObjectByType<AreaCoverageTracker>();
        }

        if (coverageTracker != null)
        {
            coverageTracker.SetSensorTransform(transform);
        }

        environmentParameters = Academy.Instance.EnvironmentParameters;
        if (useFixedZoneSize)
        {
            activeZoneWidthRange = new Vector2(fixedZoneSize.x, fixedZoneSize.x);
            activeZoneDepthRange = new Vector2(fixedZoneSize.y, fixedZoneSize.y);
        }
        else
        {
            activeZoneWidthRange = zoneWidthRange;
            activeZoneDepthRange = zoneDepthRange;
        }

        currentCurriculumProgress01 = 1f;

        // Prevent infinite episodes when scene MaxStep is left at 0.
        if (MaxStep <= 0)
        {
            MaxStep = episodeStepLimit;
        }

        ValidateObservationSizeConfiguration();
        ValidateEpisodeStepLimitConfiguration();

        rb = GetComponent<Rigidbody>();
        originalManualInputState = droneController.manualInputEnabled;
        droneController.manualInputEnabled = false;
    }

    public override void OnEpisodeBegin()
    {
        if (coverageTracker == null || droneController == null)
        {
            Debug.LogError("DroneCoverageAgent is missing references.");
            EndEpisode();
            return;
        }

        UpdateEpisodeCurriculum();
        coverageTracker.SetSensorTransform(transform);
        Bounds currentZone = coverageTracker.ZoneBounds;
        Vector2 centerXZ = new Vector2(currentZone.center.x, currentZone.center.z);
        if (useFixedZoneSize)
        {
            coverageTracker.ConfigureSearchZone(centerXZ, fixedZoneSize);
        }
        else if (randomizeZoneSizeEachEpisode)
        {
            float zoneWidth = Random.Range(activeZoneWidthRange.x, activeZoneWidthRange.y);
            float zoneDepth = Random.Range(activeZoneDepthRange.x, activeZoneDepthRange.y);
            coverageTracker.ConfigureSearchZone(centerXZ, new Vector2(zoneWidth, zoneDepth));
        }

        UpdateEpisodeStepLimitFromZoneSize();
        coverageTracker.ResetCoverage();
        ResetDrone();
        collisionDetected = false;
        ResetFrontierDistanceCache();
        smoothedAction = Vector3.zero;

        // Count initial footprint at spawn location.
        coverageTracker.MarkCoverage(transform.position);
        CacheFrontierDistance();
        previousPlanarPosition = new Vector2(transform.position.x, transform.position.z);
        Academy.Instance.StatsRecorder.Add("Coverage/CurriculumProgress01", currentCurriculumProgress01);
        Academy.Instance.StatsRecorder.Add("Coverage/ActiveZoneWidthMax", activeZoneWidthRange.y);
        Academy.Instance.StatsRecorder.Add("Coverage/ActiveZoneDepthMax", activeZoneDepthRange.y);
        episodeCounter++;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector2 normalizedPos = coverageTracker.GetNormalizedZonePosition(transform.position);
        sensor.AddObservation(normalizedPos.x);
        sensor.AddObservation(normalizedPos.y);

        Vector3 localVelocity = transform.InverseTransformDirection(rb.linearVelocity);
        float speedNorm = Mathf.Max(1f, droneController.MaxSpeed);
        sensor.AddObservation(localVelocity.x / speedNorm);
        sensor.AddObservation(localVelocity.z / speedNorm);
        sensor.AddObservation(Mathf.Clamp(localVelocity.y / 10f, -1f, 1f));

        sensor.AddObservation(transform.forward.x);
        sensor.AddObservation(transform.forward.z);

        float coverage = coverageTracker.Coverage01;
        sensor.AddObservation(coverage);
        sensor.AddObservation(Mathf.Clamp01(requiredCoverage - coverage));

        float zoneScale = Mathf.Max(1f, maxExpectedZoneDimension);
        Vector2 zoneSize = coverageTracker.ZoneSizeXZ;
        sensor.AddObservation(Mathf.Clamp01(zoneSize.x / zoneScale));
        sensor.AddObservation(Mathf.Clamp01(zoneSize.y / zoneScale));
        sensor.AddObservation(Mathf.Clamp01(coverageTracker.SensorRadius / zoneScale));

        Vector4 boundsDistances = coverageTracker.GetNormalizedDistancesToBounds(transform.position);
        sensor.AddObservation(boundsDistances.x);
        sensor.AddObservation(boundsDistances.y);
        sensor.AddObservation(boundsDistances.z);
        sensor.AddObservation(boundsDistances.w);

        AddSensorFootprintCornerObservations(sensor);
        AddLocalCoverageObservations(sensor);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var action = actions.ContinuousActions;

        Vector3 rawAction = new Vector3(
            Mathf.Clamp(action[0], -1f, 1f),
            Mathf.Clamp(action[1], -1f, 1f),
            Mathf.Clamp(action[2], -1f, 1f));

        smoothedAction = Vector3.Lerp(smoothedAction, rawAction, actionSmoothing);
        float strafe = ApplyActionDeadzone(smoothedAction.x);
        float forward = ApplyActionDeadzone(smoothedAction.y);
        float turn = ApplyActionDeadzone(smoothedAction.z);

        if (forwardYawOnly)
        {
            // Keep policy aligned with expert demos: only forward throttle + yaw.
            strafe = 0f;
            forward = Mathf.Max(0f, forward);
        }

        droneController.SetControlInputs(strafe, forward, turn);

        AddReward(-stepPenalty);
        AddReward(-Mathf.Abs(turn) * turnInputPenalty);
        AddReward(-Mathf.Clamp01(Mathf.Abs(rb.angularVelocity.y) / 6f) * yawRatePenalty);

        AreaCoverageTracker.CoverageUpdate coverageUpdate = coverageTracker.MarkCoverageDetailed(transform.position);
        float newlyCoveredFraction = coverageUpdate.NewlyCoveredZoneFraction;
        bool insideZone = coverageTracker.IsInsideZone(transform.position);
        Vector2 currentPlanarPosition = new Vector2(transform.position.x, transform.position.z);
        float planarStepDistance = Vector2.Distance(previousPlanarPosition, currentPlanarPosition);

        if (newlyCoveredFraction > 0f)
        {
            AddReward(newlyCoveredFraction * newCoverageReward);
        }

        if (insideZone)
        {
            AddReward(-coverageUpdate.FootprintOverlap01 * overlapPenalty);
            AddReward(-coverageUpdate.FootprintOverlap01 * Mathf.Abs(turn) * overlapTurnPenalty);
            if (coverageUpdate.NewlyCoveredCellCount == 0)
            {
                AddReward(-revisitingPenalty);
            }

            float horizontalSpeed = new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z).magnitude;
            float speed01 = Mathf.Clamp01(horizontalSpeed / Mathf.Max(0.1f, droneController.MaxSpeed));
            AddReward(speed01 * 0.005f);

            // Discourage diagonal traversals so policy prefers row/column sweep lines.
            Vector3 localHorizontalVelocity = transform.InverseTransformDirection(rb.linearVelocity);
            float absLocalVelX = Mathf.Abs(localHorizontalVelocity.x);
            float absLocalVelZ = Mathf.Abs(localHorizontalVelocity.z);
            float diagonal01 = Mathf.Clamp01(Mathf.Min(absLocalVelX, absLocalVelZ) / Mathf.Max(0.1f, targetExplorationSpeed));
            AddReward(-diagonal01 * diagonalMotionPenalty);

            if (coverageUpdate.FootprintOverlap01 > 0.7f && speed01 < 0.65f)
            {
                float lowSpeedFactor = (0.65f - speed01) / 0.65f;
                AddReward(-lowSpeedFactor * lowSpeedOverlapPenalty);
            }

            if (planarStepDistance < stationaryDistanceThreshold)
            {
                AddReward(-stationaryPenalty);
            }
            else
            {
                float expectedStepDistance = targetExplorationSpeed * Time.fixedDeltaTime;
                float distanceReward01 = expectedStepDistance > 0.0001f
                    ? Mathf.Clamp01(planarStepDistance / expectedStepDistance)
                    : 0f;
                AddReward(distanceReward01 * movementReward);
            }

            Academy.Instance.StatsRecorder.Add("Coverage/Speed", horizontalSpeed);
            Academy.Instance.StatsRecorder.Add("Coverage/SmoothedTurnAbs", Mathf.Abs(turn));
            Academy.Instance.StatsRecorder.Add("Coverage/PlanarStepDistance", planarStepDistance);
            Academy.Instance.StatsRecorder.Add("Coverage/DiagonalMotion01", diagonal01);
        }

        Academy.Instance.StatsRecorder.Add("Coverage/Ratio", coverageTracker.Coverage01);
        Academy.Instance.StatsRecorder.Add("Coverage/NewFraction", newlyCoveredFraction);
        Academy.Instance.StatsRecorder.Add("Coverage/FootprintOverlap01", coverageUpdate.FootprintOverlap01);
        Academy.Instance.StatsRecorder.Add("Coverage/FootprintNovelty01", coverageUpdate.FootprintNovelty01);
        Academy.Instance.StatsRecorder.Add("Coverage/StepCount", StepCount);

        if (!insideZone)
        {
            ResetFrontierDistanceCache();
            AddReward(-outsideZonePenalty);
            float outsideDistance = coverageTracker.DistanceOutsideZone(transform.position);
            Academy.Instance.StatsRecorder.Add("Coverage/OutsideDistance", outsideDistance);

            if (outsideDistance > maxDistanceOutsideZone)
            {
                AddReward(-collisionPenalty);
                EndEpisodeWithOutcome("OutOfBounds", outsideDistance);
                return;
            }
        }

        previousPlanarPosition = currentPlanarPosition;

        if (collisionDetected)
        {
            AddReward(-collisionPenalty);
            EndEpisodeWithOutcome("Collision");
            return;
        }

        if (coverageTracker.Coverage01 >= requiredCoverage)
        {
            float efficiencyBonus = Mathf.Clamp01(1f - ((float)StepCount / Mathf.Max(1, MaxStep)));
            AddReward(completionReward + (efficiencyBonus * completionEfficiencyBonus));
            EndEpisodeWithOutcome("Completed");
            return;
        }

        if (StepCount >= MaxStep)
        {
            AddReward(-timeoutPenalty);
            Academy.Instance.StatsRecorder.Add("Coverage/TimedOutEpisodes", 1f);
            EndEpisodeWithOutcome("Timeout");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var action = actionsOut.ContinuousActions;
        float precisionScale = 1f;
        if (useHeuristicPrecisionModifier && Input.GetKey(heuristicPrecisionKey))
        {
            precisionScale = heuristicPrecisionScale;
        }

        float strafe = Input.GetAxisRaw("Horizontal");
        float forward = Input.GetAxisRaw("Vertical");
        float turn = Input.GetKey(KeyCode.Q) ? -1f : (Input.GetKey(KeyCode.E) ? 1f : 0f);

        if (forwardYawOnly)
        {
            strafe = 0f;
            forward = Mathf.Max(0f, forward);
        }

        action[0] = Mathf.Clamp(strafe * heuristicMoveScale * precisionScale, -1f, 1f);
        action[1] = Mathf.Clamp(forward * heuristicMoveScale * precisionScale, -1f, 1f);
        action[2] = Mathf.Clamp(turn * heuristicTurnScale * precisionScale, -1f, 1f);
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
                $"DroneCoverageAgent expects VectorObservationSize {ExpectedVectorObservationSize}, but BehaviorParameters is {configuredSize}.",
                this);
        }
    }

    private void ValidateEpisodeStepLimitConfiguration()
    {
        if (!scaleEpisodeStepLimitByZoneArea && MaxStep > 0 && MaxStep != episodeStepLimit)
        {
            Debug.LogWarning(
                $"DroneCoverageAgent MaxStep is {MaxStep}, but Episode Step Limit is {episodeStepLimit}. " +
                "Using MaxStep for timeout behavior.",
                this);
        }
    }

    private void UpdateEpisodeStepLimitFromZoneSize()
    {
        int effectiveStepLimit = episodeStepLimit;
        float zoneArea = 0f;

        if (coverageTracker != null)
        {
            Vector2 zoneSize = coverageTracker.ZoneSizeXZ;
            zoneArea = Mathf.Max(1f, zoneSize.x * zoneSize.y);

            if (scaleEpisodeStepLimitByZoneArea)
            {
                float referenceArea = Mathf.Max(1f, referenceZoneWidth * referenceZoneDepth);
                float areaRatio = zoneArea / referenceArea;
                float scaled = referenceEpisodeStepLimit * Mathf.Pow(areaRatio, areaStepScalingExponent);
                effectiveStepLimit = Mathf.RoundToInt(scaled);
            }
        }

        if (scaleEpisodeStepLimitByZoneArea)
        {
            effectiveStepLimit = Mathf.Clamp(effectiveStepLimit, minEpisodeStepLimit, maxEpisodeStepLimit);
        }

        MaxStep = Mathf.Max(200, effectiveStepLimit);

        Academy.Instance.StatsRecorder.Add("Coverage/ActiveMaxStep", MaxStep);
        Academy.Instance.StatsRecorder.Add("Coverage/ActiveZoneArea", zoneArea);
    }

    private void UpdateEpisodeCurriculum()
    {
        if (useFixedZoneSize)
        {
            activeZoneWidthRange = new Vector2(fixedZoneSize.x, fixedZoneSize.x);
            activeZoneDepthRange = new Vector2(fixedZoneSize.y, fixedZoneSize.y);
            currentCurriculumProgress01 = 1f;
            return;
        }

        float progressFromEpisodes = useZoneSizeCurriculum && curriculumRampEpisodes > 0
            ? Mathf.Clamp01((float)episodeCounter / curriculumRampEpisodes)
            : 1f;

        float progress = progressFromEpisodes;
        if (environmentParameters != null)
        {
            float difficulty = environmentParameters.GetWithDefault("difficulty", -1f);
            if (difficulty >= 0f)
            {
                progress = Mathf.Clamp01(difficulty);
            }
        }

        if (useZoneSizeCurriculum)
        {
            activeZoneWidthRange = Vector2.Lerp(curriculumStartZoneWidthRange, curriculumFinalZoneWidthRange, progress);
            activeZoneDepthRange = Vector2.Lerp(curriculumStartZoneDepthRange, curriculumFinalZoneDepthRange, progress);
        }
        else
        {
            activeZoneWidthRange = zoneWidthRange;
            activeZoneDepthRange = zoneDepthRange;
        }

        if (environmentParameters != null)
        {
            float envMinWidth = environmentParameters.GetWithDefault("zone_width_min", -1f);
            float envMaxWidth = environmentParameters.GetWithDefault("zone_width_max", -1f);
            if (envMinWidth > 0f && envMaxWidth >= envMinWidth)
            {
                activeZoneWidthRange = new Vector2(envMinWidth, envMaxWidth);
            }

            float envMinDepth = environmentParameters.GetWithDefault("zone_depth_min", -1f);
            float envMaxDepth = environmentParameters.GetWithDefault("zone_depth_max", -1f);
            if (envMinDepth > 0f && envMaxDepth >= envMinDepth)
            {
                activeZoneDepthRange = new Vector2(envMinDepth, envMaxDepth);
            }
        }

        currentCurriculumProgress01 = progress;
    }

    private void ApplyFrontierShaping()
    {
        if (!coverageTracker.TryGetNearestUnvisitedDirection(transform.position, out Vector3 directionToFrontier, out float frontierDistance01))
        {
            ResetFrontierDistanceCache();
            return;
        }

        Vector3 flatForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
        float alignment = flatForward.sqrMagnitude > 0.0001f ? Vector3.Dot(flatForward, directionToFrontier) : 0f;
        AddReward(alignment * frontierAlignmentRewardScale);

        if (hasFrontierDistance)
        {
            float frontierDistanceDelta = previousFrontierDistance01 - frontierDistance01;
            AddReward(frontierDistanceDelta * frontierDistanceRewardScale);
            Academy.Instance.StatsRecorder.Add("Coverage/FrontierDistanceDelta", frontierDistanceDelta);
        }

        previousFrontierDistance01 = frontierDistance01;
        hasFrontierDistance = true;
        Academy.Instance.StatsRecorder.Add("Coverage/FrontierDistance01", frontierDistance01);
        Academy.Instance.StatsRecorder.Add("Coverage/FrontierAlignment", alignment);
    }

    private void CacheFrontierDistance()
    {
        if (coverageTracker.TryGetNearestUnvisitedDirection(transform.position, out _, out float frontierDistance01))
        {
            previousFrontierDistance01 = frontierDistance01;
            hasFrontierDistance = true;
            return;
        }

        ResetFrontierDistanceCache();
    }

    private void ResetFrontierDistanceCache()
    {
        hasFrontierDistance = false;
        previousFrontierDistance01 = 0f;
    }

    private float ApplyActionDeadzone(float value)
    {
        if (Mathf.Abs(value) < actionDeadzone)
        {
            return 0f;
        }

        return value;
    }

    private void RecordEpisodeOutcome(string outcome)
    {
        Academy.Instance.StatsRecorder.Add("Coverage/EpisodeEndCoverage01", coverageTracker.Coverage01);
        Academy.Instance.StatsRecorder.Add($"Coverage/EpisodeResult/{outcome}", 1f);
    }

    private void EndEpisodeWithOutcome(string outcome, float outsideDistance = -1f)
    {
        RecordEpisodeOutcome(outcome);

        if (logEpisodeEndReason)
        {
            string details = $"[DroneCoverageAgent] Episode ended: {outcome}. " +
                             $"Coverage={coverageTracker.Coverage01:F3}, Step={StepCount}, MaxStep={MaxStep}, RequiredCoverage={requiredCoverage:F3}";
            if (outsideDistance >= 0f)
            {
                details += $", OutsideDistance={outsideDistance:F3}, MaxOutsideDistance={maxDistanceOutsideZone:F3}";
            }

            Debug.Log(details, this);
        }

        EndEpisode();
    }

    private void AddSensorFootprintCornerObservations(VectorSensor sensor)
    {
        Vector3[] corners = coverageTracker.GetSensorFootprintCorners();
        if (corners != null && corners.Length == 4)
        {
            foreach (Vector3 corner in corners)
            {
                Vector4 cornerBounds = coverageTracker.GetNormalizedDistancesToBounds(corner);
                sensor.AddObservation(cornerBounds.x);
                sensor.AddObservation(cornerBounds.y);
                sensor.AddObservation(cornerBounds.z);
                sensor.AddObservation(cornerBounds.w);
            }

            return;
        }

        for (int i = 0; i < 16; i++)
        {
            sensor.AddObservation(0f);
        }
    }

    private void AddLocalCoverageObservations(VectorSensor sensor)
    {
        int halfSamples = LocalCoverageSamplesPerAxis / 2;
        float sampleSpacing = Mathf.Max(0.1f, coverageTracker.SensorRadius * localCoverageSampleSpacing);

        for (int z = -halfSamples; z <= halfSamples; z++)
        {
            for (int x = -halfSamples; x <= halfSamples; x++)
            {
                Vector3 samplePosition = transform.position +
                                         (transform.right * (x * sampleSpacing)) +
                                         (transform.forward * (z * sampleSpacing));
                sensor.AddObservation(coverageTracker.GetCoverageSample(samplePosition));
            }
        }
    }

    private void ResetDrone()
    {
        Bounds zone = coverageTracker.ZoneBounds;

        Vector3 spawnPosition;
        if (spawnAtZoneEdge)
        {
            spawnPosition = GetEdgeSpawnPosition(zone);
        }
        else
        {
            spawnPosition = new Vector3(
                Random.Range(zone.min.x, zone.max.x),
                zone.max.y + droneController.targetAltitude + spawnAltitudeOffset,
                Random.Range(zone.min.z, zone.max.z));
        }

        transform.position = spawnPosition;

        if (spawnAtZoneEdge && faceZoneCenterOnEdgeSpawn)
        {
            Vector3 toCenter = zone.center - spawnPosition;
            toCenter.y = 0f;
            if (toCenter.sqrMagnitude > 0.0001f)
            {
                transform.rotation = Quaternion.LookRotation(toCenter.normalized, Vector3.up);
            }
            else
            {
                transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
            }
        }
        else
        {
            transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);
        }

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        droneController.ClearControlInputs();
    }

    private Vector3 GetEdgeSpawnPosition(Bounds zone)
    {
        float minX = zone.min.x;
        float maxX = zone.max.x;
        float minZ = zone.min.z;
        float maxZ = zone.max.z;

        float xInset = Mathf.Min(edgeSpawnInset, Mathf.Max(0f, (maxX - minX) * 0.5f - 0.01f));
        float zInset = Mathf.Min(edgeSpawnInset, Mathf.Max(0f, (maxZ - minZ) * 0.5f - 0.01f));

        float sampleX = Random.Range(minX + xInset, maxX - xInset);
        float sampleZ = Random.Range(minZ + zInset, maxZ - zInset);

        int edge = Random.Range(0, 4);
        switch (edge)
        {
            case 0: // left
                sampleX = (minX + xInset) - edgeSpawnOutsideOffset;
                break;
            case 1: // right
                sampleX = (maxX - xInset) + edgeSpawnOutsideOffset;
                break;
            case 2: // bottom
                sampleZ = (minZ + zInset) - edgeSpawnOutsideOffset;
                break;
            default: // top
                sampleZ = (maxZ - zInset) + edgeSpawnOutsideOffset;
                break;
        }

        return new Vector3(
            sampleX,
            zone.max.y + droneController.targetAltitude + spawnAltitudeOffset,
            sampleZ);
    }
}
