using System.Collections.Generic;
using UnityEngine;

public readonly struct DroneDemoZone
{
    public readonly int ZoneId;
    public readonly Vector2 CenterXZ;
    public readonly Vector2 SizeXZ;

    public DroneDemoZone(int zoneId, Vector2 centerXZ, Vector2 sizeXZ)
    {
        ZoneId = zoneId;
        CenterXZ = centerXZ;
        SizeXZ = sizeXZ;
    }
}

[RequireComponent(typeof(DroneCoverageAgent))]
[RequireComponent(typeof(LockedAltitudeDroneController))]
public class DroneDemoMissionController : MonoBehaviour
{
    private enum ApproachEdge
    {
        Left,
        Right,
        Bottom,
        Top
    }

    private enum MissionState
    {
        Idle,
        Transit,
        Search
    }

    private readonly List<DroneDemoZone> assignedZones = new List<DroneDemoZone>();

    private DroneCoverageAgent agent;
    private LockedAltitudeDroneController droneController;
    private MissionState missionState = MissionState.Idle;
    private float transitSpeed = 22f;
    private float arrivalDistance = 18f;
    private float slowDownDistance = 90f;
    private float fullTurnAngle = 45f;
    private int currentZoneIndex;
    private float searchMaxSpeed;
    private float searchAcceleration;
    private float searchTurnSpeed;
    private string droneLabel = string.Empty;
    private Vector3 currentTransitTarget;
    private ApproachEdge currentApproachEdge;
    private float currentXInset;
    private float currentZInset;
    private DroneDemoZone currentZone;
    private Bounds currentZoneBounds;

    public int AssignedZoneCount => assignedZones.Count;
    public int CurrentZoneNumber => assignedZones.Count == 0 ? 0 : assignedZones[currentZoneIndex].ZoneId + 1;
    public string DroneLabel => droneLabel;

    public void Configure(
        DroneCoverageAgent coverageAgent,
        LockedAltitudeDroneController controller,
        AreaCoverageTracker tracker,
        IEnumerable<DroneDemoZone> zones,
        float transitSpeedMetersPerSecond,
        string label)
    {
        agent = coverageAgent != null ? coverageAgent : GetComponent<DroneCoverageAgent>();
        droneController = controller != null ? controller : GetComponent<LockedAltitudeDroneController>();
        droneLabel = string.IsNullOrWhiteSpace(label) ? gameObject.name : label;
        transitSpeed = Mathf.Max(14.5f, transitSpeedMetersPerSecond);

        assignedZones.Clear();
        if (zones != null)
        {
            assignedZones.AddRange(zones);
        }

        currentZoneIndex = 0;
        missionState = MissionState.Idle;

        if (droneController != null)
        {
            searchMaxSpeed = droneController.maxSpeed;
            searchAcceleration = droneController.acceleration;
            searchTurnSpeed = droneController.turnSpeed;
            droneController.showDebugGui = false;
            droneController.ClearControlInputs();
        }

        if (agent != null)
        {
            agent.MissionSearchEnded -= HandleMissionSearchEnded;
            agent.MissionSearchEnded += HandleMissionSearchEnded;
            agent.ConfigureRuntimeMission(tracker);
        }
    }

    public void StartMission()
    {
        if (assignedZones.Count == 0 || agent == null || droneController == null)
        {
            missionState = MissionState.Idle;
            return;
        }

        StartTransitToCurrentZone();
    }

    public void StopMission()
    {
        missionState = MissionState.Idle;

        if (agent != null)
        {
            agent.MissionSearchEnded -= HandleMissionSearchEnded;
            agent.StopRuntimeSearch();
        }

        RestoreSearchFlightProfile();

        if (droneController != null)
        {
            droneController.ClearControlInputs();
        }
    }

    private void Update()
    {
        if (missionState != MissionState.Transit || droneController == null)
        {
            return;
        }

        if (!TryGetCurrentZone(out DroneDemoZone zone))
        {
            missionState = MissionState.Idle;
            droneController.ClearControlInputs();
            return;
        }

        if (HasReachedSearchEntryPoint())
        {
            BeginSearch();
            return;
        }

        Vector3 targetPosition = new Vector3(currentTransitTarget.x, transform.position.y, currentTransitTarget.z);
        Vector3 toTarget = targetPosition - transform.position;
        toTarget.y = 0f;

        float distance = toTarget.magnitude;
        if (distance <= arrivalDistance)
        {
            Vector3 flatToCenter = new Vector3(currentZone.CenterXZ.x - transform.position.x, 0f, currentZone.CenterXZ.y - transform.position.z);
            if (flatToCenter.sqrMagnitude > 0.0001f)
            {
                toTarget = flatToCenter;
                distance = toTarget.magnitude;
            }
        }

        Vector3 flatForward = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
        if (flatForward.sqrMagnitude < 0.0001f)
        {
            flatForward = Vector3.forward;
        }

        Vector3 desiredDirection = toTarget / Mathf.Max(0.001f, distance);
        float signedAngle = Vector3.SignedAngle(flatForward, desiredDirection, Vector3.up);
        float turnInput = Mathf.Clamp(signedAngle / fullTurnAngle, -1f, 1f);

        float headingError01 = Mathf.Clamp01(Mathf.Abs(signedAngle) / 120f);
        float forwardInput = Mathf.Lerp(1f, 0.2f, headingError01);
        if (distance < slowDownDistance)
        {
            forwardInput *= Mathf.Clamp01(distance / slowDownDistance);
        }

        forwardInput = Mathf.Max(0.25f, forwardInput);
        droneController.SetControlInputs(0f, forwardInput, turnInput);
    }

    private void StartTransitToCurrentZone()
    {
        if (!TryGetCurrentZone(out DroneDemoZone zone))
        {
            missionState = MissionState.Idle;
            return;
        }

        missionState = MissionState.Transit;
        ApplyTransitFlightProfile();
        currentZone = zone;
        currentZoneBounds = new Bounds(
            new Vector3(zone.CenterXZ.x, transform.position.y, zone.CenterXZ.y),
            new Vector3(zone.SizeXZ.x, 0f, zone.SizeXZ.y));
        agent.StopRuntimeSearch();
        agent.SetRuntimeSearchZone(zone.CenterXZ, zone.SizeXZ);
        currentTransitTarget = BuildSearchEntryTarget(zone);
    }

    private void BeginSearch()
    {
        missionState = MissionState.Search;
        RestoreSearchFlightProfile();
        transform.position = new Vector3(currentTransitTarget.x, transform.position.y, currentTransitTarget.z);

        if (agent.FaceZoneCenterOnEdgeSpawn)
        {
            Vector3 toCenter = new Vector3(
                currentZone.CenterXZ.x - transform.position.x,
                0f,
                currentZone.CenterXZ.y - transform.position.z);
            if (toCenter.sqrMagnitude > 0.0001f)
            {
                transform.rotation = Quaternion.LookRotation(toCenter.normalized, Vector3.up);
            }
        }

        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        droneController.ClearControlInputs();
        agent.BeginRuntimeSearch();
    }

    private void HandleMissionSearchEnded(DroneCoverageAgent sourceAgent, DroneCoverageAgent.MissionSearchOutcome outcome)
    {
        if (sourceAgent != agent || assignedZones.Count == 0)
        {
            return;
        }

        if (outcome == DroneCoverageAgent.MissionSearchOutcome.Completed)
        {
            currentZoneIndex = (currentZoneIndex + 1) % assignedZones.Count;
        }

        StartTransitToCurrentZone();
    }

    private bool TryGetCurrentZone(out DroneDemoZone zone)
    {
        if (assignedZones.Count == 0 || currentZoneIndex < 0 || currentZoneIndex >= assignedZones.Count)
        {
            zone = default;
            return false;
        }

        zone = assignedZones[currentZoneIndex];
        return true;
    }

    private void ApplyTransitFlightProfile()
    {
        if (droneController == null)
        {
            return;
        }

        droneController.maxSpeed = Mathf.Max(searchMaxSpeed, transitSpeed);
        droneController.acceleration = Mathf.Max(searchAcceleration, 18f);
        droneController.turnSpeed = Mathf.Max(searchTurnSpeed, 110f);
    }

    private void RestoreSearchFlightProfile()
    {
        if (droneController == null)
        {
            return;
        }

        droneController.maxSpeed = searchMaxSpeed;
        droneController.acceleration = searchAcceleration;
        droneController.turnSpeed = searchTurnSpeed;
    }

    private Vector3 BuildSearchEntryTarget(DroneDemoZone zone)
    {
        currentZoneBounds = new Bounds(
            new Vector3(zone.CenterXZ.x, transform.position.y, zone.CenterXZ.y),
            new Vector3(zone.SizeXZ.x, 0f, zone.SizeXZ.y));

        currentXInset = Mathf.Min(agent.EdgeSpawnInset, Mathf.Max(0f, (currentZoneBounds.size.x * 0.5f) - 0.01f));
        currentZInset = Mathf.Min(agent.EdgeSpawnInset, Mathf.Max(0f, (currentZoneBounds.size.z * 0.5f) - 0.01f));

        float leftDistance = Mathf.Abs(transform.position.x - currentZoneBounds.min.x);
        float rightDistance = Mathf.Abs(transform.position.x - currentZoneBounds.max.x);
        float bottomDistance = Mathf.Abs(transform.position.z - currentZoneBounds.min.z);
        float topDistance = Mathf.Abs(transform.position.z - currentZoneBounds.max.z);

        float minDistance = leftDistance;
        currentApproachEdge = ApproachEdge.Left;

        if (rightDistance < minDistance)
        {
            minDistance = rightDistance;
            currentApproachEdge = ApproachEdge.Right;
        }

        if (bottomDistance < minDistance)
        {
            minDistance = bottomDistance;
            currentApproachEdge = ApproachEdge.Bottom;
        }

        if (topDistance < minDistance)
        {
            currentApproachEdge = ApproachEdge.Top;
        }

        float targetX = Mathf.Clamp(
            transform.position.x,
            currentZoneBounds.min.x + currentXInset,
            currentZoneBounds.max.x - currentXInset);
        float targetZ = Mathf.Clamp(
            transform.position.z,
            currentZoneBounds.min.z + currentZInset,
            currentZoneBounds.max.z - currentZInset);

        switch (currentApproachEdge)
        {
            case ApproachEdge.Left:
                targetX = currentZoneBounds.min.x + currentXInset;
                break;
            case ApproachEdge.Right:
                targetX = currentZoneBounds.max.x - currentXInset;
                break;
            case ApproachEdge.Bottom:
                targetZ = currentZoneBounds.min.z + currentZInset;
                break;
            default:
                targetZ = currentZoneBounds.max.z - currentZInset;
                break;
        }

        return new Vector3(targetX, transform.position.y, targetZ);
    }

    private bool HasReachedSearchEntryPoint()
    {
        Vector3 position = transform.position;
        if (position.x < currentZoneBounds.min.x ||
            position.x > currentZoneBounds.max.x ||
            position.z < currentZoneBounds.min.z ||
            position.z > currentZoneBounds.max.z)
        {
            return false;
        }

        switch (currentApproachEdge)
        {
            case ApproachEdge.Left:
                return position.x >= currentZoneBounds.min.x + currentXInset;
            case ApproachEdge.Right:
                return position.x <= currentZoneBounds.max.x - currentXInset;
            case ApproachEdge.Bottom:
                return position.z >= currentZoneBounds.min.z + currentZInset;
            default:
                return position.z <= currentZoneBounds.max.z - currentZInset;
        }
    }
}
