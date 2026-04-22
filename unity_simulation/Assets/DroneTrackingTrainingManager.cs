using Unity.MLAgents;
using UnityEngine;

[DisallowMultipleComponent]
public class DroneTrackingTrainingManager : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private DroneTrackingAgent trackingAgent;
    [SerializeField] private DroneTrackingSensorFootprint sensorFootprint;
    [SerializeField] private TruckTarget truckTarget;
    [SerializeField] private TruckTrainingMover truckMover;
    [SerializeField] private BoxCollider episodeBounds;
    [SerializeField] private Terrain terrain;

    [Header("Spawn Settings")]
    [SerializeField, Min(5f)] private float spawnMargin = 90f;
    [SerializeField] private bool useRadialFootprintSpawn = true;
    [SerializeField] private Vector2 truckForwardPlacementRange01 = new Vector2(0.30f, 0.70f);
    [SerializeField, Min(0f)] private float truckMaxCenterlineOffset = 4f;
    [SerializeField] private Vector2 truckSpawnRadiusRange01 = new Vector2(1.08f, 1.28f);
    [SerializeField] private Vector2 truckEntryTargetRadiusRange01 = new Vector2(0.12f, 0.35f);
    [SerializeField] private Vector2 truckHeadingOffsetRange = new Vector2(-20f, 20f);
    [SerializeField] private Vector2 truckSpeedRange = new Vector2(10f, 12f);

    private LockedAltitudeDroneController droneController;
    private Rigidbody droneBody;

    public TruckTarget TruckTarget => truckTarget;

    private void Awake()
    {
        Academy.Instance.AutomaticSteppingEnabled = true;
        ResolveReferences();
    }

    public void ResetEpisode()
    {
        ResolveReferences();

        if (trackingAgent == null ||
            truckTarget == null ||
            truckMover == null ||
            episodeBounds == null ||
            droneController == null)
        {
            Debug.LogWarning("DroneTrackingTrainingManager is missing one or more references.", this);
            return;
        }

        Bounds bounds = episodeBounds.bounds;
        float minX = bounds.min.x + spawnMargin;
        float maxX = bounds.max.x - spawnMargin;
        float minZ = bounds.min.z + spawnMargin;
        float maxZ = bounds.max.z - spawnMargin;

        if (minX >= maxX || minZ >= maxZ)
        {
            Debug.LogWarning("EpisodeBounds is too small for the configured training margins.", episodeBounds);
            return;
        }

        Vector3 dronePosition = new Vector3(
            Random.Range(minX, maxX),
            0f,
            Random.Range(minZ, maxZ));
        dronePosition.y = SampleSurfaceHeight(dronePosition) + droneController.targetAltitude;

        float droneYaw = Random.Range(0f, 360f);
        trackingAgent.transform.SetPositionAndRotation(dronePosition, Quaternion.Euler(0f, droneYaw, 0f));

        if (droneBody != null)
        {
            droneBody.linearVelocity = Vector3.zero;
            droneBody.angularVelocity = Vector3.zero;
        }

        droneController.ClearControlInputs();
        droneController.manualInputEnabled = false;
        Physics.SyncTransforms();

        float minHeadingOffset = Mathf.Min(truckHeadingOffsetRange.x, truckHeadingOffsetRange.y);
        float maxHeadingOffset = Mathf.Max(truckHeadingOffsetRange.x, truckHeadingOffsetRange.y);
        float truckHeading = Mathf.Repeat(droneYaw + Random.Range(minHeadingOffset, maxHeadingOffset), 360f);
        float truckSpeed = Random.Range(
            Mathf.Min(truckSpeedRange.x, truckSpeedRange.y),
            Mathf.Max(truckSpeedRange.x, truckSpeedRange.y));

        Vector3 truckPosition;
        if (!TryGetTruckSpawnFromFootprint(bounds, droneYaw, out truckPosition, out float entryHeading))
        {
            Vector3 fallbackForward = Quaternion.Euler(0f, droneYaw, 0f) * Vector3.forward;
            truckPosition = dronePosition + (fallbackForward * 45f);
            truckPosition = ClampInsideBounds(truckPosition, bounds, spawnMargin);
            truckPosition.y = SampleSurfaceHeight(truckPosition);
        }
        else
        {
            truckHeading = entryHeading;
        }

        truckTarget.ResetTrackingState();
        truckMover.ResetMotion(episodeBounds, truckPosition, truckHeading, truckSpeed);
        Physics.SyncTransforms();
    }

    public float DistanceOutsideEpisodeBounds(Vector3 worldPosition)
    {
        if (episodeBounds == null)
        {
            return 0f;
        }

        Bounds bounds = episodeBounds.bounds;
        float dx = Mathf.Max(bounds.min.x - worldPosition.x, 0f, worldPosition.x - bounds.max.x);
        float dz = Mathf.Max(bounds.min.z - worldPosition.z, 0f, worldPosition.z - bounds.max.z);
        return Mathf.Sqrt((dx * dx) + (dz * dz));
    }

    public Vector4 GetNormalizedDistancesToBounds(Vector3 worldPosition)
    {
        if (episodeBounds == null)
        {
            return Vector4.one * 0.5f;
        }

        Bounds bounds = episodeBounds.bounds;
        float width = Mathf.Max(0.001f, bounds.size.x);
        float depth = Mathf.Max(0.001f, bounds.size.z);

        float toMinX = Mathf.Clamp01((worldPosition.x - bounds.min.x) / width);
        float toMaxX = Mathf.Clamp01((bounds.max.x - worldPosition.x) / width);
        float toMinZ = Mathf.Clamp01((worldPosition.z - bounds.min.z) / depth);
        float toMaxZ = Mathf.Clamp01((bounds.max.z - worldPosition.z) / depth);

        return new Vector4(toMinX, toMaxX, toMinZ, toMaxZ);
    }

    public float SampleSurfaceHeight(Vector3 worldPosition)
    {
        if (terrain != null && terrain.terrainData != null)
        {
            return terrain.SampleHeight(worldPosition) + terrain.GetPosition().y;
        }

        Vector3 rayOrigin = worldPosition + (Vector3.up * 200f);
        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hitInfo, 500f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
        {
            return hitInfo.point.y;
        }

        return worldPosition.y;
    }

    private void ResolveReferences()
    {
        if (trackingAgent == null)
        {
            trackingAgent = FindFirstObjectByType<DroneTrackingAgent>();
        }

        if (truckTarget == null)
        {
            truckTarget = FindFirstObjectByType<TruckTarget>();
        }

        if (truckMover == null && truckTarget != null)
        {
            truckMover = truckTarget.GetComponent<TruckTrainingMover>();
        }

        if (episodeBounds == null)
        {
            BoxCollider[] colliders = FindObjectsByType<BoxCollider>(FindObjectsSortMode.None);
            for (int i = 0; i < colliders.Length; i++)
            {
                if (colliders[i] != null && colliders[i].isTrigger)
                {
                    episodeBounds = colliders[i];
                    break;
                }
            }
        }

        if (terrain == null)
        {
            terrain = Terrain.activeTerrain != null
                ? Terrain.activeTerrain
                : FindFirstObjectByType<Terrain>();
        }

        if (trackingAgent != null)
        {
            if (sensorFootprint == null)
            {
                sensorFootprint = trackingAgent.GetComponent<DroneTrackingSensorFootprint>();
            }

            if (droneController == null)
            {
                droneController = trackingAgent.GetComponent<LockedAltitudeDroneController>();
            }

            if (droneBody == null)
            {
                droneBody = trackingAgent.GetComponent<Rigidbody>();
            }
        }
    }

    private static Vector3 ClampInsideBounds(Vector3 position, Bounds bounds, float margin)
    {
        position.x = Mathf.Clamp(position.x, bounds.min.x + margin, bounds.max.x - margin);
        position.z = Mathf.Clamp(position.z, bounds.min.z + margin, bounds.max.z - margin);
        return position;
    }

    private bool TryGetTruckSpawnFromFootprint(Bounds bounds, float droneYaw, out Vector3 truckPosition, out float truckHeading)
    {
        truckPosition = default;
        truckHeading = 0f;

        if (sensorFootprint == null ||
            !sensorFootprint.TryGetGroundFootprint(out Vector3[] footprintCorners, out int cornerCount) ||
            footprintCorners == null ||
            cornerCount < 4)
        {
            return false;
        }

        Vector3 footprintCenter = Vector3.zero;
        for (int i = 0; i < cornerCount; i++)
        {
            footprintCenter += footprintCorners[i];
        }

        footprintCenter /= cornerCount;

        Vector3 forwardAxis = Quaternion.Euler(0f, droneYaw, 0f) * Vector3.forward;
        forwardAxis.y = 0f;
        forwardAxis = forwardAxis.sqrMagnitude > 0.0001f ? forwardAxis.normalized : Vector3.forward;

        Vector3 rightAxis = Quaternion.Euler(0f, droneYaw, 0f) * Vector3.right;
        rightAxis.y = 0f;
        rightAxis = rightAxis.sqrMagnitude > 0.0001f ? rightAxis.normalized : Vector3.right;

        float minForward = float.PositiveInfinity;
        float maxForward = float.NegativeInfinity;
        float minRight = float.PositiveInfinity;
        float maxRight = float.NegativeInfinity;

        for (int i = 0; i < cornerCount; i++)
        {
            Vector3 offset = footprintCorners[i] - footprintCenter;
            offset.y = 0f;

            float forwardProjection = Vector3.Dot(offset, forwardAxis);
            float rightProjection = Vector3.Dot(offset, rightAxis);

            minForward = Mathf.Min(minForward, forwardProjection);
            maxForward = Mathf.Max(maxForward, forwardProjection);
            minRight = Mathf.Min(minRight, rightProjection);
            maxRight = Mathf.Max(maxRight, rightProjection);
        }

        if (!float.IsFinite(minForward) || !float.IsFinite(maxForward))
        {
            return false;
        }

        float forwardHalfExtent = Mathf.Max(Mathf.Abs(minForward), Mathf.Abs(maxForward));
        float rightHalfExtent = Mathf.Max(Mathf.Abs(minRight), Mathf.Abs(maxRight));

        if (useRadialFootprintSpawn && forwardHalfExtent > 0.001f && rightHalfExtent > 0.001f)
        {
            float minSpawnRadius01 = Mathf.Min(truckSpawnRadiusRange01.x, truckSpawnRadiusRange01.y);
            float maxSpawnRadius01 = Mathf.Max(truckSpawnRadiusRange01.x, truckSpawnRadiusRange01.y);
            float minTargetRadius01 = Mathf.Clamp01(Mathf.Min(truckEntryTargetRadiusRange01.x, truckEntryTargetRadiusRange01.y));
            float maxTargetRadius01 = Mathf.Clamp01(Mathf.Max(truckEntryTargetRadiusRange01.x, truckEntryTargetRadiusRange01.y));
            float minHeadingOffset = Mathf.Min(truckHeadingOffsetRange.x, truckHeadingOffsetRange.y);
            float maxHeadingOffset = Mathf.Max(truckHeadingOffsetRange.x, truckHeadingOffsetRange.y);
            float angle = Random.Range(0f, Mathf.PI * 2f);
            float spawnRadius01 = Random.Range(minSpawnRadius01, maxSpawnRadius01);
            float targetRadius01 = Random.Range(minTargetRadius01, maxTargetRadius01);

            float entryForwardOffset = Mathf.Cos(angle) * forwardHalfExtent;
            float entryRightOffset = Mathf.Sin(angle) * rightHalfExtent;

            truckPosition = footprintCenter +
                            (forwardAxis * entryForwardOffset * spawnRadius01) +
                            (rightAxis * entryRightOffset * spawnRadius01);
            truckPosition = ClampInsideBounds(truckPosition, bounds, spawnMargin);

            Vector3 entryTarget = footprintCenter +
                                  (forwardAxis * entryForwardOffset * targetRadius01) +
                                  (rightAxis * entryRightOffset * targetRadius01);
            entryTarget = ClampInsideBounds(entryTarget, bounds, spawnMargin);

            Vector3 toEntryTarget = entryTarget - truckPosition;
            toEntryTarget.y = 0f;
            if (toEntryTarget.sqrMagnitude > 0.0001f)
            {
                truckHeading = Quaternion.LookRotation(toEntryTarget.normalized, Vector3.up).eulerAngles.y;
                truckHeading = Mathf.Repeat(truckHeading + Random.Range(minHeadingOffset, maxHeadingOffset), 360f);
            }

            truckPosition.y = SampleSurfaceHeight(truckPosition);
            return true;
        }

        float minForward01 = Mathf.Clamp01(Mathf.Min(truckForwardPlacementRange01.x, truckForwardPlacementRange01.y));
        float maxForward01 = Mathf.Clamp01(Mathf.Max(truckForwardPlacementRange01.x, truckForwardPlacementRange01.y));
        float forwardOffset = Mathf.Lerp(minForward, maxForward, Random.Range(minForward01, maxForward01));

        float centerRight = Mathf.Lerp(minRight, maxRight, 0.5f);
        float rightInset = Mathf.Min(1f, Mathf.Max(0f, (maxRight - minRight) * 0.1f));
        float rightOffset = centerRight + Random.Range(-truckMaxCenterlineOffset, truckMaxCenterlineOffset);
        rightOffset = Mathf.Clamp(rightOffset, minRight + rightInset, maxRight - rightInset);

        truckPosition = footprintCenter + (forwardAxis * forwardOffset) + (rightAxis * rightOffset);
        truckPosition = ClampInsideBounds(truckPosition, bounds, spawnMargin);
        truckPosition.y = SampleSurfaceHeight(truckPosition);
        return true;
    }
}
