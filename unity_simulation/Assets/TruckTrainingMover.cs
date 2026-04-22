using UnityEngine;

[DisallowMultipleComponent]
public class TruckTrainingMover : MonoBehaviour
{
    [SerializeField] private BoxCollider movementBounds;
    [SerializeField, Min(1f)] private float turnRateDegrees = 20f;
    [SerializeField, Min(0f)] private float steeringJitterDegrees = 8f;
    [SerializeField, Min(5f)] private float boundaryMargin = 40f;
    [SerializeField] private Vector2 headingChangeIntervalRange = new Vector2(1.25f, 2.5f);
    [SerializeField] private Vector2 straightDriveDurationRange = new Vector2(3f, 5f);
    [SerializeField, Min(0f)] private float defaultSpeed = 10f;

    private Terrain terrain;
    private Rigidbody rb;
    private float currentSpeed;
    private float desiredHeading;
    private float nextHeadingChangeTime;
    private float headingChangesAllowedAtTime;
    private float surfaceHeightOffset;
    private bool isConfigured;

    private void Awake()
    {
        terrain = Terrain.activeTerrain != null
            ? Terrain.activeTerrain
            : FindFirstObjectByType<Terrain>();

        rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        surfaceHeightOffset = transform.position.y - SampleSurfaceHeight(transform.position);
    }

    private void FixedUpdate()
    {
        if (!isConfigured)
        {
            return;
        }

        if (Time.time >= headingChangesAllowedAtTime && Time.time >= nextHeadingChangeTime)
        {
            desiredHeading += Random.Range(-steeringJitterDegrees, steeringJitterDegrees);
            ScheduleNextHeadingChange();
        }

        if (movementBounds != null)
        {
            Vector3 toCenter = movementBounds.bounds.center - transform.position;
            toCenter.y = 0f;

            Vector3 clampedPosition = ClampInsideBounds(transform.position, movementBounds.bounds, boundaryMargin);
            bool nearEdge = Vector3.Distance(
                new Vector3(transform.position.x, 0f, transform.position.z),
                new Vector3(clampedPosition.x, 0f, clampedPosition.z)) > 0.05f;

            if (nearEdge && toCenter.sqrMagnitude > 0.001f)
            {
                desiredHeading = Quaternion.LookRotation(toCenter.normalized, Vector3.up).eulerAngles.y;
            }
        }

        float currentYaw = transform.eulerAngles.y;
        float nextYaw = Mathf.MoveTowardsAngle(currentYaw, desiredHeading, turnRateDegrees * Time.fixedDeltaTime);
        transform.rotation = Quaternion.Euler(0f, nextYaw, 0f);

        Vector3 nextPosition = transform.position + (transform.forward * currentSpeed * Time.fixedDeltaTime);

        if (movementBounds != null)
        {
            nextPosition = ClampInsideBounds(nextPosition, movementBounds.bounds, boundaryMargin * 0.5f);
        }

        nextPosition.y = SampleSurfaceHeight(nextPosition) + surfaceHeightOffset;
        transform.position = nextPosition;
    }

    public void ResetMotion(BoxCollider boundsCollider, Vector3 startPosition, float headingDegrees, float speed)
    {
        movementBounds = boundsCollider;
        currentSpeed = Mathf.Max(0f, speed > 0f ? speed : defaultSpeed);
        desiredHeading = headingDegrees;
        headingChangesAllowedAtTime = Time.time + SampleDuration(straightDriveDurationRange);

        startPosition.y = SampleSurfaceHeight(startPosition) + surfaceHeightOffset;
        transform.SetPositionAndRotation(startPosition, Quaternion.Euler(0f, headingDegrees, 0f));

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        isConfigured = true;
        ScheduleNextHeadingChange();
    }

    public void StopMotion()
    {
        isConfigured = false;

        if (rb != null)
        {
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    private void ScheduleNextHeadingChange()
    {
        float minInterval = Mathf.Min(headingChangeIntervalRange.x, headingChangeIntervalRange.y);
        float maxInterval = Mathf.Max(headingChangeIntervalRange.x, headingChangeIntervalRange.y);
        nextHeadingChangeTime = Time.time + Random.Range(minInterval, maxInterval);
    }

    private static float SampleDuration(Vector2 durationRange)
    {
        float minDuration = Mathf.Min(durationRange.x, durationRange.y);
        float maxDuration = Mathf.Max(durationRange.x, durationRange.y);
        return Random.Range(minDuration, maxDuration);
    }

    private float SampleSurfaceHeight(Vector3 worldPosition)
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

    private static Vector3 ClampInsideBounds(Vector3 position, Bounds bounds, float margin)
    {
        position.x = Mathf.Clamp(position.x, bounds.min.x + margin, bounds.max.x - margin);
        position.z = Mathf.Clamp(position.z, bounds.min.z + margin, bounds.max.z - margin);
        return position;
    }
}
