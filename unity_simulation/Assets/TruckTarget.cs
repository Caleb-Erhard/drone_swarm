using UnityEngine;

[DisallowMultipleComponent]
public class TruckTarget : MonoBehaviour
{
    [SerializeField] private int truckId = 1;
    [SerializeField] private Transform trackingPointOverride;
    [SerializeField, Min(0f)] private float trackingPointHeightOffset = 1.5f;
    [SerializeField] private bool useColliderBoundsForTrackingPoint = true;

    private int currentTrackerDroneId = -1;
    private Vector3 previousPosition;
    private Vector3 currentVelocity;
    private bool hasPreviousPosition;
    private Collider[] cachedColliders;

    public int TruckId => truckId;
    public int CurrentTrackerDroneId => currentTrackerDroneId;
    public bool IsClaimed => currentTrackerDroneId >= 0;
    public Vector3 CurrentVelocity => currentVelocity;

    private void OnEnable()
    {
        CacheColliders();
        ResetTrackingState();
    }

    private void FixedUpdate()
    {
        if (!hasPreviousPosition)
        {
            previousPosition = transform.position;
            hasPreviousPosition = true;
            currentVelocity = Vector3.zero;
            return;
        }

        currentVelocity = (transform.position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.0001f);
        previousPosition = transform.position;
    }

    public void ResetTrackingState()
    {
        currentTrackerDroneId = -1;
        currentVelocity = Vector3.zero;
        previousPosition = transform.position;
        hasPreviousPosition = true;
    }

    public bool TryClaim(int droneId)
    {
        if (IsClaimed && currentTrackerDroneId != droneId)
        {
            return false;
        }

        currentTrackerDroneId = droneId;
        return true;
    }

    public void ReleaseClaim(int droneId)
    {
        if (currentTrackerDroneId == droneId)
        {
            currentTrackerDroneId = -1;
        }
    }

    public Vector3 GetTrackingPoint()
    {
        if (trackingPointOverride != null)
        {
            return trackingPointOverride.position;
        }

        if (useColliderBoundsForTrackingPoint && TryGetColliderBounds(out Bounds bounds))
        {
            return new Vector3(
                bounds.center.x,
                bounds.center.y + trackingPointHeightOffset,
                bounds.center.z);
        }

        return transform.position + (Vector3.up * trackingPointHeightOffset);
    }

    private bool TryGetColliderBounds(out Bounds bounds)
    {
        CacheColliders();

        bool found = false;
        bounds = default;
        for (int i = 0; i < cachedColliders.Length; i++)
        {
            Collider colliderComponent = cachedColliders[i];
            if (colliderComponent == null || !colliderComponent.enabled)
            {
                continue;
            }

            if (!found)
            {
                bounds = colliderComponent.bounds;
                found = true;
            }
            else
            {
                bounds.Encapsulate(colliderComponent.bounds);
            }
        }

        return found;
    }

    private void CacheColliders()
    {
        if (cachedColliders == null || cachedColliders.Length == 0)
        {
            cachedColliders = GetComponentsInChildren<Collider>(true);
        }
    }
}
