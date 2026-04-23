using UnityEngine;

[DisallowMultipleComponent]
public class TruckTarget : MonoBehaviour
{
    [SerializeField] private int truckId = 1;
    [SerializeField] private Transform trackingPointOverride;
    [SerializeField, Min(0f)] private float trackingPointHeightOffset = 1.5f;
    [SerializeField] private bool useColliderBoundsForTrackingPoint = true;
    [SerializeField] private string preferredTrackingColliderName = "TruckBody_Col";
    [SerializeField] private string fallbackTrackingColliderName = "Collider";
    [SerializeField, Min(0f)] private float velocitySmoothing = 10f;
    [SerializeField, Min(0.0001f)] private float minimumSampleInterval = 0.008f;
    [SerializeField, Min(0f)] private float minimumMovementDelta = 0.0025f;

    private int currentTrackerDroneId = -1;
    private Vector3 previousPosition;
    private float previousSampleTime;
    private Vector3 currentVelocity;
    private bool hasPreviousPosition;
    private Collider[] cachedColliders;
    private Rigidbody cachedRigidbody;
    private Collider preferredTrackingCollider;

    public int TruckId => truckId;
    public int CurrentTrackerDroneId => currentTrackerDroneId;
    public bool IsClaimed => currentTrackerDroneId >= 0;
    public Vector3 CurrentVelocity => currentVelocity;

    private void OnEnable()
    {
        CacheColliders();
        if (cachedRigidbody == null)
        {
            cachedRigidbody = GetComponent<Rigidbody>();
        }

        ResetTrackingState();
    }

    private void FixedUpdate()
    {
        if (cachedRigidbody != null && !cachedRigidbody.isKinematic)
        {
            currentVelocity = cachedRigidbody.linearVelocity;
            previousPosition = transform.position;
            previousSampleTime = Time.time;
            hasPreviousPosition = true;
            return;
        }

        SampleTransformVelocity(Time.time);
    }

    private void LateUpdate()
    {
        if (cachedRigidbody != null && !cachedRigidbody.isKinematic)
        {
            return;
        }

        SampleTransformVelocity(Time.time);
    }

    public void ResetTrackingState()
    {
        currentTrackerDroneId = -1;
        currentVelocity = Vector3.zero;
        previousPosition = transform.position;
        previousSampleTime = Time.time;
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

        Collider resolvedTrackingCollider = ResolvePreferredTrackingCollider();
        if (resolvedTrackingCollider != null && resolvedTrackingCollider.enabled)
        {
            bounds = resolvedTrackingCollider.bounds;
            return true;
        }

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
            preferredTrackingCollider = null;
        }
    }

    private Collider ResolvePreferredTrackingCollider()
    {
        CacheColliders();

        if (preferredTrackingCollider != null)
        {
            return preferredTrackingCollider;
        }

        preferredTrackingCollider = FindNamedCollider(preferredTrackingColliderName);
        if (preferredTrackingCollider != null)
        {
            return preferredTrackingCollider;
        }

        preferredTrackingCollider = FindNamedCollider(fallbackTrackingColliderName);
        return preferredTrackingCollider;
    }

    private Collider FindNamedCollider(string colliderName)
    {
        if (string.IsNullOrWhiteSpace(colliderName) || cachedColliders == null)
        {
            return null;
        }

        for (int i = 0; i < cachedColliders.Length; i++)
        {
            Collider colliderComponent = cachedColliders[i];
            if (colliderComponent == null)
            {
                continue;
            }

            if (string.Equals(colliderComponent.gameObject.name, colliderName, System.StringComparison.OrdinalIgnoreCase))
            {
                return colliderComponent;
            }
        }

        return null;
    }

    private void SampleTransformVelocity(float sampleTime)
    {
        if (!hasPreviousPosition)
        {
            previousPosition = transform.position;
            previousSampleTime = sampleTime;
            hasPreviousPosition = true;
            currentVelocity = Vector3.zero;
            return;
        }

        float deltaTime = sampleTime - previousSampleTime;
        if (deltaTime < minimumSampleInterval)
        {
            return;
        }

        Vector3 currentPosition = transform.position;
        Vector3 deltaPosition = currentPosition - previousPosition;
        float minimumMovementDeltaSq = minimumMovementDelta * minimumMovementDelta;
        if (deltaPosition.sqrMagnitude <= minimumMovementDeltaSq)
        {
            previousPosition = currentPosition;
            previousSampleTime = sampleTime;
            return;
        }

        Vector3 rawVelocity = deltaPosition / Mathf.Max(deltaTime, 0.0001f);
        if (velocitySmoothing <= 0f || currentVelocity.sqrMagnitude <= 0.0001f)
        {
            currentVelocity = rawVelocity;
        }
        else
        {
            float blend = 1f - Mathf.Exp(-velocitySmoothing * deltaTime);
            currentVelocity = Vector3.Lerp(currentVelocity, rawVelocity, blend);
        }

        previousPosition = currentPosition;
        previousSampleTime = sampleTime;
    }
}
