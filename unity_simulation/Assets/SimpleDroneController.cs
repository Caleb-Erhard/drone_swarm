using UnityEngine;

public class LockedAltitudeDroneController : MonoBehaviour
{
    [Header("Altitude Settings")]
    public float targetAltitude = 40f;       // Height above ground to maintain
    public float altitudeForce = 50f;        // Force to maintain altitude
    public LayerMask groundLayer;            // What counts as ground (set in Inspector)

    [Header("Ground Tracking")]
    public float groundProbeStartHeight = 120f;
    public float groundProbeDistance = 300f;
    public float terrainLookAheadTime = 0.75f;
    public float maxTerrainLookAheadDistance = 45f;
    public float terrainProbeLateralOffset = 6f;
    public float altitudeRecoveryBoost = 1.75f;
    
    [Header("Movement Settings")]
    public float maxSpeed = 12f;             // Maximum horizontal speed (m/s)
    public float acceleration = 5f;          // How quickly drone reaches max speed
    public float turnSpeed = 100f;           // Rotation speed
    public float tiltAmount = 20f;           // Visual tilt when moving
    public float yawControlResponse = 12f;   // How quickly yaw rate tracks turn command
    
    [Header("Physics Settings")]
    public float drag = 2f;
    public float angularDrag = 3f;
    public bool lockPitchAndRoll = true;

    [Header("Visual Tilt")]
    public bool applyVisualTilt = false;
    public Transform visualTiltTransform;
    public float tiltSmoothness = 6f;

    [Header("Debug")]
    public bool showDebugGui = true;
    
    private Rigidbody rb;
    private float currentGroundHeight = 0f;
    private Vector3 currentTilt = Vector3.zero;
    private bool hasGroundSample;
    
    // For ML-Agents: these will be set by the agent
    [HideInInspector] public Vector2 movementInput = Vector2.zero;  // x = strafe, y = forward
    [HideInInspector] public float turnInput = 0f;                   // -1 to 1 for rotation

    [Header("Control Source")]
    public bool manualInputEnabled = true;

    public float MaxSpeed => maxSpeed;
    public float CurrentGroundHeight => currentGroundHeight;
    public float CurrentAltitudeAboveGround => transform.position.y - currentGroundHeight;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.linearDamping = drag;
        rb.angularDamping = angularDrag;
        rb.useGravity = true;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

        if (lockPitchAndRoll)
        {
            rb.constraints |= RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        }
    }

    void Update()
    {
        if (!manualInputEnabled)
        {
            return;
        }

        // Manual keyboard control for testing (can be disabled for ML-Agent control)
        movementInput.y = Input.GetAxisRaw("Vertical");    // W/S
        movementInput.x = Input.GetAxisRaw("Horizontal");  // A/D

        turnInput = 0f;
        if (Input.GetKey(KeyCode.Q)) turnInput = -1f;
        if (Input.GetKey(KeyCode.E)) turnInput = 1f;
    }

    void FixedUpdate()
    {
        // Get current ground height using raycast
        UpdateGroundHeight();
        
        // Lock altitude at target height above ground
        MaintainAltitude();
        
        // Apply horizontal movement
        ApplyMovement();
        
        // Apply rotation
        ApplyRotation();
        
        // Visual tilt
        ApplyTilt();
    }

    void UpdateGroundHeight()
    {
        Vector3 currentPosition = transform.position;
        float highestGround = float.NegativeInfinity;
        bool foundGround = TrySampleGroundHeight(currentPosition, out highestGround);

        Vector3 horizontalVelocity = rb != null
            ? new Vector3(rb.linearVelocity.x, 0f, rb.linearVelocity.z)
            : Vector3.zero;
        float horizontalSpeed = horizontalVelocity.magnitude;

        if (horizontalSpeed > 0.1f)
        {
            Vector3 travelDirection = horizontalVelocity / horizontalSpeed;
            float lookAheadDistance = Mathf.Min(maxTerrainLookAheadDistance, horizontalSpeed * terrainLookAheadTime);
            Vector3 lookAheadOffset = travelDirection * lookAheadDistance;
            Vector3 lateralOffset = Vector3.Cross(Vector3.up, travelDirection) * terrainProbeLateralOffset;

            foundGround |= TryPromoteGroundSample(currentPosition + lookAheadOffset, ref highestGround);
            foundGround |= TryPromoteGroundSample(currentPosition + lookAheadOffset + lateralOffset, ref highestGround);
            foundGround |= TryPromoteGroundSample(currentPosition + lookAheadOffset - lateralOffset, ref highestGround);
        }

        if (foundGround)
        {
            currentGroundHeight = highestGround;
            hasGroundSample = true;
            return;
        }

        if (!hasGroundSample)
        {
            currentGroundHeight = currentPosition.y - targetAltitude;
        }
    }

    void MaintainAltitude()
    {
        // Calculate desired Y position
        float desiredY = currentGroundHeight + targetAltitude;
        float currentY = transform.position.y;
        float altitudeError = desiredY - currentY;
        
        // Calculate hover force (counteract gravity)
        float hoverForce = -Physics.gravity.y * rb.mass;
        
        // Add correction force based on altitude error (PD controller)
        float correctionForce = altitudeError * altitudeForce;
        if (altitudeError > 0f)
        {
            float recoveryScale = Mathf.Lerp(1f, altitudeRecoveryBoost, Mathf.Clamp01(altitudeError / Mathf.Max(1f, targetAltitude)));
            correctionForce *= recoveryScale;
        }

        float dampingForce = -rb.linearVelocity.y * (altitudeForce * 0.5f); // Damping to prevent oscillation
        if (altitudeError > 0f && rb.linearVelocity.y < 0f)
        {
            dampingForce += -rb.linearVelocity.y * altitudeForce;
        }
        
        float totalLift = hoverForce + correctionForce + dampingForce;
        rb.AddForce(Vector3.up * totalLift, ForceMode.Force);
    }

    void ApplyMovement()
    {
        // Clamp input to ensure it's between -1 and 1
        movementInput = Vector2.ClampMagnitude(movementInput, 1f);
        
        // Calculate desired velocity in local space
        Vector3 forwardFlat = Vector3.ProjectOnPlane(transform.forward, Vector3.up).normalized;
        Vector3 rightFlat = Vector3.ProjectOnPlane(transform.right, Vector3.up).normalized;
        
        Vector3 desiredVelocity = forwardFlat * movementInput.y * maxSpeed;
        desiredVelocity += rightFlat * movementInput.x * maxSpeed;

        // Rigidbody linear damping is applied after FixedUpdate.
        // Pre-compensate desired horizontal velocity so post-physics speed still tracks maxSpeed.
        float dampingCompensation = 1f + (Mathf.Max(0f, rb.linearDamping) * Time.fixedDeltaTime);
        Vector3 compensatedDesiredVelocity = desiredVelocity * dampingCompensation;
        
        Vector3 currentVelocity = rb.linearVelocity;
        Vector3 currentHorizontalVelocity = new Vector3(currentVelocity.x, 0f, currentVelocity.z);
        Vector3 nextHorizontalVelocity = Vector3.MoveTowards(
            currentHorizontalVelocity,
            compensatedDesiredVelocity,
            Mathf.Max(0.1f, acceleration) * Time.fixedDeltaTime);

        rb.linearVelocity = new Vector3(nextHorizontalVelocity.x, currentVelocity.y, nextHorizontalVelocity.z);
    }

    void ApplyRotation()
    {
        float targetYawRate = turnInput * turnSpeed * Mathf.Deg2Rad;
        float nextYawRate = Mathf.MoveTowards(
            rb.angularVelocity.y,
            targetYawRate,
            Mathf.Max(0.1f, yawControlResponse) * Time.fixedDeltaTime);

        rb.angularVelocity = new Vector3(0f, nextYawRate, 0f);
    }

    void ApplyTilt()
    {
        if (!applyVisualTilt || visualTiltTransform == null)
        {
            return;
        }

        // Visual tilt based on movement
        Vector3 targetTilt = new Vector3(movementInput.y * tiltAmount, 0f, -movementInput.x * tiltAmount);
        currentTilt = Vector3.Lerp(currentTilt, targetTilt, Time.fixedDeltaTime * Mathf.Max(0.1f, tiltSmoothness));
        visualTiltTransform.localRotation = Quaternion.Euler(currentTilt.x, 0f, currentTilt.z);
    }

    public void SetControlInputs(float strafe, float forward, float turn)
    {
        movementInput = Vector2.ClampMagnitude(new Vector2(strafe, forward), 1f);
        turnInput = Mathf.Clamp(turn, -1f, 1f);
    }

    public void ClearControlInputs()
    {
        movementInput = Vector2.zero;
        turnInput = 0f;
    }

    bool TryPromoteGroundSample(Vector3 probePosition, ref float highestGround)
    {
        if (!TrySampleGroundHeight(probePosition, out float sampledGroundHeight))
        {
            return false;
        }

        if (sampledGroundHeight > highestGround)
        {
            highestGround = sampledGroundHeight;
        }

        return true;
    }

    bool TrySampleGroundHeight(Vector3 probePosition, out float groundHeight)
    {
        Vector3 rayOrigin = probePosition + Vector3.up * Mathf.Max(5f, groundProbeStartHeight);
        float rayDistance = Mathf.Max(10f, groundProbeStartHeight + groundProbeDistance);

        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, rayDistance, groundLayer, QueryTriggerInteraction.Ignore))
        {
            groundHeight = hit.point.y;
            return true;
        }

        Terrain activeTerrain = Terrain.activeTerrain;
        if (activeTerrain != null && activeTerrain.terrainData != null)
        {
            Vector3 terrainOrigin = activeTerrain.GetPosition();
            Vector3 terrainSize = activeTerrain.terrainData.size;
            bool withinTerrainBounds =
                probePosition.x >= terrainOrigin.x &&
                probePosition.x <= terrainOrigin.x + terrainSize.x &&
                probePosition.z >= terrainOrigin.z &&
                probePosition.z <= terrainOrigin.z + terrainSize.z;

            if (withinTerrainBounds)
            {
                groundHeight = activeTerrain.SampleHeight(probePosition) + terrainOrigin.y;
                return true;
            }
        }

        groundHeight = 0f;
        return false;
    }

    // Debug visualization
    void OnDrawGizmos()
    {
        // Draw target altitude line
        Gizmos.color = Color.green;
        Gizmos.DrawWireSphere(new Vector3(transform.position.x, currentGroundHeight + targetAltitude, transform.position.z), 2f);
        
        // Draw line to ground
        Gizmos.color = Color.red;
        Gizmos.DrawLine(transform.position, new Vector3(transform.position.x, currentGroundHeight, transform.position.z));
    }

    void OnGUI()
    {
        if (!showDebugGui || rb == null)
        {
            return;
        }

        float currentAltitude = transform.position.y - currentGroundHeight;
        GUI.Label(new Rect(10, 10, 400, 20), $"Altitude: {currentAltitude:F1}m / Target: {targetAltitude}m");
        GUI.Label(new Rect(10, 30, 400, 20), $"Speed: {new Vector3(rb.linearVelocity.x, 0, rb.linearVelocity.z).magnitude:F1} m/s / Max: {maxSpeed} m/s");
        GUI.Label(new Rect(10, 50, 400, 20), manualInputEnabled ? "Controls: WASD=Move, Q/E=Turn" : "Controls: ML-Agent");
        GUI.Label(new Rect(10, 70, 400, 20), $"Ground Height: {currentGroundHeight:F1}m");
    }
}
