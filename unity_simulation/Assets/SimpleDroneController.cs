using UnityEngine;

public class LockedAltitudeDroneController : MonoBehaviour
{
    [Header("Altitude Settings")]
    public float targetAltitude = 40f;       // Height above ground to maintain
    public float altitudeForce = 50f;        // Force to maintain altitude
    public LayerMask groundLayer;            // What counts as ground (set in Inspector)
    
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
    
    // For ML-Agents: these will be set by the agent
    [HideInInspector] public Vector2 movementInput = Vector2.zero;  // x = strafe, y = forward
    [HideInInspector] public float turnInput = 0f;                   // -1 to 1 for rotation

    [Header("Control Source")]
    public bool manualInputEnabled = true;

    public float MaxSpeed => maxSpeed;
    
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.linearDamping = drag;
        rb.angularDamping = angularDrag;
        rb.useGravity = true;
        rb.interpolation = RigidbodyInterpolation.Interpolate;

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
        RaycastHit hit;
        // Cast ray downward from drone
        if (Physics.Raycast(transform.position, Vector3.down, out hit, 1000f, groundLayer))
        {
            currentGroundHeight = hit.point.y;
        }
        else
        {
            // If no ground found, use current Y position minus target altitude as fallback
            currentGroundHeight = transform.position.y - targetAltitude;
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
        float dampingForce = -rb.linearVelocity.y * (altitudeForce * 0.5f); // Damping to prevent oscillation
        
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
