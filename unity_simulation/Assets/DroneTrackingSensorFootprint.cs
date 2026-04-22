using UnityEngine;

[DisallowMultipleComponent]
public class DroneTrackingSensorFootprint : MonoBehaviour
{
    private static readonly Vector2[] ViewportCorners =
    {
        new Vector2(0f, 0f),
        new Vector2(0f, 1f),
        new Vector2(1f, 1f),
        new Vector2(1f, 0f)
    };

    [Header("References")]
    [SerializeField] private Transform sensorTransform;
    [SerializeField] private Camera coverageCamera;

    [Header("Footprint")]
    [SerializeField, Min(0.1f)] private float sensorRelativePlaneDistance = 40f;

    [Header("Debug")]
    [SerializeField] private bool drawGroundFootprint = true;
    [SerializeField, Min(4)] private int sensorFootprintSegments = 48;
    [SerializeField] private float sensorFootprintGroundOffset = 0.08f;
    [SerializeField] private Color sensorFootprintColor = new Color(1f, 0.85f, 0.2f, 0.9f);

    private readonly Vector3[] cameraGroundFootprintCorners = new Vector3[4];
    private bool hasCameraGroundFootprint;
    private float equivalentRadius = 0.1f;

    public Transform SensorTransform => sensorTransform != null ? sensorTransform : transform;
    public Camera CoverageCamera => ResolveCoverageCamera();
    public Vector3 SensorWorldPosition => SensorTransform.position;
    public float EquivalentRadius => Mathf.Max(0.1f, equivalentRadius);

    private void Awake()
    {
        ResolveReferences();
    }

    private void OnValidate()
    {
        sensorRelativePlaneDistance = Mathf.Max(0.1f, sensorRelativePlaneDistance);
        sensorFootprintSegments = Mathf.Max(4, sensorFootprintSegments);
        ResolveReferences();
    }

    public bool IsPointInView(Vector3 worldPoint)
    {
        Camera activeCamera = ResolveCoverageCamera();
        if (activeCamera == null)
        {
            return false;
        }

        Vector3 viewportPoint = activeCamera.WorldToViewportPoint(worldPoint);
        return viewportPoint.z > 0f &&
               viewportPoint.x >= 0f && viewportPoint.x <= 1f &&
               viewportPoint.y >= 0f && viewportPoint.y <= 1f;
    }

    public bool TryGetGroundFootprint(out Vector3[] footprintCorners, out int cornerCount)
    {
        ResolveReferences();

        if (!TryUpdateCameraGroundFootprint())
        {
            footprintCorners = null;
            cornerCount = 0;
            return false;
        }

        footprintCorners = cameraGroundFootprintCorners;
        cornerCount = cameraGroundFootprintCorners.Length;
        return true;
    }

    private void ResolveReferences()
    {
        if (sensorTransform == null)
        {
            sensorTransform = transform;
        }

        if (coverageCamera == null)
        {
            coverageCamera = GetComponentInChildren<Camera>(true);
        }
    }

    private Camera ResolveCoverageCamera()
    {
        ResolveReferences();
        return coverageCamera;
    }

    private bool TryUpdateCameraGroundFootprint()
    {
        Camera activeCamera = ResolveCoverageCamera();
        if (activeCamera == null)
        {
            hasCameraGroundFootprint = false;
            return false;
        }

        for (int i = 0; i < ViewportCorners.Length; i++)
        {
            if (!TryProjectViewportToGround(activeCamera, ViewportCorners[i], out Vector3 groundPoint))
            {
                hasCameraGroundFootprint = false;
                equivalentRadius = 0.1f;
                return false;
            }

            cameraGroundFootprintCorners[i] = groundPoint;
        }

        hasCameraGroundFootprint = true;
        equivalentRadius = ComputeEquivalentRadius(cameraGroundFootprintCorners, cameraGroundFootprintCorners.Length);
        return true;
    }

    private bool TryProjectViewportToGround(Camera camera, Vector2 viewportPoint, out Vector3 groundPoint)
    {
        Ray ray = camera.ViewportPointToRay(new Vector3(viewportPoint.x, viewportPoint.y, 0f));
        float planeY = SensorTransform.position.y - sensorRelativePlaneDistance;
        return TryProjectRayToPlaneY(ray, planeY, out groundPoint);
    }

    private static bool TryProjectRayToPlaneY(Ray ray, float planeY, out Vector3 projectedPoint)
    {
        float directionY = ray.direction.y;
        if (Mathf.Abs(directionY) < 0.00001f)
        {
            projectedPoint = default;
            return false;
        }

        float t = (planeY - ray.origin.y) / directionY;
        if (t <= 0f)
        {
            projectedPoint = default;
            return false;
        }

        projectedPoint = ray.origin + (ray.direction * t);
        return true;
    }

    private static float ComputeEquivalentRadius(Vector3[] polygon, int vertexCount)
    {
        float area = ComputePolygonAreaXZ(polygon, vertexCount);
        return area <= 0.0001f ? 0.1f : Mathf.Sqrt(area / Mathf.PI);
    }

    private static float ComputePolygonAreaXZ(Vector3[] polygon, int vertexCount)
    {
        if (polygon == null || vertexCount < 3)
        {
            return 0f;
        }

        float sum = 0f;
        for (int i = 0; i < vertexCount; i++)
        {
            int j = (i + 1) % vertexCount;
            sum += (polygon[i].x * polygon[j].z) - (polygon[j].x * polygon[i].z);
        }

        return Mathf.Abs(sum) * 0.5f;
    }

    private void OnDrawGizmosSelected()
    {
        if (!drawGroundFootprint || !TryUpdateCameraGroundFootprint() || !hasCameraGroundFootprint)
        {
            return;
        }

        Gizmos.color = sensorFootprintColor;
        DrawPolygon(cameraGroundFootprintCorners, sensorFootprintGroundOffset);
    }

    private static void DrawPolygon(Vector3[] polygon, float yOffset)
    {
        if (polygon == null || polygon.Length < 2)
        {
            return;
        }

        int count = polygon.Length;
        for (int i = 0; i < count; i++)
        {
            Vector3 a = polygon[i];
            Vector3 b = polygon[(i + 1) % count];
            a.y += yOffset;
            b.y += yOffset;
            Gizmos.DrawLine(a, b);
        }
    }
}
