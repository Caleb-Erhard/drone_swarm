using System;
using UnityEngine;

public class AreaCoverageTracker : MonoBehaviour
{
    public enum CoverageShape
    {
        CircularRadius = 0,
        CameraGroundFootprint = 1
    }

    public struct CoverageUpdate
    {
        public int FootprintCellCount;
        public int NewlyCoveredCellCount;
        public int PreviouslyCoveredCellCount;
        public float NewlyCoveredZoneFraction;

        public float FootprintNovelty01 =>
            FootprintCellCount > 0 ? (float)NewlyCoveredCellCount / FootprintCellCount : 0f;

        public float FootprintOverlap01 =>
            FootprintCellCount > 0 ? (float)PreviouslyCoveredCellCount / FootprintCellCount : 0f;
    }

    public struct WorkloadSplit
    {
        public Vector2 FirstCenterXZ;
        public Vector2 FirstSizeXZ;
        public Vector2 SecondCenterXZ;
        public Vector2 SecondSizeXZ;
        public int FirstUnvisitedCellCount;
        public int SecondUnvisitedCellCount;
        public bool SplitAlongX;
    }

    [Header("Search Zone")]
    [SerializeField] private BoxCollider searchZone;
    [SerializeField, Min(0.25f)] private float cellSize = 1f;
    [SerializeField, Min(0.1f)] private float sensorRadius = 2.5f;

    [Header("Coverage Source")]
    [SerializeField] private CoverageShape coverageShape = CoverageShape.CameraGroundFootprint;
    [SerializeField] private Camera coverageCamera;
    [SerializeField, Min(0.1f)] private float sensorRelativePlaneDistance = 40f;

    [Header("Debug")]
    [SerializeField] private bool drawZoneBounds = true;
    [SerializeField] private bool drawVisitedCells = true;
    [SerializeField] private bool drawSensorFootprint = true;
    [SerializeField, Min(4)] private int sensorFootprintSegments = 48;
    [SerializeField] private float sensorFootprintGroundOffset = 0.08f;
    [SerializeField] private Transform sensorTransform;
    [SerializeField] private Color zoneColor = new Color(0f, 0.8f, 1f, 0.9f);
    [SerializeField] private Color visitedColor = new Color(0.2f, 1f, 0.2f, 0.35f);
    [SerializeField] private Color sensorFootprintColor = new Color(1f, 0.85f, 0.2f, 0.9f);

    private bool[] visitedCells = Array.Empty<bool>();
    private int gridWidth;
    private int gridHeight;
    private int visitedCount;
    private int totalCells;
    private Bounds zoneBounds;
    private float minX;
    private float minZ;
    private bool hasCameraGroundFootprint;
    private float currentCoverageRadius;
    private readonly Vector3[] cameraGroundFootprintCorners = new Vector3[4];

    private static readonly Vector2[] ViewportCorners =
    {
        new Vector2(0f, 0f), // bottom-left
        new Vector2(0f, 1f), // top-left
        new Vector2(1f, 1f), // top-right
        new Vector2(1f, 0f)  // bottom-right
    };

    public Bounds ZoneBounds => zoneBounds;
    public Vector2 ZoneSizeXZ => new Vector2(zoneBounds.size.x, zoneBounds.size.z);
    public float Coverage01 => totalCells > 0 ? (float)visitedCount / totalCells : 0f;
    public float SensorRadius => Mathf.Max(0.1f, currentCoverageRadius > 0f ? currentCoverageRadius : sensorRadius);
    public int TotalCells => totalCells;
    public int VisitedCells => visitedCount;
    public int UnvisitedCells => Mathf.Max(0, totalCells - visitedCount);
    public BoxCollider SearchZoneCollider => searchZone;

    private void Awake()
    {
        RebuildGrid();
    }

    private void OnValidate()
    {
        cellSize = Mathf.Max(0.25f, cellSize);
        sensorRadius = Mathf.Max(0.1f, sensorRadius);
        sensorRelativePlaneDistance = Mathf.Max(0.1f, sensorRelativePlaneDistance);
        RebuildGrid();
    }

    public void RebuildGrid()
    {
        if (searchZone == null)
        {
            visitedCells = Array.Empty<bool>();
            gridWidth = 0;
            gridHeight = 0;
            totalCells = 0;
            visitedCount = 0;
            zoneBounds = new Bounds(transform.position, Vector3.zero);
            currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
            hasCameraGroundFootprint = false;
            return;
        }

        zoneBounds = searchZone.bounds;
        minX = zoneBounds.min.x;
        minZ = zoneBounds.min.z;

        gridWidth = Mathf.Max(1, Mathf.CeilToInt(zoneBounds.size.x / cellSize));
        gridHeight = Mathf.Max(1, Mathf.CeilToInt(zoneBounds.size.z / cellSize));
        totalCells = gridWidth * gridHeight;
        visitedCells = new bool[totalCells];
        visitedCount = 0;
        currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
        hasCameraGroundFootprint = false;
    }

    public void ResetCoverage()
    {
        if (totalCells <= 0 || visitedCells.Length != totalCells)
        {
            RebuildGrid();
        }

        Array.Clear(visitedCells, 0, visitedCells.Length);
        visitedCount = 0;
    }

    public float MarkCoverage(Vector3 worldPosition)
    {
        return MarkCoverageDetailed(worldPosition).NewlyCoveredZoneFraction;
    }

    public CoverageUpdate MarkCoverageDetailed(Vector3 worldPosition)
    {
        if (totalCells <= 0)
        {
            return default;
        }

        if (coverageShape == CoverageShape.CameraGroundFootprint && TryUpdateCameraGroundFootprint())
        {
            return MarkCoverageFromPolygon(cameraGroundFootprintCorners, cameraGroundFootprintCorners.Length);
        }

        hasCameraGroundFootprint = false;
        currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
        return MarkCoverageFromCircle(worldPosition, currentCoverageRadius);
    }

    private CoverageUpdate MarkCoverageFromCircle(Vector3 worldPosition, float radius)
    {
        radius = Mathf.Max(0.1f, radius);
        currentCoverageRadius = radius;

        int minCellX = WorldToGridX(worldPosition.x - radius);
        int maxCellX = WorldToGridX(worldPosition.x + radius);
        int minCellZ = WorldToGridZ(worldPosition.z - radius);
        int maxCellZ = WorldToGridZ(worldPosition.z + radius);

        float radiusSqr = radius * radius;
        int newCells = 0;
        int previouslyCoveredCells = 0;
        int footprintCellCount = 0;

        for (int z = minCellZ; z <= maxCellZ; z++)
        {
            for (int x = minCellX; x <= maxCellX; x++)
            {
                Vector3 cellCenter = GetCellCenterWorld(x, z);
                float dx = cellCenter.x - worldPosition.x;
                float dz = cellCenter.z - worldPosition.z;

                if ((dx * dx) + (dz * dz) > radiusSqr)
                {
                    continue;
                }

                footprintCellCount++;
                int index = GridToIndex(x, z);
                if (visitedCells[index])
                {
                    previouslyCoveredCells++;
                    continue;
                }

                visitedCells[index] = true;
                newCells++;
            }
        }

        visitedCount += newCells;
        return new CoverageUpdate
        {
            FootprintCellCount = footprintCellCount,
            NewlyCoveredCellCount = newCells,
            PreviouslyCoveredCellCount = previouslyCoveredCells,
            NewlyCoveredZoneFraction = totalCells > 0 ? (float)newCells / totalCells : 0f
        };
    }

    private CoverageUpdate MarkCoverageFromPolygon(Vector3[] polygon, int vertexCount)
    {
        if (polygon == null || vertexCount < 3)
        {
            return default;
        }

        float minFootprintX = polygon[0].x;
        float maxFootprintX = polygon[0].x;
        float minFootprintZ = polygon[0].z;
        float maxFootprintZ = polygon[0].z;

        for (int i = 1; i < vertexCount; i++)
        {
            Vector3 p = polygon[i];
            minFootprintX = Mathf.Min(minFootprintX, p.x);
            maxFootprintX = Mathf.Max(maxFootprintX, p.x);
            minFootprintZ = Mathf.Min(minFootprintZ, p.z);
            maxFootprintZ = Mathf.Max(maxFootprintZ, p.z);
        }

        int minCellX = WorldToGridX(minFootprintX);
        int maxCellX = WorldToGridX(maxFootprintX);
        int minCellZ = WorldToGridZ(minFootprintZ);
        int maxCellZ = WorldToGridZ(maxFootprintZ);

        int newCells = 0;
        int previouslyCoveredCells = 0;
        int footprintCellCount = 0;

        for (int z = minCellZ; z <= maxCellZ; z++)
        {
            for (int x = minCellX; x <= maxCellX; x++)
            {
                Vector3 cellCenter = GetCellCenterWorld(x, z);
                if (!IsPointInsidePolygonXZ(cellCenter.x, cellCenter.z, polygon, vertexCount))
                {
                    continue;
                }

                footprintCellCount++;
                int index = GridToIndex(x, z);
                if (visitedCells[index])
                {
                    previouslyCoveredCells++;
                    continue;
                }

                visitedCells[index] = true;
                newCells++;
            }
        }

        visitedCount += newCells;
        if (footprintCellCount > 0)
        {
            float effectiveArea = footprintCellCount * cellSize * cellSize;
            currentCoverageRadius = Mathf.Sqrt(effectiveArea / Mathf.PI);
        }
        else
        {
            currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
        }

        return new CoverageUpdate
        {
            FootprintCellCount = footprintCellCount,
            NewlyCoveredCellCount = newCells,
            PreviouslyCoveredCellCount = previouslyCoveredCells,
            NewlyCoveredZoneFraction = totalCells > 0 ? (float)newCells / totalCells : 0f
        };
    }

    public bool IsCoveredAtWorldPosition(Vector3 worldPosition)
    {
        if (!IsInsideZone(worldPosition) || totalCells <= 0)
        {
            return false;
        }

        int x = WorldToGridX(worldPosition.x);
        int z = WorldToGridZ(worldPosition.z);
        return visitedCells[GridToIndex(x, z)];
    }

    public float GetCoverageSample(Vector3 worldPosition)
    {
        if (!IsInsideZone(worldPosition))
        {
            return -1f;
        }

        return IsCoveredAtWorldPosition(worldPosition) ? 1f : 0f;
    }

    public Vector4 GetNormalizedDistancesToBounds(Vector3 worldPosition)
    {
        if (totalCells <= 0)
        {
            return Vector4.zero;
        }

        float width = Mathf.Max(0.001f, zoneBounds.size.x);
        float depth = Mathf.Max(0.001f, zoneBounds.size.z);

        float toMinX = Mathf.Clamp01((worldPosition.x - zoneBounds.min.x) / width);
        float toMaxX = Mathf.Clamp01((zoneBounds.max.x - worldPosition.x) / width);
        float toMinZ = Mathf.Clamp01((worldPosition.z - zoneBounds.min.z) / depth);
        float toMaxZ = Mathf.Clamp01((zoneBounds.max.z - worldPosition.z) / depth);

        return new Vector4(toMinX, toMaxX, toMinZ, toMaxZ);
    }

    public Vector3[] GetSensorFootprintCorners()
    {
        if (coverageShape != CoverageShape.CameraGroundFootprint)
        {
            return null;
        }

        return TryUpdateCameraGroundFootprint() ? cameraGroundFootprintCorners : null;
    }

    public bool IsPointInsideSensorFootprint(Vector3 worldPoint)
    {
        if (sensorTransform == null)
        {
            return false;
        }

        if (coverageShape == CoverageShape.CameraGroundFootprint && TryUpdateCameraGroundFootprint() && hasCameraGroundFootprint)
        {
            return IsPointInsidePolygonXZ(worldPoint.x, worldPoint.z, cameraGroundFootprintCorners, cameraGroundFootprintCorners.Length);
        }

        Vector3 sensorPosition = sensorTransform.position;
        float dx = worldPoint.x - sensorPosition.x;
        float dz = worldPoint.z - sensorPosition.z;
        float radius = Mathf.Max(0.1f, currentCoverageRadius > 0f ? currentCoverageRadius : sensorRadius);
        return (dx * dx) + (dz * dz) <= radius * radius;
    }

    public void SetSensorTransform(Transform sensor)
    {
        sensorTransform = sensor;
        coverageCamera = sensorTransform != null
            ? sensorTransform.GetComponentInChildren<Camera>(true)
            : null;
    }

    public void SetSearchZone(BoxCollider zone)
    {
        searchZone = zone;
        RebuildGrid();
    }

    public void ConfigureDebugVisualization(
        bool showVisitedCellFill,
        bool showSensorFootprintOutline,
        int footprintSegments = -1,
        bool showZoneBounds = true)
    {
        drawZoneBounds = showZoneBounds;
        drawVisitedCells = showVisitedCellFill;
        drawSensorFootprint = showSensorFootprintOutline;

        if (footprintSegments > 0)
        {
            sensorFootprintSegments = Mathf.Max(4, footprintSegments);
        }
    }

    public void CopySettingsFrom(AreaCoverageTracker source)
    {
        if (source == null || ReferenceEquals(source, this))
        {
            return;
        }

        cellSize = source.cellSize;
        sensorRadius = source.sensorRadius;
        coverageShape = source.coverageShape;
        sensorRelativePlaneDistance = source.sensorRelativePlaneDistance;
        drawZoneBounds = source.drawZoneBounds;
        drawVisitedCells = source.drawVisitedCells;
        drawSensorFootprint = source.drawSensorFootprint;
        sensorFootprintSegments = source.sensorFootprintSegments;
        sensorFootprintGroundOffset = source.sensorFootprintGroundOffset;
        zoneColor = source.zoneColor;
        visitedColor = source.visitedColor;
        sensorFootprintColor = source.sensorFootprintColor;
        coverageCamera = null;
        sensorTransform = null;
        RebuildGrid();
    }

    public void ConfigureSearchZone(Vector2 centerXZ, Vector2 sizeXZ)
    {
        if (searchZone == null)
        {
            return;
        }

        Transform zoneTransform = searchZone.transform;
        Vector3 zonePosition = zoneTransform.position;
        zonePosition.x = centerXZ.x;
        zonePosition.z = centerXZ.y;
        zoneTransform.position = zonePosition;

        Vector3 lossyScale = zoneTransform.lossyScale;
        Vector3 colliderSize = searchZone.size;
        colliderSize.x = Mathf.Max(cellSize, sizeXZ.x) / Mathf.Max(0.001f, Mathf.Abs(lossyScale.x));
        colliderSize.z = Mathf.Max(cellSize, sizeXZ.y) / Mathf.Max(0.001f, Mathf.Abs(lossyScale.z));
        searchZone.size = colliderSize;

        RebuildGrid();
    }

    public bool IsInsideZone(Vector3 worldPosition)
    {
        if (totalCells <= 0)
        {
            return false;
        }

        return worldPosition.x >= zoneBounds.min.x &&
               worldPosition.x <= zoneBounds.max.x &&
               worldPosition.z >= zoneBounds.min.z &&
               worldPosition.z <= zoneBounds.max.z;
    }

    public float DistanceOutsideZone(Vector3 worldPosition)
    {
        if (totalCells <= 0)
        {
            return 0f;
        }

        float dx = Mathf.Max(zoneBounds.min.x - worldPosition.x, 0f, worldPosition.x - zoneBounds.max.x);
        float dz = Mathf.Max(zoneBounds.min.z - worldPosition.z, 0f, worldPosition.z - zoneBounds.max.z);
        return Mathf.Sqrt((dx * dx) + (dz * dz));
    }

    public Vector2 GetNormalizedZonePosition(Vector3 worldPosition)
    {
        if (totalCells <= 0)
        {
            return Vector2.zero;
        }

        float x = Mathf.InverseLerp(zoneBounds.min.x, zoneBounds.max.x, worldPosition.x) * 2f - 1f;
        float z = Mathf.InverseLerp(zoneBounds.min.z, zoneBounds.max.z, worldPosition.z) * 2f - 1f;
        return new Vector2(Mathf.Clamp(x, -1f, 1f), Mathf.Clamp(z, -1f, 1f));
    }

    public bool TryGetNearestUnvisitedDirection(Vector3 worldPosition, out Vector3 directionToTarget, out float normalizedDistance)
    {
        directionToTarget = Vector3.zero;
        normalizedDistance = 0f;

        if (totalCells <= 0 || visitedCount >= totalCells)
        {
            return false;
        }

        float bestDistanceSqr = float.MaxValue;
        Vector3 bestCellPosition = Vector3.zero;
        bool found = false;

        for (int z = 0; z < gridHeight; z++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                int index = GridToIndex(x, z);
                if (visitedCells[index])
                {
                    continue;
                }

                Vector3 cellCenter = GetCellCenterWorld(x, z);
                float dx = cellCenter.x - worldPosition.x;
                float dz = cellCenter.z - worldPosition.z;
                float distanceSqr = (dx * dx) + (dz * dz);

                if (distanceSqr >= bestDistanceSqr)
                {
                    continue;
                }

                bestDistanceSqr = distanceSqr;
                bestCellPosition = cellCenter;
                found = true;
            }
        }

        if (!found)
        {
            return false;
        }

        Vector3 toTarget = bestCellPosition - worldPosition;
        toTarget.y = 0f;
        float distance = toTarget.magnitude;
        if (distance > 0.0001f)
        {
            directionToTarget = toTarget / distance;
        }
        else
        {
            directionToTarget = Vector3.zero;
        }

        float zoneDiagonal = Mathf.Sqrt((zoneBounds.size.x * zoneBounds.size.x) + (zoneBounds.size.z * zoneBounds.size.z));
        normalizedDistance = zoneDiagonal > 0.001f ? Mathf.Clamp01(distance / zoneDiagonal) : 0f;
        return true;
    }

    public bool TryGetWorkloadBalancedSplit(out WorkloadSplit split, bool preferLongAxis = true)
    {
        split = default;
        if (totalCells < 2 || visitedCells.Length != totalCells)
        {
            return false;
        }

        bool canSplitX = gridWidth >= 2;
        bool canSplitZ = gridHeight >= 2;
        if (!canSplitX && !canSplitZ)
        {
            return false;
        }

        int[] unvisitedPerColumn = canSplitX ? new int[gridWidth] : Array.Empty<int>();
        int[] unvisitedPerRow = canSplitZ ? new int[gridHeight] : Array.Empty<int>();
        for (int z = 0; z < gridHeight; z++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                if (visitedCells[GridToIndex(x, z)])
                {
                    continue;
                }

                if (canSplitX)
                {
                    unvisitedPerColumn[x]++;
                }

                if (canSplitZ)
                {
                    unvisitedPerRow[z]++;
                }
            }
        }

        int totalUnvisited = UnvisitedCells;
        SplitCandidate splitX = canSplitX
            ? BuildBestSplitCandidate(unvisitedPerColumn, totalUnvisited, true)
            : SplitCandidate.Invalid;
        SplitCandidate splitZ = canSplitZ
            ? BuildBestSplitCandidate(unvisitedPerRow, totalUnvisited, false)
            : SplitCandidate.Invalid;

        SplitCandidate bestSplit = ChooseBetterSplit(splitX, splitZ, preferLongAxis);
        if (!bestSplit.IsValid)
        {
            return false;
        }

        if (bestSplit.SplitAlongX)
        {
            float firstWidth = zoneBounds.size.x * ((bestSplit.FirstSliceCount) / (float)gridWidth);
            float secondWidth = zoneBounds.size.x - firstWidth;
            if (firstWidth <= 0f || secondWidth <= 0f)
            {
                return false;
            }

            split = new WorkloadSplit
            {
                FirstCenterXZ = new Vector2(zoneBounds.min.x + (firstWidth * 0.5f), zoneBounds.center.z),
                FirstSizeXZ = new Vector2(firstWidth, zoneBounds.size.z),
                SecondCenterXZ = new Vector2(zoneBounds.min.x + firstWidth + (secondWidth * 0.5f), zoneBounds.center.z),
                SecondSizeXZ = new Vector2(secondWidth, zoneBounds.size.z),
                FirstUnvisitedCellCount = bestSplit.FirstUnvisitedCells,
                SecondUnvisitedCellCount = bestSplit.SecondUnvisitedCells,
                SplitAlongX = true
            };
            return true;
        }

        float firstDepth = zoneBounds.size.z * ((bestSplit.FirstSliceCount) / (float)gridHeight);
        float secondDepth = zoneBounds.size.z - firstDepth;
        if (firstDepth <= 0f || secondDepth <= 0f)
        {
            return false;
        }

        split = new WorkloadSplit
        {
            FirstCenterXZ = new Vector2(zoneBounds.center.x, zoneBounds.min.z + (firstDepth * 0.5f)),
            FirstSizeXZ = new Vector2(zoneBounds.size.x, firstDepth),
            SecondCenterXZ = new Vector2(zoneBounds.center.x, zoneBounds.min.z + firstDepth + (secondDepth * 0.5f)),
            SecondSizeXZ = new Vector2(zoneBounds.size.x, secondDepth),
            FirstUnvisitedCellCount = bestSplit.FirstUnvisitedCells,
            SecondUnvisitedCellCount = bestSplit.SecondUnvisitedCells,
            SplitAlongX = false
        };
        return true;
    }

    private readonly struct SplitCandidate
    {
        public static SplitCandidate Invalid => default;

        public readonly bool IsValid;
        public readonly bool SplitAlongX;
        public readonly int FirstSliceCount;
        public readonly int FirstUnvisitedCells;
        public readonly int SecondUnvisitedCells;
        public readonly int UnvisitedDelta;
        public readonly int SliceDelta;

        public SplitCandidate(
            bool splitAlongX,
            int firstSliceCount,
            int firstUnvisitedCells,
            int secondUnvisitedCells,
            int unvisitedDelta,
            int sliceDelta)
        {
            IsValid = true;
            SplitAlongX = splitAlongX;
            FirstSliceCount = firstSliceCount;
            FirstUnvisitedCells = firstUnvisitedCells;
            SecondUnvisitedCells = secondUnvisitedCells;
            UnvisitedDelta = unvisitedDelta;
            SliceDelta = sliceDelta;
        }
    }

    private static SplitCandidate BuildBestSplitCandidate(int[] unvisitedPerSlice, int totalUnvisited, bool splitAlongX)
    {
        if (unvisitedPerSlice == null || unvisitedPerSlice.Length < 2)
        {
            return SplitCandidate.Invalid;
        }

        int running = 0;
        int bestFirstSliceCount = 1;
        int bestFirstUnvisited = 0;
        int bestSecondUnvisited = totalUnvisited;
        int bestUnvisitedDelta = int.MaxValue;
        int bestSliceDelta = int.MaxValue;

        for (int i = 0; i < unvisitedPerSlice.Length - 1; i++)
        {
            running += unvisitedPerSlice[i];
            int firstUnvisited = running;
            int secondUnvisited = totalUnvisited - firstUnvisited;
            int unvisitedDelta = Mathf.Abs(firstUnvisited - secondUnvisited);

            int firstSliceCount = i + 1;
            int secondSliceCount = unvisitedPerSlice.Length - firstSliceCount;
            int sliceDelta = Mathf.Abs(firstSliceCount - secondSliceCount);

            bool betterDelta = unvisitedDelta < bestUnvisitedDelta;
            bool tieWithBetterGeometry = unvisitedDelta == bestUnvisitedDelta && sliceDelta < bestSliceDelta;
            if (!betterDelta && !tieWithBetterGeometry)
            {
                continue;
            }

            bestFirstSliceCount = firstSliceCount;
            bestFirstUnvisited = firstUnvisited;
            bestSecondUnvisited = secondUnvisited;
            bestUnvisitedDelta = unvisitedDelta;
            bestSliceDelta = sliceDelta;
        }

        return new SplitCandidate(
            splitAlongX,
            bestFirstSliceCount,
            bestFirstUnvisited,
            bestSecondUnvisited,
            bestUnvisitedDelta,
            bestSliceDelta);
    }

    private SplitCandidate ChooseBetterSplit(SplitCandidate splitX, SplitCandidate splitZ, bool preferLongAxis)
    {
        if (!splitX.IsValid)
        {
            return splitZ;
        }

        if (!splitZ.IsValid)
        {
            return splitX;
        }

        if (splitX.UnvisitedDelta < splitZ.UnvisitedDelta)
        {
            return splitX;
        }

        if (splitZ.UnvisitedDelta < splitX.UnvisitedDelta)
        {
            return splitZ;
        }

        float axisDiff = zoneBounds.size.x - zoneBounds.size.z;
        if (Mathf.Abs(axisDiff) > 0.001f)
        {
            bool xIsLongAxis = axisDiff > 0f;
            if (preferLongAxis)
            {
                return xIsLongAxis ? splitX : splitZ;
            }

            return xIsLongAxis ? splitZ : splitX;
        }

        if (splitX.SliceDelta < splitZ.SliceDelta)
        {
            return splitX;
        }

        if (splitZ.SliceDelta < splitX.SliceDelta)
        {
            return splitZ;
        }

        return splitX;
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
                currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
                return false;
            }

            cameraGroundFootprintCorners[i] = groundPoint;
        }

        hasCameraGroundFootprint = true;
        UpdateCoverageRadiusFromFootprint(cameraGroundFootprintCorners, cameraGroundFootprintCorners.Length);
        return true;
    }

    private Camera ResolveCoverageCamera()
    {
        if (coverageCamera != null)
        {
            return coverageCamera;
        }

        if (sensorTransform != null)
        {
            coverageCamera = sensorTransform.GetComponentInChildren<Camera>(true);
        }

        return coverageCamera;
    }

    private bool TryProjectViewportToGround(Camera camera, Vector2 viewportPoint, out Vector3 groundPoint)
    {
        if (sensorTransform == null)
        {
            groundPoint = default;
            return false;
        }

        Ray ray = camera.ViewportPointToRay(new Vector3(viewportPoint.x, viewportPoint.y, 0f));
        float planeY = sensorTransform.position.y - sensorRelativePlaneDistance;
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

    private void UpdateCoverageRadiusFromFootprint(Vector3[] polygon, int vertexCount)
    {
        float area = ComputePolygonAreaXZ(polygon, vertexCount);
        if (area <= 0.0001f)
        {
            currentCoverageRadius = Mathf.Max(0.1f, sensorRadius);
            return;
        }

        currentCoverageRadius = Mathf.Sqrt(area / Mathf.PI);
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

    private static bool IsPointInsidePolygonXZ(float x, float z, Vector3[] polygon, int vertexCount)
    {
        bool inside = false;
        for (int i = 0, j = vertexCount - 1; i < vertexCount; j = i++)
        {
            float xi = polygon[i].x;
            float zi = polygon[i].z;
            float xj = polygon[j].x;
            float zj = polygon[j].z;

            bool edgeCrosses = ((zi > z) != (zj > z));
            if (!edgeCrosses)
            {
                continue;
            }

            float zDelta = Mathf.Abs(zj - zi) < 0.00001f ? 0.00001f : (zj - zi);
            float xIntersection = ((xj - xi) * (z - zi) / zDelta) + xi;
            if (x < xIntersection)
            {
                inside = !inside;
            }
        }

        return inside;
    }

    private int WorldToGridX(float worldX)
    {
        int x = Mathf.FloorToInt((worldX - minX) / cellSize);
        return Mathf.Clamp(x, 0, gridWidth - 1);
    }

    private int WorldToGridZ(float worldZ)
    {
        int z = Mathf.FloorToInt((worldZ - minZ) / cellSize);
        return Mathf.Clamp(z, 0, gridHeight - 1);
    }

    private int GridToIndex(int x, int z)
    {
        return (z * gridWidth) + x;
    }

    private Vector3 GetCellCenterWorld(int x, int z)
    {
        float xWorld = minX + ((x + 0.5f) * cellSize);
        float zWorld = minZ + ((z + 0.5f) * cellSize);
        return new Vector3(xWorld, zoneBounds.center.y, zWorld);
    }

    private void OnDrawGizmos()
    {
        if (searchZone == null)
        {
            return;
        }

        Bounds drawBounds = searchZone.bounds;
        if (drawZoneBounds)
        {
            Gizmos.color = zoneColor;
            Gizmos.DrawWireCube(drawBounds.center, drawBounds.size);
        }

        if (drawVisitedCells && totalCells > 0 && visitedCells.Length == totalCells)
        {
            Vector3 cellDrawSize = new Vector3(cellSize * 0.9f, 0.02f, cellSize * 0.9f);
            Gizmos.color = visitedColor;
            for (int z = 0; z < gridHeight; z++)
            {
                for (int x = 0; x < gridWidth; x++)
                {
                    int index = GridToIndex(x, z);
                    if (!visitedCells[index])
                    {
                        continue;
                    }

                    Vector3 center = GetCellCenterWorld(x, z);
                    center.y = drawBounds.min.y + 0.05f;
                    Gizmos.DrawCube(center, cellDrawSize);
                }
            }
        }

        if (drawSensorFootprint && sensorTransform != null && sensorRadius > 0f)
        {
            Gizmos.color = sensorFootprintColor;
            if (coverageShape == CoverageShape.CameraGroundFootprint && TryUpdateCameraGroundFootprint() && hasCameraGroundFootprint)
            {
                DrawPolygon(cameraGroundFootprintCorners, sensorFootprintGroundOffset);
            }
            else
            {
                Vector3 center = sensorTransform.position;
                center.y = drawBounds.min.y + sensorFootprintGroundOffset;
                DrawCircle(center, Mathf.Max(0.1f, sensorRadius), sensorFootprintSegments);
            }
        }
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

    private static void DrawCircle(Vector3 center, float radius, int segments)
    {
        int clampedSegments = Mathf.Max(4, segments);
        float step = (Mathf.PI * 2f) / clampedSegments;

        Vector3 previous = center + new Vector3(radius, 0f, 0f);
        for (int i = 1; i <= clampedSegments; i++)
        {
            float angle = i * step;
            Vector3 next = center + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(previous, next);
            previous = next;
        }
    }
}
