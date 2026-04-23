using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Splines;
using float3 = Unity.Mathematics.float3;

public class TruckTrafficManager : MonoBehaviour
{
    private const float SurfaceProbeHeight = 200f;
    private const float SurfaceProbeDistance = 500f;
    private const float SurfaceOffsetPadding = 0.05f;
    private const float MinimumRouteWidth = 220f;
    private const float MinimumRouteDepth = 220f;
    private static readonly Vector2 DefaultTruckSpeedRange = new Vector2(10f, 15f);

    private readonly List<GameObject> activeTrucks = new List<GameObject>();
    private readonly List<SplineContainer> activeRoutes = new List<SplineContainer>();
    private readonly List<List<Collider>> activeTruckColliders = new List<List<Collider>>();

    private Transform trafficRoot;
    private Terrain cachedTerrain;
    private float defaultSurfaceHeight;
    private float surfaceHeightOffset;

    private readonly struct RouteProfile
    {
        public RouteProfile(Vector2 offset, Vector2 size)
        {
            Offset = offset;
            Size = size;
        }

        public Vector2 Offset { get; }
        public Vector2 Size { get; }
    }

    public IReadOnlyList<GameObject> ActiveTrucks => activeTrucks;

    public int StartTraffic(
        IReadOnlyList<GameObject> truckTemplates,
        int truckCount,
        Vector2 missionCenter,
        Vector2 missionAreaSize,
        Vector2? truckSpeedRange = null)
    {
        StopTraffic();

        if (truckTemplates == null || truckTemplates.Count == 0 || truckCount <= 0)
        {
            return 0;
        }

        cachedTerrain = Terrain.activeTerrain ?? FindFirstObjectByType<Terrain>();
        defaultSurfaceHeight = ResolveDefaultSurfaceHeight(truckTemplates);
        surfaceHeightOffset = ResolveSurfaceHeightOffset(truckTemplates);

        trafficRoot = new GameObject("Truck Traffic Runtime").transform;
        trafficRoot.SetParent(transform, false);

        BuildRoutes(missionCenter, missionAreaSize);
        if (activeRoutes.Count == 0)
        {
            return 0;
        }

        int routeCount = activeRoutes.Count;
        int[] routeUseCounts = new int[routeCount];
        int targetPerRoute = Mathf.Max(1, Mathf.CeilToInt(truckCount / (float)routeCount));
        int routeStartIndex = Random.Range(0, routeCount);
        Vector2 resolvedSpeedRange = truckSpeedRange ?? DefaultTruckSpeedRange;
        float minTruckSpeed = Mathf.Max(4f, Mathf.Min(resolvedSpeedRange.x, resolvedSpeedRange.y));
        float maxTruckSpeed = Mathf.Max(minTruckSpeed, Mathf.Max(resolvedSpeedRange.x, resolvedSpeedRange.y));

        for (int truckIndex = 0; truckIndex < truckCount; truckIndex++)
        {
            int routeIndex = (routeStartIndex + truckIndex) % routeCount;
            int slotIndex = routeUseCounts[routeIndex]++;
            float startOffset = Mathf.Repeat(
                (slotIndex / (float)targetPerRoute) + Random.Range(0f, 0.08f),
                1f);

            GameObject template = truckTemplates[truckIndex % truckTemplates.Count];
            GameObject truckObject = Instantiate(template, trafficRoot);
            truckObject.name = $"Traffic Truck {truckIndex + 1:00}";

            ConfigureTruckInstance(
                truckObject,
                activeRoutes[routeIndex],
                startOffset,
                Random.Range(minTruckSpeed, maxTruckSpeed));

            RegisterTruckColliders(truckObject);
            activeTrucks.Add(truckObject);
        }

        return activeTrucks.Count;
    }

    public void StopTraffic()
    {
        activeTrucks.Clear();
        activeRoutes.Clear();
        activeTruckColliders.Clear();

        if (trafficRoot != null)
        {
            Destroy(trafficRoot.gameObject);
            trafficRoot = null;
        }
    }

    private void BuildRoutes(Vector2 missionCenter, Vector2 missionAreaSize)
    {
        activeRoutes.Clear();

        float width = Mathf.Max(MinimumRouteWidth, missionAreaSize.x);
        float depth = Mathf.Max(MinimumRouteDepth, missionAreaSize.y);

        RouteProfile[] routeProfiles =
        {
            new RouteProfile(Vector2.zero, new Vector2(width * 0.76f, depth * 0.76f)),
            new RouteProfile(new Vector2(-width * 0.12f, depth * 0.08f), new Vector2(width * 0.54f, depth * 0.68f)),
            new RouteProfile(new Vector2(width * 0.12f, -depth * 0.08f), new Vector2(width * 0.54f, depth * 0.68f)),
            new RouteProfile(new Vector2(0f, depth * 0.2f), new Vector2(width * 0.68f, depth * 0.34f)),
            new RouteProfile(new Vector2(0f, -depth * 0.2f), new Vector2(width * 0.68f, depth * 0.34f)),
            new RouteProfile(Vector2.zero, new Vector2(width * 0.38f, depth * 0.38f))
        };

        for (int i = 0; i < routeProfiles.Length; i++)
        {
            RouteProfile profile = routeProfiles[i];
            if (profile.Size.x < MinimumRouteWidth || profile.Size.y < MinimumRouteDepth)
            {
                continue;
            }

            Vector2 routeCenter = missionCenter + profile.Offset;
            SplineContainer route = CreateClosedRoute($"Truck Route {i + 1:00}", routeCenter, profile.Size);
            if (route != null)
            {
                activeRoutes.Add(route);
            }
        }
    }

    private SplineContainer CreateClosedRoute(string routeName, Vector2 centerXZ, Vector2 sizeXZ)
    {
        if (trafficRoot == null)
        {
            return null;
        }

        float halfWidth = sizeXZ.x * 0.5f;
        float halfDepth = sizeXZ.y * 0.5f;
        List<float3> points = new List<float3>(8)
        {
            BuildRoutePoint(centerXZ.x - halfWidth, centerXZ.y - (halfDepth * 0.35f)),
            BuildRoutePoint(centerXZ.x - (halfWidth * 0.35f), centerXZ.y - halfDepth),
            BuildRoutePoint(centerXZ.x + (halfWidth * 0.35f), centerXZ.y - halfDepth),
            BuildRoutePoint(centerXZ.x + halfWidth, centerXZ.y - (halfDepth * 0.35f)),
            BuildRoutePoint(centerXZ.x + halfWidth, centerXZ.y + (halfDepth * 0.35f)),
            BuildRoutePoint(centerXZ.x + (halfWidth * 0.35f), centerXZ.y + halfDepth),
            BuildRoutePoint(centerXZ.x - (halfWidth * 0.35f), centerXZ.y + halfDepth),
            BuildRoutePoint(centerXZ.x - halfWidth, centerXZ.y + (halfDepth * 0.35f))
        };

        GameObject routeObject = new GameObject(routeName);
        routeObject.transform.SetParent(trafficRoot, false);

        SplineContainer container = routeObject.AddComponent<SplineContainer>();
        container.Spline = new Spline(points, TangentMode.AutoSmooth, true);
        return container;
    }

    private void ConfigureTruckInstance(GameObject truckObject, SplineContainer route, float startOffset, float maxSpeed)
    {
        if (truckObject == null || route == null)
        {
            return;
        }

        foreach (SplineAnimate existingAnimate in truckObject.GetComponents<SplineAnimate>())
        {
            Destroy(existingAnimate);
        }

        foreach (TruckTerrainFollower existingFollower in truckObject.GetComponents<TruckTerrainFollower>())
        {
            Destroy(existingFollower);
        }

        Rigidbody rigidbodyComponent = truckObject.GetComponent<Rigidbody>();
        if (rigidbodyComponent != null)
        {
            rigidbodyComponent.isKinematic = true;
            rigidbodyComponent.useGravity = false;
        }

        SplineAnimate splineAnimate = truckObject.AddComponent<SplineAnimate>();
        splineAnimate.Container = route;
        splineAnimate.PlayOnAwake = false;
        splineAnimate.AnimationMethod = SplineAnimate.Method.Speed;
        splineAnimate.MaxSpeed = Mathf.Max(4f, maxSpeed);
        splineAnimate.Loop = SplineAnimate.LoopMode.Loop;
        splineAnimate.Alignment = SplineAnimate.AlignmentMode.SplineElement;
        splineAnimate.Easing = SplineAnimate.EasingMode.None;
        splineAnimate.StartOffset = Mathf.Repeat(startOffset, 1f);

        TruckTerrainFollower terrainFollower = truckObject.AddComponent<TruckTerrainFollower>();
        terrainFollower.Configure(surfaceHeightOffset);

        TruckTarget truckTarget = truckObject.GetComponent<TruckTarget>() ?? truckObject.AddComponent<TruckTarget>();
        truckTarget.ResetTrackingState();

        truckObject.SetActive(true);
        splineAnimate.Restart(true);
        splineAnimate.NormalizedTime = splineAnimate.StartOffset;
    }

    private void RegisterTruckColliders(GameObject truckObject)
    {
        Collider[] colliders = truckObject.GetComponentsInChildren<Collider>(true);
        List<Collider> truckColliders = new List<Collider>(colliders.Length);

        for (int i = 0; i < colliders.Length; i++)
        {
            Collider colliderComponent = colliders[i];
            if (colliderComponent != null)
            {
                truckColliders.Add(colliderComponent);
            }
        }

        for (int groupIndex = 0; groupIndex < activeTruckColliders.Count; groupIndex++)
        {
            List<Collider> existingGroup = activeTruckColliders[groupIndex];
            for (int existingIndex = 0; existingIndex < existingGroup.Count; existingIndex++)
            {
                Collider existingCollider = existingGroup[existingIndex];
                if (existingCollider == null)
                {
                    continue;
                }

                for (int newIndex = 0; newIndex < truckColliders.Count; newIndex++)
                {
                    Collider newCollider = truckColliders[newIndex];
                    if (newCollider != null)
                    {
                        Physics.IgnoreCollision(existingCollider, newCollider, true);
                    }
                }
            }
        }

        activeTruckColliders.Add(truckColliders);
    }

    private float ResolveDefaultSurfaceHeight(IReadOnlyList<GameObject> truckTemplates)
    {
        for (int i = 0; i < truckTemplates.Count; i++)
        {
            if (truckTemplates[i] != null)
            {
                return truckTemplates[i].transform.position.y;
            }
        }

        return 0f;
    }

    private float ResolveSurfaceHeightOffset(IReadOnlyList<GameObject> truckTemplates)
    {
        float highestObservedOffset = SurfaceOffsetPadding;

        for (int i = 0; i < truckTemplates.Count; i++)
        {
            GameObject truckTemplate = truckTemplates[i];
            if (truckTemplate == null)
            {
                continue;
            }

            Vector3 templatePosition = truckTemplate.transform.position;
            if (TryResolveSurfaceHeight(templatePosition.x, templatePosition.z, out float surfaceHeight))
            {
                highestObservedOffset = Mathf.Max(
                    highestObservedOffset,
                    truckTemplate.transform.position.y - surfaceHeight + SurfaceOffsetPadding);
            }
        }

        return highestObservedOffset;
    }

    private float3 BuildRoutePoint(float x, float z)
    {
        float y = defaultSurfaceHeight;

        if (TryResolveSurfaceHeight(x, z, out float surfaceHeight))
        {
            y = surfaceHeight + surfaceHeightOffset;
        }

        return new float3(x, y, z);
    }

    private bool TryResolveSurfaceHeight(float x, float z, out float surfaceHeight)
    {
        Vector3 probeOrigin = new Vector3(x, SurfaceProbeHeight, z);
        if (Physics.Raycast(probeOrigin, Vector3.down, out RaycastHit hitInfo, SurfaceProbeDistance, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
        {
            surfaceHeight = hitInfo.point.y;
            return true;
        }

        if (cachedTerrain != null && cachedTerrain.terrainData != null)
        {
            surfaceHeight = cachedTerrain.SampleHeight(new Vector3(x, 0f, z));
            return true;
        }

        surfaceHeight = defaultSurfaceHeight;
        return false;
    }
}
