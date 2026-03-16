using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;

using UnityEngine;

public class DroneDemoManager : MonoBehaviour
{
    private const int TotalZones = 20;
    private const int ZoneColumns = 4;
    private const int ZoneRows = 5;
    private const float FullAreaWidth = 1000f;
    private const float FullAreaDepth = 1000f;
    private const float ZoneWidth = 250f;
    private const float ZoneDepth = 200f;
    private const string DemoModelResourcePath = "DroneCoverageDemo";
    private const int DemoFootprintSegments = 24;

    private readonly List<DroneDemoMissionController> activeDrones = new List<DroneDemoMissionController>();

    private DroneCoverageAgent templateAgent;
    private AreaCoverageTracker templateTracker;
    private BoxCollider templateSearchZone;
    private Transform runtimeRoot;
    private Vector3 spawnAnchorPosition;
    private Quaternion spawnAnchorRotation;
    private Unity.InferenceEngine.ModelAsset demoModel;

    private bool initialized;
    private bool demoRunning;
    private string statusMessage = string.Empty;
    private string droneCountInput = "10";
    private string transitSpeedInput = "22";
    private string missionCenterXInput = string.Empty;
    private string missionCenterZInput = string.Empty;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        if (FindFirstObjectByType<DroneDemoManager>() != null)
        {
            return;
        }

        GameObject bootstrapObject = new GameObject("Drone Demo Manager");
        bootstrapObject.AddComponent<DroneDemoManager>();
    }

    private void Start()
    {
        InitializeIfNeeded();
    }

    private void OnDestroy()
    {
        StopDemo();
    }

    private void OnGUI()
    {
        InitializeIfNeeded();

        GUILayout.BeginArea(new Rect(12f, 12f, 360f, 270f), GUI.skin.box);
        GUILayout.Label("Drone Demo Control");
        GUILayout.Label(demoModel != null
            ? $"Model: {demoModel.name}"
            : "Model: Missing Assets/Resources/DroneCoverageDemo.onnx");

        GUILayout.Space(6f);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("10 Drones", GUILayout.Height(28f)))
        {
            droneCountInput = "10";
        }

        if (GUILayout.Button("20 Drones", GUILayout.Height(28f)))
        {
            droneCountInput = "20";
        }
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        GUILayout.Label("Drone Count", GUILayout.Width(100f));
        droneCountInput = GUILayout.TextField(droneCountInput, 3, GUILayout.Width(80f));
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        GUILayout.Label("Transit Speed", GUILayout.Width(100f));
        transitSpeedInput = GUILayout.TextField(transitSpeedInput, 6, GUILayout.Width(80f));
        GUILayout.Label("m/s");
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        GUILayout.Label("Center X", GUILayout.Width(100f));
        missionCenterXInput = GUILayout.TextField(missionCenterXInput, 10, GUILayout.Width(120f));
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        GUILayout.Label("Center Z", GUILayout.Width(100f));
        missionCenterZInput = GUILayout.TextField(missionCenterZInput, 10, GUILayout.Width(120f));
        GUILayout.EndHorizontal();

        GUILayout.Space(6f);
        if (GUILayout.Button(demoRunning ? "Restart Demo" : "Start Demo", GUILayout.Height(30f)))
        {
            StartDemo();
        }

        if (GUILayout.Button("Stop Demo", GUILayout.Height(26f)))
        {
            StopDemo();
        }

        GUILayout.Space(6f);
        GUILayout.Label($"Area: {FullAreaWidth:F0} x {FullAreaDepth:F0} meters");
        GUILayout.Label($"Cells: {TotalZones} total, each {ZoneWidth:F0} x {ZoneDepth:F0}");
        if (!string.IsNullOrWhiteSpace(statusMessage))
        {
            GUILayout.Label(statusMessage);
        }

        GUILayout.EndArea();
    }

    private void InitializeIfNeeded()
    {
        if (initialized)
        {
            return;
        }

        templateAgent = FindFirstObjectByType<DroneCoverageAgent>();
        templateTracker = FindFirstObjectByType<AreaCoverageTracker>();

        if (templateAgent == null || templateTracker == null)
        {
            statusMessage = "MainScene is missing the template drone or coverage tracker.";
            initialized = true;
            return;
        }

        templateSearchZone = templateTracker.SearchZoneCollider;
        spawnAnchorPosition = templateAgent.transform.position;
        spawnAnchorRotation = templateAgent.transform.rotation;

        Vector2 defaultMissionCenter = ResolveDefaultMissionCenter();
        missionCenterXInput = defaultMissionCenter.x.ToString("F1");
        missionCenterZInput = defaultMissionCenter.y.ToString("F1");

        DemonstrationRecorder recorder = templateAgent.GetComponent<DemonstrationRecorder>();
        if (recorder != null)
        {
            recorder.Record = false;
            recorder.enabled = false;
            Destroy(recorder);
        }

        SetTemplateObjectsActive(false);

        Academy.Instance.AutomaticSteppingEnabled = true;
        demoModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(DemoModelResourcePath);
        statusMessage = demoModel != null
            ? "Ready. Choose 10 or 20 drones and start the demo."
            : "Missing demo model. Copy the trained ONNX into Assets/Resources/DroneCoverageDemo.onnx.";

        initialized = true;
    }

    private Vector2 ResolveDefaultMissionCenter()
    {
        Terrain terrain = Terrain.activeTerrain ?? FindFirstObjectByType<Terrain>();
        if (terrain != null && terrain.terrainData != null)
        {
            Vector3 terrainPosition = terrain.GetPosition();
            Vector3 terrainSize = terrain.terrainData.size;
            return new Vector2(
                terrainPosition.x + (terrainSize.x * 0.5f),
                terrainPosition.z + (terrainSize.z * 0.5f));
        }

        Bounds defaultZone = templateTracker != null ? templateTracker.ZoneBounds : default;
        return new Vector2(defaultZone.center.x, defaultZone.center.z);
    }

    private void StartDemo()
    {
        InitializeIfNeeded();

        if (templateAgent == null || templateTracker == null)
        {
            statusMessage = "Cannot start demo because the template scene objects were not found.";
            return;
        }

        if (demoModel == null)
        {
            demoModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(DemoModelResourcePath);
        }

        if (demoModel == null)
        {
            statusMessage = "Cannot start demo until DroneCoverageDemo.onnx is available in Assets/Resources.";
            return;
        }

        if (!TryParseInputs(out int droneCount, out float transitSpeed, out Vector2 missionCenter))
        {
            return;
        }

        StopDemo();

        runtimeRoot = new GameObject("Drone Demo Runtime").transform;
        List<DroneDemoZone> zones = BuildMissionZones(missionCenter);
        List<List<DroneDemoZone>> assignments = BuildAssignments(zones, droneCount);

        for (int i = 0; i < assignments.Count; i++)
        {
            DroneDemoMissionController missionController = CreateDrone(i, assignments[i], transitSpeed);
            if (missionController != null)
            {
                activeDrones.Add(missionController);
            }
        }

        demoRunning = activeDrones.Count > 0;
        statusMessage = demoRunning
            ? $"Running demo with {activeDrones.Count} drones."
            : "No drones were created for the demo.";
    }

    private void StopDemo()
    {
        for (int i = 0; i < activeDrones.Count; i++)
        {
            if (activeDrones[i] != null)
            {
                activeDrones[i].StopMission();
            }
        }

        activeDrones.Clear();

        if (runtimeRoot != null)
        {
            Destroy(runtimeRoot.gameObject);
            runtimeRoot = null;
        }

        demoRunning = false;
    }

    private DroneDemoMissionController CreateDrone(int droneIndex, List<DroneDemoZone> assignedZones, float transitSpeed)
    {
        GameObject droneObject = Instantiate(templateAgent.gameObject, runtimeRoot);
        droneObject.name = $"Demo Drone {droneIndex + 1:00}";

        Transform droneTransform = droneObject.transform;
        Vector3 spawnPosition = GetSpawnPosition(droneIndex);
        droneTransform.SetPositionAndRotation(spawnPosition, spawnAnchorRotation);

        DemonstrationRecorder recorder = droneObject.GetComponent<DemonstrationRecorder>();
        if (recorder != null)
        {
            recorder.Record = false;
            recorder.enabled = false;
            Destroy(recorder);
        }

        DisableCloneSensorsAndCameras(droneObject);

        DroneCoverageAgent agent = droneObject.GetComponent<DroneCoverageAgent>();
        LockedAltitudeDroneController controller = droneObject.GetComponent<LockedAltitudeDroneController>();
        DecisionRequester decisionRequester = droneObject.GetComponent<DecisionRequester>();
        BehaviorParameters behaviorParameters = droneObject.GetComponent<BehaviorParameters>();

        if (controller != null)
        {
            controller.showDebugGui = false;
        }

        if (decisionRequester != null)
        {
            decisionRequester.enabled = true;
        }

        AreaCoverageTracker tracker = CreateTrackerForDrone(droneObject.name, droneTransform);
        DroneDemoMissionController missionController =
            droneObject.GetComponent<DroneDemoMissionController>() ?? droneObject.AddComponent<DroneDemoMissionController>();
        missionController.Configure(agent, controller, tracker, assignedZones, transitSpeed, droneObject.name);

        if (assignedZones.Count > 0)
        {
            Vector3 lookTarget = new Vector3(assignedZones[0].CenterXZ.x, droneTransform.position.y, assignedZones[0].CenterXZ.y);
            Vector3 toTarget = lookTarget - droneTransform.position;
            toTarget.y = 0f;
            if (toTarget.sqrMagnitude > 0.0001f)
            {
                droneTransform.rotation = Quaternion.LookRotation(toTarget.normalized, Vector3.up);
            }
        }

        droneObject.SetActive(true);

        if (behaviorParameters != null)
        {
            behaviorParameters.BehaviorName = "DroneCoverage";
            behaviorParameters.Model = demoModel;
            behaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
        }

        if (agent != null && demoModel != null)
        {
            agent.SetModel("DroneCoverage", demoModel);
            agent.RequestDecision();
        }

        missionController.StartMission();
        return missionController;
    }

    private AreaCoverageTracker CreateTrackerForDrone(string droneName, Transform droneTransform)
    {
        GameObject trackerObject = new GameObject($"{droneName} Tracker");
        trackerObject.transform.SetParent(runtimeRoot, false);

        AreaCoverageTracker tracker = trackerObject.AddComponent<AreaCoverageTracker>();
        tracker.CopySettingsFrom(templateTracker);
        tracker.ConfigureDebugVisualization(showVisitedCellFill: false, showSensorFootprintOutline: true, footprintSegments: DemoFootprintSegments);

        GameObject zoneObject = new GameObject($"{droneName} Search Zone");
        zoneObject.transform.SetParent(trackerObject.transform, false);

        BoxCollider zoneCollider = zoneObject.AddComponent<BoxCollider>();
        zoneCollider.isTrigger = true;
        zoneCollider.size = new Vector3(ZoneWidth, 200f, ZoneDepth);

        tracker.SetSearchZone(zoneCollider);
        tracker.SetSensorTransform(droneTransform);
        return tracker;
    }

    private static void DisableCloneSensorsAndCameras(GameObject droneObject)
    {
        foreach (Camera cameraComponent in droneObject.GetComponentsInChildren<Camera>(true))
        {
            cameraComponent.enabled = false;
        }

        foreach (AudioListener audioListener in droneObject.GetComponentsInChildren<AudioListener>(true))
        {
            audioListener.enabled = false;
        }
    }

    private List<DroneDemoZone> BuildMissionZones(Vector2 missionCenter)
    {
        List<DroneDemoZone> zones = new List<DroneDemoZone>(TotalZones);
        float minX = missionCenter.x - (FullAreaWidth * 0.5f) + (ZoneWidth * 0.5f);
        float minZ = missionCenter.y - (FullAreaDepth * 0.5f) + (ZoneDepth * 0.5f);

        int zoneId = 0;
        for (int row = 0; row < ZoneRows; row++)
        {
            for (int column = 0; column < ZoneColumns; column++)
            {
                Vector2 centerXZ = new Vector2(minX + (column * ZoneWidth), minZ + (row * ZoneDepth));
                zones.Add(new DroneDemoZone(zoneId, centerXZ, new Vector2(ZoneWidth, ZoneDepth)));
                zoneId++;
            }
        }

        return zones;
    }

    private static List<List<DroneDemoZone>> BuildAssignments(List<DroneDemoZone> zones, int droneCount)
    {
        List<List<DroneDemoZone>> assignments = new List<List<DroneDemoZone>>(droneCount);
        int baseZoneCount = zones.Count / droneCount;
        int remainder = zones.Count % droneCount;
        int zoneIndex = 0;

        for (int droneIndex = 0; droneIndex < droneCount; droneIndex++)
        {
            int assignedCount = baseZoneCount + (droneIndex < remainder ? 1 : 0);
            List<DroneDemoZone> droneZones = new List<DroneDemoZone>(assignedCount);

            for (int i = 0; i < assignedCount && zoneIndex < zones.Count; i++)
            {
                droneZones.Add(zones[zoneIndex]);
                zoneIndex++;
            }

            assignments.Add(droneZones);
        }

        return assignments;
    }

    private Vector3 GetSpawnPosition(int droneIndex)
    {
        int columns = 5;
        int row = droneIndex / columns;
        int column = droneIndex % columns;
        float xOffset = (column - ((columns - 1) * 0.5f)) * 40f;
        float zOffset = -(row * 40f);
        return spawnAnchorPosition + new Vector3(xOffset, 0f, zOffset);
    }

    private bool TryParseInputs(out int droneCount, out float transitSpeed, out Vector2 missionCenter)
    {
        droneCount = 0;
        transitSpeed = 0f;
        missionCenter = Vector2.zero;

        if (!int.TryParse(droneCountInput, out droneCount))
        {
            statusMessage = "Drone count must be a whole number between 1 and 20.";
            return false;
        }

        droneCount = Mathf.Clamp(droneCount, 1, TotalZones);
        droneCountInput = droneCount.ToString();

        if (!float.TryParse(transitSpeedInput, out transitSpeed))
        {
            statusMessage = "Transit speed must be a number greater than 14.";
            return false;
        }

        transitSpeed = Mathf.Max(14.5f, transitSpeed);
        transitSpeedInput = transitSpeed.ToString("F1");

        if (!float.TryParse(missionCenterXInput, out float centerX) ||
            !float.TryParse(missionCenterZInput, out float centerZ))
        {
            statusMessage = "Mission center must be valid X/Z coordinates.";
            return false;
        }

        missionCenter = new Vector2(centerX, centerZ);
        statusMessage = string.Empty;
        return true;
    }

    private void SetTemplateObjectsActive(bool active)
    {
        if (templateAgent != null)
        {
            templateAgent.gameObject.SetActive(active);
        }

        if (templateTracker != null)
        {
            templateTracker.gameObject.SetActive(active);
        }

        if (templateSearchZone != null)
        {
            templateSearchZone.gameObject.SetActive(active);
        }
    }
}
