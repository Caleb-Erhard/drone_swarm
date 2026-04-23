using System.Collections.Generic;
using Unity.MLAgents;
using Unity.MLAgents.Demonstrations;
using Unity.MLAgents.Policies;
using UnityEngine;
using UnityEngine.SceneManagement;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class DroneDemoManager : MonoBehaviour
{
    private const int TotalZones = 20;
    private const int ZoneColumns = 4;
    private const int ZoneRows = 5;
    private const float FullAreaWidth = 1000f;
    private const float FullAreaDepth = 1000f;
    private const float ZoneWidth = 250f;
    private const float ZoneDepth = 200f;
    private const string SearchModelResourcePath = "DroneCoverageDemo";
    private const string TrackingModelResourcePath = "DroneTrackingDemo";
    // Package naming is reversed: ZIL130_NoCover is the visually covered truck variant.
    private const string CoverTruckTemplateName = "ZIL130_NoCover";
    private const string CoverTruckPrefabPath = "Assets/ZIL130_MilitaryTruck/Prefabs/ZIL130_NoCover.prefab";
    private const string RuntimeDroneLayerName = "Ignore Raycast";
    private const int DemoFootprintSegments = 24;
    private const float MinimumTransitSpeed = 1f;
    private const float DetectionScanInterval = 0.05f;
    private const float TransitTruckClaimUnlockDelay = 5f;
    private const float SearchDetectionViewportPadding = 0.12f;
    private const float TrackingHandoffMinDistance = 18f;
    private const float TrackingHandoffPreferredDistance = 55f;
    private const float TrackingHandoffMaxDistance = 130f;
    private const float TrackingHandoffViewportPadding = 0.08f;
    private static readonly Vector2 MainSceneTruckSpeedRange = new Vector2(10f, 12f);
    private static readonly Color MissionZoneColor = new Color(0f, 0.8f, 1f, 0.9f);

    private sealed class RuntimeDroneUnit
    {
        public int DroneId;
        public string Label = string.Empty;
        public GameObject SearchObject;
        public DroneCoverageAgent SearchAgent;
        public LockedAltitudeDroneController SearchController;
        public Rigidbody SearchBody;
        public AreaCoverageTracker SearchTracker;
        public DroneDemoMissionController SearchMission;
        public DroneTrackingSensorFootprint SearchFootprint;
        public GameObject TrackingObject;
        public DroneTrackingAgent TrackingAgent;
        public LockedAltitudeDroneController TrackingController;
        public Rigidbody TrackingBody;
        public DroneTrackingSensorFootprint TrackingFootprint;
        public Transform SharedSensorCameraTransform;
        public float BaseTrackingMaxSpeed;
        public float BaseTrackingAcceleration;
        public float BaseTrackingTurnSpeed;
        public TruckTarget ClaimedTruck;
        public readonly List<DroneDemoZone> AssignedZones = new List<DroneDemoZone>();

        public bool IsTracking => ClaimedTruck != null && TrackingObject != null && TrackingObject.activeSelf;

        public Vector3 CurrentPosition
        {
            get
            {
                if (IsTracking && TrackingObject != null)
                {
                    return TrackingObject.transform.position;
                }

                return SearchObject != null
                    ? SearchObject.transform.position
                    : Vector3.zero;
            }
        }
    }

    private readonly List<RuntimeDroneUnit> activeDroneUnits = new List<RuntimeDroneUnit>();
    private readonly List<GameObject> truckTemplates = new List<GameObject>();
    private readonly List<TruckTarget> activeTruckTargets = new List<TruckTarget>();
    private readonly List<DroneDemoZone> missionZones = new List<DroneDemoZone>(TotalZones);

    private DroneCoverageAgent templateAgent;
    private AreaCoverageTracker templateTracker;
    private BoxCollider templateSearchZone;
    private Transform runtimeRoot;
    private TruckTrafficManager truckTrafficManager;
    private Vector3 spawnAnchorPosition;
    private Quaternion spawnAnchorRotation;
    private Unity.InferenceEngine.ModelAsset searchModel;
    private Unity.InferenceEngine.ModelAsset trackingModel;

    private bool initialized;
    private bool demoRunning;
    private bool matchTruckCountToDrones = true;
    private string statusMessage = string.Empty;
    private string droneCountInput = "10";
    private string truckCountInput = "10";
    private string transitSpeedInput = "22";
    private string missionCenterXInput = string.Empty;
    private string missionCenterZInput = string.Empty;
    private float nextDetectionScanTime;
    private float demoStartTime;
    private float activeTransitSpeed = 22f;
    private Vector2 activeMissionCenter;
    private Vector2 controlPanelScroll;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        Scene activeScene = SceneManager.GetActiveScene();
        if (!activeScene.IsValid() || activeScene.name != "MainScene")
        {
            return;
        }

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

    private void Update()
    {
        if (!demoRunning)
        {
            return;
        }

        RecoverInvalidUnitStates();

        if (Time.time >= nextDetectionScanTime)
        {
            nextDetectionScanTime = Time.time + DetectionScanInterval;
            ScanForTruckDetections();
        }

        UpdateRuntimeStatusMessage();
    }

    private void OnDestroy()
    {
        StopDemo();
    }

    private void OnGUI()
    {
        InitializeIfNeeded();

        float screenHeight = Mathf.Max(240f, Screen.height - 24f);
        float controlPanelHeight = Mathf.Min(520f, screenHeight);
        GUILayout.BeginArea(new Rect(12f, 12f, 420f, controlPanelHeight), GUI.skin.box);
        controlPanelScroll = GUILayout.BeginScrollView(controlPanelScroll, false, true);
        GUILayout.Label("Drone Demo Control");
        GUILayout.Label(searchModel != null
            ? $"Search Model: {searchModel.name}"
            : "Search Model: Missing Assets/Resources/DroneCoverageDemo.onnx");
        GUILayout.Label(trackingModel != null
            ? $"Tracking Model: {trackingModel.name}"
            : "Tracking Model: Missing Assets/Resources/DroneTrackingDemo.onnx");

        GUILayout.Space(6f);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("10 Drones", GUILayout.Height(28f)))
        {
            droneCountInput = "10";
            if (matchTruckCountToDrones)
            {
                truckCountInput = droneCountInput;
            }
        }

        if (GUILayout.Button("20 Drones", GUILayout.Height(28f)))
        {
            droneCountInput = "20";
            if (matchTruckCountToDrones)
            {
                truckCountInput = droneCountInput;
            }
        }
        GUILayout.EndHorizontal();

        GUILayout.BeginHorizontal();
        GUILayout.Label("Drone Count", GUILayout.Width(100f));
        droneCountInput = GUILayout.TextField(droneCountInput, 3, GUILayout.Width(80f));
        GUILayout.EndHorizontal();

        matchTruckCountToDrones = GUILayout.Toggle(matchTruckCountToDrones, "Match truck count to drone count");
        if (matchTruckCountToDrones)
        {
            truckCountInput = droneCountInput;
            GUILayout.Label($"Truck Count: {truckCountInput}");
        }
        else
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label("Truck Count", GUILayout.Width(100f));
            truckCountInput = GUILayout.TextField(truckCountInput, 3, GUILayout.Width(80f));
            GUILayout.EndHorizontal();
        }

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
        GUILayout.Label($"Truck Speed: {MainSceneTruckSpeedRange.x:F0}-{MainSceneTruckSpeedRange.y:F0} m/s");

        if (demoRunning)
        {
            int trackingCount = CountTrackingUnits();
            int searchingCount = activeDroneUnits.Count - trackingCount;
            GUILayout.Label($"Search Drones: {searchingCount}  Tracking Drones: {trackingCount}");
            GUILayout.Label($"Claimed Trucks: {CountClaimedTrucks()} / {activeTruckTargets.Count}");
            DrawTrackingDebugPanel();
        }

        if (!string.IsNullOrWhiteSpace(statusMessage))
        {
            GUILayout.Label(statusMessage);
        }

        GUILayout.EndScrollView();
        GUILayout.EndArea();

        if (demoRunning)
        {
            DrawTrackingDebugOverlay();
        }
    }

    private void InitializeIfNeeded()
    {
        if (initialized)
        {
            return;
        }

        templateAgent = FindFirstObjectByType<DroneCoverageAgent>();
        templateTracker = FindFirstObjectByType<AreaCoverageTracker>();
        DiscoverTruckTemplates();

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
        searchModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(SearchModelResourcePath);
        trackingModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(TrackingModelResourcePath);

        if (searchModel == null)
        {
            statusMessage = "Missing search model. Copy the trained ONNX into Assets/Resources/DroneCoverageDemo.onnx.";
        }
        else if (trackingModel == null)
        {
            statusMessage = "Missing tracking model. Copy the trained ONNX into Assets/Resources/DroneTrackingDemo.onnx.";
        }
        else if (truckTemplates.Count == 0)
        {
            statusMessage = "Ready for drones, but truck templates were not found in MainScene.";
        }
        else
        {
            statusMessage = "Ready. Choose 10 or 20 drones, set transit speed, and start the demo.";
        }

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

        if (searchModel == null)
        {
            searchModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(SearchModelResourcePath);
        }

        if (trackingModel == null)
        {
            trackingModel = Resources.Load<Unity.InferenceEngine.ModelAsset>(TrackingModelResourcePath);
        }

        if (searchModel == null || trackingModel == null)
        {
            statusMessage = "Cannot start demo until both DroneCoverageDemo.onnx and DroneTrackingDemo.onnx are available in Assets/Resources.";
            return;
        }

        if (!TryParseInputs(out int droneCount, out int truckCount, out float transitSpeed, out Vector2 missionCenter))
        {
            return;
        }

        StopDemo();

        runtimeRoot = new GameObject("Drone Demo Runtime").transform;
        runtimeRoot.SetParent(transform, false);
        ConfigureRuntimeDronePhysics();

        activeMissionCenter = missionCenter;
        activeTransitSpeed = transitSpeed;
        demoStartTime = Time.time;
        missionZones.Clear();
        missionZones.AddRange(BuildMissionZones(missionCenter));
        CreateMissionZoneVisualizers(missionZones);

        for (int i = 0; i < droneCount; i++)
        {
            RuntimeDroneUnit droneUnit = CreateRuntimeDroneUnit(i);
            if (droneUnit != null)
            {
                activeDroneUnits.Add(droneUnit);
            }
        }

        int activeTruckCount = 0;
        if (truckCount > 0 && truckTemplates.Count > 0)
        {
            truckTrafficManager = runtimeRoot.gameObject.AddComponent<TruckTrafficManager>();
            activeTruckCount = truckTrafficManager.StartTraffic(
                truckTemplates,
                truckCount,
                missionCenter,
                new Vector2(FullAreaWidth, FullAreaDepth),
                MainSceneTruckSpeedRange);
        }

        CollectActiveTruckTargets();
        ReassignSearchZones();

        demoRunning = activeDroneUnits.Count > 0;
        nextDetectionScanTime = Time.time;

        if (demoRunning)
        {
            statusMessage = activeTruckCount > 0
                ? $"Running mission with {activeDroneUnits.Count} drones and {activeTruckCount} trucks."
                : $"Running mission with {activeDroneUnits.Count} drones. No trucks are active.";
        }
        else
        {
            statusMessage = "No drones were created for the demo.";
        }
    }

    private void StopDemo()
    {
        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit == null)
            {
                continue;
            }

            if (unit.SearchMission != null)
            {
                unit.SearchMission.StopMission();
            }

            if (unit.TrackingAgent != null)
            {
                unit.TrackingAgent.RuntimeTrackingEnded -= HandleRuntimeTrackingEnded;
                unit.TrackingAgent.StopRuntimeTracking();
            }

            if (unit.ClaimedTruck != null)
            {
                unit.ClaimedTruck.ReleaseClaim(unit.DroneId);
                unit.ClaimedTruck = null;
            }
        }

        activeDroneUnits.Clear();
        activeTruckTargets.Clear();
        missionZones.Clear();

        if (truckTrafficManager != null)
        {
            truckTrafficManager.StopTraffic();
            truckTrafficManager = null;
        }

        if (runtimeRoot != null)
        {
            Destroy(runtimeRoot.gameObject);
            runtimeRoot = null;
        }

        demoRunning = false;
    }

    private RuntimeDroneUnit CreateRuntimeDroneUnit(int droneIndex)
    {
        if (templateAgent == null || runtimeRoot == null)
        {
            return null;
        }

        Vector3 spawnPosition = GetSpawnPosition(droneIndex);
        string droneLabel = $"Demo Drone {droneIndex + 1:00}";

        GameObject searchObject = Instantiate(templateAgent.gameObject, runtimeRoot);
        searchObject.name = $"{droneLabel} Search";
        searchObject.transform.SetPositionAndRotation(spawnPosition, spawnAnchorRotation);
        ConfigureRuntimeDroneObject(searchObject);
        RemoveRecorder(searchObject);
        DisableCloneSensorsAndCameras(searchObject);
        Transform sharedSensorCameraTransform = FindPrimarySensorCameraTransform(searchObject);

        DroneCoverageAgent searchAgent = searchObject.GetComponent<DroneCoverageAgent>();
        LockedAltitudeDroneController searchController = searchObject.GetComponent<LockedAltitudeDroneController>();
        Rigidbody searchBody = searchObject.GetComponent<Rigidbody>();
        DecisionRequester searchDecisionRequester = searchObject.GetComponent<DecisionRequester>();
        BehaviorParameters searchBehaviorParameters = searchObject.GetComponent<BehaviorParameters>();

        if (searchController != null)
        {
            searchController.showDebugGui = false;
        }

        if (searchDecisionRequester != null)
        {
            searchDecisionRequester.enabled = true;
            searchDecisionRequester.DecisionPeriod = 5;
            searchDecisionRequester.DecisionStep = 0;
            searchDecisionRequester.TakeActionsBetweenDecisions = true;
        }

        if (searchBehaviorParameters != null)
        {
            searchBehaviorParameters.BehaviorName = "DroneCoverage";
            searchBehaviorParameters.Model = searchModel;
            searchBehaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
            searchBehaviorParameters.BrainParameters.VectorObservationSize = 57;
        }

        AreaCoverageTracker tracker = CreateTrackerForDrone(droneLabel, searchObject.transform);
        DroneDemoMissionController missionController =
            searchObject.GetComponent<DroneDemoMissionController>() ?? searchObject.AddComponent<DroneDemoMissionController>();
        DroneTrackingSensorFootprint searchFootprint =
            searchObject.GetComponent<DroneTrackingSensorFootprint>() ?? searchObject.AddComponent<DroneTrackingSensorFootprint>();
        searchFootprint.SetSensorTransform(searchObject.transform);

        missionController.Configure(searchAgent, searchController, tracker, null, activeTransitSpeed, droneLabel);

        searchObject.SetActive(true);

        if (searchAgent != null && searchModel != null)
        {
            searchAgent.SetModel("DroneCoverage", searchModel);
        }

        GameObject trackingObject = Instantiate(templateAgent.gameObject, runtimeRoot);
        trackingObject.name = $"{droneLabel} Tracking";
        trackingObject.transform.SetPositionAndRotation(spawnPosition, spawnAnchorRotation);
        ConfigureRuntimeDroneObject(trackingObject);
        RemoveRecorder(trackingObject);
        DisableCloneSensorsAndCameras(trackingObject);
        RemoveSensorCameraComponents(trackingObject);

        DroneCoverageAgent trackingCoverageAgent = trackingObject.GetComponent<DroneCoverageAgent>();
        if (trackingCoverageAgent != null)
        {
            Destroy(trackingCoverageAgent);
        }

        LockedAltitudeDroneController trackingController = trackingObject.GetComponent<LockedAltitudeDroneController>();
        Rigidbody trackingBody = trackingObject.GetComponent<Rigidbody>();
        DecisionRequester trackingDecisionRequester = trackingObject.GetComponent<DecisionRequester>();
        BehaviorParameters trackingBehaviorParameters = trackingObject.GetComponent<BehaviorParameters>();

        if (trackingController != null)
        {
            trackingController.showDebugGui = false;
            trackingController.ClearControlInputs();
        }

        if (trackingDecisionRequester != null)
        {
            trackingDecisionRequester.enabled = false;
        }

        if (trackingBehaviorParameters != null)
        {
            trackingBehaviorParameters.BehaviorName = "DroneTrack";
            trackingBehaviorParameters.Model = trackingModel;
            trackingBehaviorParameters.BehaviorType = BehaviorType.InferenceOnly;
            trackingBehaviorParameters.BrainParameters.VectorObservationSize = 19;
        }

        DroneTrackingSensorFootprint trackingFootprint =
            trackingObject.GetComponent<DroneTrackingSensorFootprint>() ?? trackingObject.AddComponent<DroneTrackingSensorFootprint>();
        trackingFootprint.SetSensorTransform(trackingObject.transform);
        DroneTrackingAgent trackingAgent =
            trackingObject.GetComponent<DroneTrackingAgent>() ?? trackingObject.AddComponent<DroneTrackingAgent>();
        trackingAgent.RuntimeTrackingEnded -= HandleRuntimeTrackingEnded;
        trackingAgent.RuntimeTrackingEnded += HandleRuntimeTrackingEnded;
        trackingAgent.RefreshSensorReferences();

        trackingObject.SetActive(false);

        return new RuntimeDroneUnit
        {
            DroneId = droneIndex + 1,
            Label = droneLabel,
            SearchObject = searchObject,
            SearchAgent = searchAgent,
            SearchController = searchController,
            SearchBody = searchBody,
            SearchTracker = tracker,
            SearchMission = missionController,
            SearchFootprint = searchFootprint,
            TrackingObject = trackingObject,
            TrackingAgent = trackingAgent,
            TrackingController = trackingController,
            TrackingBody = trackingBody,
            TrackingFootprint = trackingFootprint,
            SharedSensorCameraTransform = sharedSensorCameraTransform,
            BaseTrackingMaxSpeed = trackingController != null ? trackingController.maxSpeed : 14f,
            BaseTrackingAcceleration = trackingController != null ? trackingController.acceleration : 15f,
            BaseTrackingTurnSpeed = trackingController != null ? trackingController.turnSpeed : 85f
        };
    }

    private AreaCoverageTracker CreateTrackerForDrone(string droneName, Transform droneTransform)
    {
        GameObject trackerObject = new GameObject($"{droneName} Tracker");
        trackerObject.transform.SetParent(runtimeRoot, false);

        AreaCoverageTracker tracker = trackerObject.AddComponent<AreaCoverageTracker>();
        tracker.CopySettingsFrom(templateTracker);
        tracker.ConfigureDebugVisualization(
            showVisitedCellFill: false,
            showSensorFootprintOutline: true,
            footprintSegments: DemoFootprintSegments,
            showZoneBounds: false);

        GameObject zoneObject = new GameObject($"{droneName} Search Zone");
        zoneObject.transform.SetParent(trackerObject.transform, false);

        BoxCollider zoneCollider = zoneObject.AddComponent<BoxCollider>();
        zoneCollider.isTrigger = true;
        zoneCollider.size = new Vector3(ZoneWidth, 200f, ZoneDepth);

        tracker.SetSearchZone(zoneCollider);
        tracker.SetSensorTransform(droneTransform);
        return tracker;
    }

    private void ReassignSearchZones()
    {
        List<RuntimeDroneUnit> searchUnits = new List<RuntimeDroneUnit>();
        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit != null && !unit.IsTracking && unit.SearchObject != null)
            {
                searchUnits.Add(unit);
            }
        }

        List<List<DroneDemoZone>> assignments = BuildAssignments(missionZones, searchUnits);

        for (int i = 0; i < searchUnits.Count; i++)
        {
            RuntimeDroneUnit unit = searchUnits[i];
            if (unit.SearchObject == null || unit.SearchMission == null)
            {
                continue;
            }

            if (!unit.SearchObject.activeSelf)
            {
                unit.SearchObject.SetActive(true);
            }

            if (unit.SearchTracker != null && !unit.SearchTracker.gameObject.activeSelf)
            {
                unit.SearchTracker.gameObject.SetActive(true);
            }

            List<DroneDemoZone> nextAssignment = i < assignments.Count
                ? assignments[i]
                : new List<DroneDemoZone>();

            bool assignmentChanged = !AreAssignmentsEqual(unit.AssignedZones, nextAssignment);
            bool missionNeedsStart = !unit.SearchMission.IsMissionActive;
            if (!assignmentChanged && !missionNeedsStart)
            {
                continue;
            }

            unit.AssignedZones.Clear();
            unit.AssignedZones.AddRange(nextAssignment);
            unit.SearchMission.SetAssignedZones(unit.AssignedZones, activeTransitSpeed);
        }
    }

    private void ScanForTruckDetections()
    {
        if (!demoRunning || activeTruckTargets.Count == 0)
        {
            return;
        }

        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit == null ||
                unit.IsTracking ||
                unit.SearchObject == null ||
                !unit.SearchObject.activeInHierarchy ||
                unit.SearchMission == null ||
                !unit.SearchMission.IsMissionActive)
            {
                continue;
            }

            bool canClaimWhileTransiting = Time.time >= demoStartTime + TransitTruckClaimUnlockDelay;
            if (!unit.SearchMission.IsSearching && !canClaimWhileTransiting)
            {
                continue;
            }

            TruckTarget visibleTruck = FindBestVisibleUnclaimedTruck(unit);
            if (visibleTruck == null || !visibleTruck.TryClaim(unit.DroneId))
            {
                continue;
            }

            SwitchUnitToTracking(unit, visibleTruck);
        }
    }

    private void RecoverInvalidUnitStates()
    {
        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit == null)
            {
                continue;
            }

            bool trackingObjectActive = unit.TrackingObject != null && unit.TrackingObject.activeSelf;
            bool trackingStillValid = unit.ClaimedTruck != null &&
                                      unit.TrackingAgent != null &&
                                      unit.TrackingAgent.RuntimeTrackingActive &&
                                      trackingObjectActive;
            if ((trackingObjectActive || unit.ClaimedTruck != null) && !trackingStillValid)
            {
                ReturnUnitToSearch(unit);
                continue;
            }

            if (trackingStillValid)
            {
                continue;
            }

            if (unit.SearchObject != null && !unit.SearchObject.activeSelf)
            {
                unit.SearchObject.SetActive(true);
            }

            if (unit.SearchTracker != null && !unit.SearchTracker.gameObject.activeSelf)
            {
                unit.SearchTracker.gameObject.SetActive(true);
            }

            if (unit.SearchMission != null &&
                unit.AssignedZones.Count > 0 &&
                !unit.SearchMission.IsMissionActive)
            {
                unit.SearchMission.SetAssignedZones(unit.AssignedZones, activeTransitSpeed);
            }
        }
    }

    private TruckTarget FindBestVisibleUnclaimedTruck(RuntimeDroneUnit unit)
    {
        TruckTarget bestTruck = null;
        float bestScore = float.PositiveInfinity;

        for (int i = activeTruckTargets.Count - 1; i >= 0; i--)
        {
            TruckTarget truckTarget = activeTruckTargets[i];
            if (truckTarget == null)
            {
                activeTruckTargets.RemoveAt(i);
                continue;
            }

            if (!truckTarget.gameObject.activeInHierarchy || truckTarget.IsClaimed)
            {
                continue;
            }

            if (!TryScoreTrackingHandoffCandidate(unit, truckTarget, out float handoffScore))
            {
                continue;
            }

            if (handoffScore < bestScore)
            {
                bestScore = handoffScore;
                bestTruck = truckTarget;
            }
        }

        return bestTruck;
    }

    private bool TryScoreTrackingHandoffCandidate(RuntimeDroneUnit unit, TruckTarget truckTarget, out float score)
    {
        score = float.PositiveInfinity;

        if (unit == null || truckTarget == null || unit.SearchObject == null || unit.SearchTracker == null)
        {
            return false;
        }

        Vector3 trackingPoint = truckTarget.GetTrackingPoint();
        bool insideSearchFootprint = unit.SearchTracker.IsPointInsideSensorFootprint(trackingPoint);
        if (!insideSearchFootprint)
        {
            return false;
        }

        Vector3 flatToTarget = trackingPoint - unit.SearchObject.transform.position;
        flatToTarget.y = 0f;
        float handoffDistance = flatToTarget.magnitude;

        float preferredDistanceScore = Mathf.Abs(handoffDistance - TrackingHandoffPreferredDistance);
        float farDistancePenalty = handoffDistance > TrackingHandoffMaxDistance
            ? (handoffDistance - TrackingHandoffMaxDistance) * 0.5f
            : 0f;
        float nearDistancePenalty = handoffDistance < TrackingHandoffMinDistance
            ? (TrackingHandoffMinDistance - handoffDistance) * 0.15f
            : 0f;
        score = preferredDistanceScore + farDistancePenalty + nearDistancePenalty;
        return true;
    }

    private void SwitchUnitToTracking(RuntimeDroneUnit unit, TruckTarget truckTarget)
    {
        if (unit == null || truckTarget == null || unit.TrackingObject == null || unit.TrackingAgent == null)
        {
            return;
        }

        if (unit.SearchMission != null)
        {
            unit.SearchMission.StopMission();
        }

        SyncDroneState(unit.SearchObject, unit.SearchBody, unit.TrackingObject, unit.TrackingBody);
        AttachUnitSensorCamera(unit, unit.TrackingObject);
        ApplyTrackingFlightProfile(unit);
        Vector3 flatToTruck = truckTarget.GetTrackingPoint() - unit.TrackingObject.transform.position;
        flatToTruck.y = 0f;
        if (flatToTruck.sqrMagnitude > 0.0001f)
        {
            unit.TrackingObject.transform.rotation = Quaternion.LookRotation(flatToTruck.normalized, Vector3.up);
            if (unit.TrackingBody != null)
            {
                float currentVerticalVelocity = unit.TrackingBody.linearVelocity.y;
                float carriedHorizontalSpeed = new Vector3(
                    unit.TrackingBody.linearVelocity.x,
                    0f,
                    unit.TrackingBody.linearVelocity.z).magnitude;
                float maxTrackingSpeed = unit.TrackingController != null
                    ? Mathf.Max(0.1f, unit.TrackingController.maxSpeed)
                    : Mathf.Max(0.1f, unit.BaseTrackingMaxSpeed);
                float alignedSpeed = Mathf.Min(carriedHorizontalSpeed, maxTrackingSpeed);
                Vector3 alignedHorizontalVelocity = unit.TrackingObject.transform.forward * alignedSpeed;
                unit.TrackingBody.linearVelocity = new Vector3(
                    alignedHorizontalVelocity.x,
                    currentVerticalVelocity,
                    alignedHorizontalVelocity.z);
                unit.TrackingBody.angularVelocity = Vector3.zero;
            }
        }

        if (unit.SearchObject != null)
        {
            unit.SearchObject.SetActive(false);
        }

        if (unit.SearchTracker != null)
        {
            unit.SearchTracker.gameObject.SetActive(false);
        }

        unit.ClaimedTruck = truckTarget;
        unit.TrackingObject.SetActive(true);
        unit.TrackingAgent.SetModel("DroneTrack", trackingModel);
        unit.TrackingAgent.ConfigureRuntimeTracking(activeMissionCenter, new Vector2(FullAreaWidth, FullAreaDepth));
        unit.TrackingAgent.BeginRuntimeTracking(truckTarget);

        ReassignSearchZones();
    }

    private void ReturnUnitToSearch(RuntimeDroneUnit unit)
    {
        if (unit == null)
        {
            return;
        }

        TruckTarget previouslyClaimedTruck = unit.ClaimedTruck;
        if (previouslyClaimedTruck != null)
        {
            previouslyClaimedTruck.ReleaseClaim(unit.DroneId);
        }

        if (unit.TrackingAgent != null)
        {
            unit.TrackingAgent.StopRuntimeTracking();
        }

        SyncDroneState(unit.TrackingObject, unit.TrackingBody, unit.SearchObject, unit.SearchBody);
        AttachUnitSensorCamera(unit, unit.SearchObject);

        if (unit.TrackingObject != null)
        {
            unit.TrackingObject.SetActive(false);
        }

        unit.ClaimedTruck = null;

        if (unit.SearchObject != null)
        {
            unit.SearchObject.SetActive(true);
        }

        if (unit.SearchTracker != null)
        {
            unit.SearchTracker.gameObject.SetActive(true);
        }

        ReassignSearchZones();
    }

    private void HandleRuntimeTrackingEnded(DroneTrackingAgent sourceAgent, DroneTrackingAgent.RuntimeTrackingOutcome outcome)
    {
        RuntimeDroneUnit unit = FindUnitByTrackingAgent(sourceAgent);
        if (unit == null)
        {
            return;
        }

        ReturnUnitToSearch(unit);
    }

    private RuntimeDroneUnit FindUnitByTrackingAgent(DroneTrackingAgent trackingAgent)
    {
        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit != null && unit.TrackingAgent == trackingAgent)
            {
                return unit;
            }
        }

        return null;
    }

    private bool IsTruckVisibleToSearchDrone(RuntimeDroneUnit unit, TruckTarget truckTarget)
    {
        if (unit == null || truckTarget == null || unit.SearchObject == null)
        {
            return false;
        }

        Vector3 trackingPoint = truckTarget.GetTrackingPoint();
        Vector3 toTarget = trackingPoint - unit.SearchObject.transform.position;
        Vector3 flatToTarget = new Vector3(toTarget.x, 0f, toTarget.z);
        float maxTrackingDistance = unit.TrackingAgent != null
            ? unit.TrackingAgent.MaxTrackingDistance
            : 180f;

        if (flatToTarget.sqrMagnitude > maxTrackingDistance * maxTrackingDistance)
        {
            return false;
        }

        bool withinSearchFootprint = unit.SearchTracker != null && unit.SearchTracker.IsPointInsideSensorFootprint(trackingPoint);
        bool withinSearchViewport = unit.SearchFootprint != null &&
                                    unit.SearchFootprint.IsPointInView(trackingPoint, SearchDetectionViewportPadding);
        return withinSearchFootprint || withinSearchViewport;
    }

    private void ApplyTrackingFlightProfile(RuntimeDroneUnit unit)
    {
        if (unit == null || unit.TrackingController == null)
        {
            return;
        }

        // Keep runtime tracking on the same flight dynamics it was trained with.
        unit.TrackingController.maxSpeed = Mathf.Max(0.1f, unit.BaseTrackingMaxSpeed);
        unit.TrackingController.acceleration = Mathf.Max(0.1f, unit.BaseTrackingAcceleration);
        unit.TrackingController.turnSpeed = Mathf.Max(0.1f, unit.BaseTrackingTurnSpeed);
    }

    private static bool HasLineOfSight(DroneTrackingSensorFootprint footprint, TruckTarget truckTarget, Vector3 targetPoint)
    {
        if (footprint == null || truckTarget == null)
        {
            return false;
        }

        Vector3 origin = footprint.CoverageCamera != null
            ? footprint.CoverageCamera.transform.position
            : footprint.SensorWorldPosition;

        Vector3 toTarget = targetPoint - origin;
        float distance = toTarget.magnitude;
        if (distance <= 0.001f)
        {
            return true;
        }

        Vector3 direction = toTarget / distance;
        origin += direction * 0.5f;

        if (!Physics.Raycast(origin, direction, out RaycastHit hitInfo, distance + 0.1f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
        {
            return false;
        }

        return hitInfo.transform == truckTarget.transform || hitInfo.transform.IsChildOf(truckTarget.transform);
    }

    private void CollectActiveTruckTargets()
    {
        activeTruckTargets.Clear();
        if (truckTrafficManager == null)
        {
            return;
        }

        IReadOnlyList<GameObject> trucks = truckTrafficManager.ActiveTrucks;
        for (int i = 0; i < trucks.Count; i++)
        {
            GameObject truckObject = trucks[i];
            if (truckObject == null)
            {
                continue;
            }

            TruckTarget truckTarget = truckObject.GetComponent<TruckTarget>();
            if (truckTarget != null)
            {
                truckTarget.ResetTrackingState();
                activeTruckTargets.Add(truckTarget);
            }
        }
    }

    private void UpdateRuntimeStatusMessage()
    {
        if (!demoRunning)
        {
            return;
        }

        int trackingCount = CountTrackingUnits();
        int searchingCount = activeDroneUnits.Count - trackingCount;
        statusMessage = $"Running mission. Searching: {searchingCount}, Tracking: {trackingCount}, Claimed Trucks: {CountClaimedTrucks()} / {activeTruckTargets.Count}.";
    }

    private int CountTrackingUnits()
    {
        int count = 0;
        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            if (activeDroneUnits[i] != null && activeDroneUnits[i].IsTracking)
            {
                count++;
            }
        }

        return count;
    }

    private int CountClaimedTrucks()
    {
        int count = 0;
        for (int i = 0; i < activeTruckTargets.Count; i++)
        {
            if (activeTruckTargets[i] != null && activeTruckTargets[i].IsClaimed)
            {
                count++;
            }
        }

        return count;
    }

    private void DrawTrackingDebugPanel()
    {
        RuntimeDroneUnit debugUnit = FindDebugTrackingUnit();
        if (debugUnit == null || debugUnit.TrackingAgent == null)
        {
            GUILayout.Label("Tracker Debug: No active tracking drone");
            return;
        }

        DroneTrackingAgent trackingAgent = debugUnit.TrackingAgent;
        TruckTarget truckTarget = debugUnit.ClaimedTruck;

        GUILayout.Space(6f);
        GUILayout.Label("Tracker Debug");
        GUILayout.Label($"{debugUnit.Label}  Truck: {(truckTarget != null ? truckTarget.TruckId.ToString() : "None")}");
        GUILayout.Label(
            $"Visible: {trackingAgent.DebugLastVisible}  DirectSight: {trackingAgent.DebugHasDirectSight}  " +
            $"HasMemory: {trackingAgent.DebugHasTrackedTarget}  Collision: {trackingAgent.DebugCollisionDetected}");
        GUILayout.Label(
            $"Dist: {trackingAgent.DebugLastDistanceToTarget:F1}  Observed: {trackingAgent.DebugLastObservedDistance:F1}  " +
            $"LostFor: {trackingAgent.DebugTimeSinceLastVisible:F2}s");
        GUILayout.Label(
            $"Speed: {trackingAgent.DebugCurrentSpeed:F1} / {trackingAgent.DebugCurrentMaxSpeed:F1}  " +
            $"TurnSpeed: {trackingAgent.DebugCurrentTurnSpeed:F0}");
        if (debugUnit.TrackingController != null)
        {
            GUILayout.Label(
                $"Altitude: {debugUnit.TrackingController.CurrentAltitudeAboveGround:F1}  " +
                $"GroundY: {debugUnit.TrackingController.CurrentGroundHeight:F1}  " +
                $"Target: {debugUnit.TrackingController.targetAltitude:F1}");
        }
        GUILayout.Label(
            $"Cmd Forward: {trackingAgent.DebugLastForwardCommand:F2}  " +
            $"Turn: {trackingAgent.DebugLastTurnCommand:F2}  " +
            $"Strafe: {trackingAgent.DebugLastStrafeCommand:F2}");
    }

    private void DrawTrackingDebugOverlay()
    {
        RuntimeDroneUnit debugUnit = FindDebugTrackingUnit();
        if (debugUnit == null || debugUnit.TrackingAgent == null)
        {
            return;
        }

        DroneTrackingAgent trackingAgent = debugUnit.TrackingAgent;
        TruckTarget truckTarget = debugUnit.ClaimedTruck;

        float overlayWidth = 360f;
        float overlayHeight = 174f;
        float overlayX = Mathf.Max(12f, Screen.width - overlayWidth - 12f);
        float overlayY = 12f;

        GUILayout.BeginArea(new Rect(overlayX, overlayY, overlayWidth, overlayHeight), GUI.skin.box);
        GUILayout.Label("Tracker Debug");
        GUILayout.Label($"{debugUnit.Label}  Truck: {(truckTarget != null ? truckTarget.TruckId.ToString() : "None")}");
        GUILayout.Label(
            $"Visible: {trackingAgent.DebugLastVisible}  DirectSight: {trackingAgent.DebugHasDirectSight}  " +
            $"Memory: {trackingAgent.DebugHasTrackedTarget}  Collision: {trackingAgent.DebugCollisionDetected}");
        GUILayout.Label(
            $"Dist: {trackingAgent.DebugLastDistanceToTarget:F1}  Observed: {trackingAgent.DebugLastObservedDistance:F1}  " +
            $"LostFor: {trackingAgent.DebugTimeSinceLastVisible:F2}s");
        GUILayout.Label(
            $"Speed: {trackingAgent.DebugCurrentSpeed:F1}/{trackingAgent.DebugCurrentMaxSpeed:F1}  " +
            $"TurnSpeed: {trackingAgent.DebugCurrentTurnSpeed:F0}");
        if (debugUnit.TrackingController != null)
        {
            GUILayout.Label(
                $"Altitude: {debugUnit.TrackingController.CurrentAltitudeAboveGround:F1}  " +
                $"GroundY: {debugUnit.TrackingController.CurrentGroundHeight:F1}  " +
                $"Target: {debugUnit.TrackingController.targetAltitude:F1}");
        }
        GUILayout.Label(
            $"Cmd Fwd: {trackingAgent.DebugLastForwardCommand:F2}  " +
            $"Turn: {trackingAgent.DebugLastTurnCommand:F2}  " +
            $"Strafe: {trackingAgent.DebugLastStrafeCommand:F2}");
        GUILayout.EndArea();
    }

    private RuntimeDroneUnit FindDebugTrackingUnit()
    {
        RuntimeDroneUnit bestUnit = null;
        float bestScore = float.NegativeInfinity;

        for (int i = 0; i < activeDroneUnits.Count; i++)
        {
            RuntimeDroneUnit unit = activeDroneUnits[i];
            if (unit != null && unit.IsTracking && unit.TrackingAgent != null)
            {
                DroneTrackingAgent trackingAgent = unit.TrackingAgent;
                float speedRatio = trackingAgent.DebugCurrentMaxSpeed > 0.01f
                    ? trackingAgent.DebugCurrentSpeed / trackingAgent.DebugCurrentMaxSpeed
                    : 0f;

                float score = 0f;
                if (!trackingAgent.DebugLastVisible)
                {
                    score += 1000f;
                }

                if (!trackingAgent.DebugHasDirectSight)
                {
                    score += 100f;
                }

                if (trackingAgent.DebugCollisionDetected)
                {
                    score += 500f;
                }

                score += Mathf.Clamp(trackingAgent.DebugTimeSinceLastVisible, 0f, 10f) * 100f;
                score += (1f - Mathf.Clamp01(speedRatio)) * 25f;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestUnit = unit;
                }
            }
        }

        return bestUnit;
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

    private void CreateMissionZoneVisualizers(IEnumerable<DroneDemoZone> zones)
    {
        if (runtimeRoot == null || zones == null)
        {
            return;
        }

        foreach (DroneDemoZone zone in zones)
        {
            GameObject zoneObject = new GameObject($"Mission Zone {zone.ZoneId + 1:00}");
            zoneObject.transform.SetParent(runtimeRoot, false);

            DroneDemoZoneVisualizer visualizer = zoneObject.AddComponent<DroneDemoZoneVisualizer>();
            visualizer.Configure(zone.CenterXZ, zone.SizeXZ, MissionZoneColor);
        }
    }

    private static List<List<DroneDemoZone>> BuildAssignments(List<DroneDemoZone> zones, List<RuntimeDroneUnit> searchUnits)
    {
        List<List<DroneDemoZone>> assignments = new List<List<DroneDemoZone>>(searchUnits.Count);
        if (searchUnits.Count == 0)
        {
            return assignments;
        }

        int[] targetCounts = new int[searchUnits.Count];
        int[] retainedCounts = new int[searchUnits.Count];
        int baseZoneCount = zones.Count / searchUnits.Count;
        int remainder = zones.Count % searchUnits.Count;
        Dictionary<int, DroneDemoZone> canonicalZones = new Dictionary<int, DroneDemoZone>(zones.Count);
        HashSet<int> unassignedZoneIds = new HashSet<int>();

        for (int zoneIndex = 0; zoneIndex < zones.Count; zoneIndex++)
        {
            DroneDemoZone zone = zones[zoneIndex];
            canonicalZones[zone.ZoneId] = zone;
            unassignedZoneIds.Add(zone.ZoneId);
        }

        for (int unitIndex = 0; unitIndex < searchUnits.Count; unitIndex++)
        {
            assignments.Add(new List<DroneDemoZone>());
            targetCounts[unitIndex] = baseZoneCount + (unitIndex < remainder ? 1 : 0);
        }

        for (int unitIndex = 0; unitIndex < searchUnits.Count; unitIndex++)
        {
            RuntimeDroneUnit unit = searchUnits[unitIndex];
            if (unit == null || unit.AssignedZones == null)
            {
                continue;
            }

            for (int assignedIndex = 0; assignedIndex < unit.AssignedZones.Count; assignedIndex++)
            {
                DroneDemoZone assignedZone = unit.AssignedZones[assignedIndex];
                if (assignments[unitIndex].Count >= targetCounts[unitIndex] ||
                    !canonicalZones.TryGetValue(assignedZone.ZoneId, out DroneDemoZone canonicalZone) ||
                    !unassignedZoneIds.Remove(canonicalZone.ZoneId))
                {
                    continue;
                }

                assignments[unitIndex].Add(canonicalZone);
            }

            retainedCounts[unitIndex] = assignments[unitIndex].Count;
        }

        List<DroneDemoZone> remainingZones = new List<DroneDemoZone>(unassignedZoneIds.Count);
        foreach (DroneDemoZone zone in zones)
        {
            if (unassignedZoneIds.Contains(zone.ZoneId))
            {
                remainingZones.Add(zone);
            }
        }

        remainingZones.Sort((left, right) =>
            EstimateZonePriority(right, searchUnits).CompareTo(EstimateZonePriority(left, searchUnits)));

        for (int zoneIndex = 0; zoneIndex < remainingZones.Count; zoneIndex++)
        {
            DroneDemoZone zone = remainingZones[zoneIndex];
            int bestUnitIndex = FindBestUnitForZone(zone, searchUnits, assignments, targetCounts);
            if (bestUnitIndex < 0)
            {
                bestUnitIndex = FindLeastLoadedUnit(assignments);
            }

            if (bestUnitIndex < 0)
            {
                continue;
            }

            assignments[bestUnitIndex].Add(zone);
        }

        for (int i = 0; i < assignments.Count; i++)
        {
            RuntimeDroneUnit unit = searchUnits[i];
            int retainedCount = retainedCounts[i];
            if (assignments[i].Count <= retainedCount)
            {
                continue;
            }

            if (retainedCount == 0)
            {
                assignments[i].Sort((left, right) =>
                    GetZoneDistanceSq(unit, left).CompareTo(GetZoneDistanceSq(unit, right)));
            }
            else
            {
                List<DroneDemoZone> additionalZones = assignments[i].GetRange(retainedCount, assignments[i].Count - retainedCount);
                additionalZones.Sort((left, right) =>
                    GetZoneDistanceSq(unit, left).CompareTo(GetZoneDistanceSq(unit, right)));
                assignments[i].RemoveRange(retainedCount, assignments[i].Count - retainedCount);
                assignments[i].AddRange(additionalZones);
            }
        }

        return assignments;
    }

    private static float EstimateZonePriority(DroneDemoZone zone, List<RuntimeDroneUnit> searchUnits)
    {
        float bestDistanceSq = float.PositiveInfinity;
        for (int i = 0; i < searchUnits.Count; i++)
        {
            bestDistanceSq = Mathf.Min(bestDistanceSq, GetZoneDistanceSq(searchUnits[i], zone));
        }

        return bestDistanceSq;
    }

    private static int FindBestUnitForZone(
        DroneDemoZone zone,
        List<RuntimeDroneUnit> searchUnits,
        List<List<DroneDemoZone>> currentAssignments,
        int[] targetCounts)
    {
        int bestUnitIndex = -1;
        float bestDistanceSq = float.PositiveInfinity;

        for (int i = 0; i < searchUnits.Count; i++)
        {
            if (currentAssignments[i].Count >= targetCounts[i])
            {
                continue;
            }

            float distanceSq = GetZoneDistanceSq(searchUnits[i], zone);
            if (distanceSq < bestDistanceSq)
            {
                bestDistanceSq = distanceSq;
                bestUnitIndex = i;
            }
        }

        return bestUnitIndex;
    }

    private static int FindLeastLoadedUnit(List<List<DroneDemoZone>> assignments)
    {
        int bestUnitIndex = -1;
        int fewestZones = int.MaxValue;

        for (int i = 0; i < assignments.Count; i++)
        {
            int zoneCount = assignments[i].Count;
            if (zoneCount < fewestZones)
            {
                fewestZones = zoneCount;
                bestUnitIndex = i;
            }
        }

        return bestUnitIndex;
    }

    private static bool AreAssignmentsEqual(List<DroneDemoZone> left, List<DroneDemoZone> right)
    {
        if (ReferenceEquals(left, right))
        {
            return true;
        }

        if (left == null || right == null || left.Count != right.Count)
        {
            return false;
        }

        for (int i = 0; i < left.Count; i++)
        {
            if (left[i].ZoneId != right[i].ZoneId)
            {
                return false;
            }
        }

        return true;
    }

    private static float GetZoneDistanceSq(RuntimeDroneUnit unit, DroneDemoZone zone)
    {
        Vector3 position = unit.CurrentPosition;
        float dx = position.x - zone.CenterXZ.x;
        float dz = position.z - zone.CenterXZ.y;
        return (dx * dx) + (dz * dz);
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

    private bool TryParseInputs(out int droneCount, out int truckCount, out float transitSpeed, out Vector2 missionCenter)
    {
        droneCount = 0;
        truckCount = 0;
        transitSpeed = 0f;
        missionCenter = Vector2.zero;

        if (!int.TryParse(droneCountInput, out droneCount))
        {
            statusMessage = "Drone count must be a whole number between 1 and 20.";
            return false;
        }

        droneCount = Mathf.Clamp(droneCount, 1, TotalZones);
        droneCountInput = droneCount.ToString();
        if (matchTruckCountToDrones)
        {
            truckCount = droneCount;
            truckCountInput = droneCountInput;
        }
        else
        {
            if (!int.TryParse(truckCountInput, out truckCount))
            {
                statusMessage = "Truck count must be a whole number between 0 and 20.";
                return false;
            }

            truckCount = Mathf.Clamp(truckCount, 0, TotalZones);
            truckCountInput = truckCount.ToString();
        }

        if (!float.TryParse(transitSpeedInput, out transitSpeed))
        {
            statusMessage = "Transit speed must be a number greater than 0.";
            return false;
        }

        transitSpeed = Mathf.Max(MinimumTransitSpeed, transitSpeed);
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

        for (int i = 0; i < truckTemplates.Count; i++)
        {
            if (truckTemplates[i] != null && truckTemplates[i].scene.IsValid())
            {
                truckTemplates[i].SetActive(active);
            }
        }
    }

    private void DiscoverTruckTemplates()
    {
        truckTemplates.Clear();
        TryAddTruckTemplate(CoverTruckTemplateName, CoverTruckPrefabPath);
    }

    private static void ConfigureRuntimeDronePhysics()
    {
        int runtimeDroneLayer = LayerMask.NameToLayer(RuntimeDroneLayerName);
        if (runtimeDroneLayer >= 0)
        {
            Physics.IgnoreLayerCollision(runtimeDroneLayer, runtimeDroneLayer, true);
        }
    }

    private static void ConfigureRuntimeDroneObject(GameObject droneObject)
    {
        if (droneObject == null)
        {
            return;
        }

        int runtimeDroneLayer = LayerMask.NameToLayer(RuntimeDroneLayerName);
        if (runtimeDroneLayer >= 0)
        {
            SetLayerRecursively(droneObject.transform, runtimeDroneLayer);
        }

        LockedAltitudeDroneController controller = droneObject.GetComponent<LockedAltitudeDroneController>();
        if (controller != null)
        {
            int groundMask = controller.groundLayer.value;
            if (groundMask == 0)
            {
                groundMask = Physics.DefaultRaycastLayers;
            }

            if (runtimeDroneLayer >= 0)
            {
                groundMask &= ~(1 << runtimeDroneLayer);
            }

            controller.groundLayer = groundMask;
        }
    }

    private void TryAddTruckTemplate(string truckObjectName, string prefabPath)
    {
        GameObject truckObject = GameObject.Find(truckObjectName);
#if UNITY_EDITOR
        if (truckObject == null)
        {
            truckObject = AssetDatabase.LoadAssetAtPath<GameObject>(prefabPath);
        }
#endif
        if (truckObject != null)
        {
            truckTemplates.Add(truckObject);
        }
    }

    private static void SyncDroneState(GameObject sourceObject, Rigidbody sourceBody, GameObject targetObject, Rigidbody targetBody)
    {
        if (sourceObject == null || targetObject == null)
        {
            return;
        }

        targetObject.transform.SetPositionAndRotation(sourceObject.transform.position, sourceObject.transform.rotation);

        if (targetBody != null)
        {
            targetBody.linearVelocity = sourceBody != null ? sourceBody.linearVelocity : Vector3.zero;
            targetBody.angularVelocity = sourceBody != null ? sourceBody.angularVelocity : Vector3.zero;
        }
    }

    private static void SetLayerRecursively(Transform root, int layer)
    {
        if (root == null)
        {
            return;
        }

        root.gameObject.layer = layer;
        for (int i = 0; i < root.childCount; i++)
        {
            SetLayerRecursively(root.GetChild(i), layer);
        }
    }

    private static void DisableCloneSensorsAndCameras(GameObject droneObject)
    {
        foreach (Camera cameraComponent in droneObject.GetComponentsInChildren<Camera>(true))
        {
            cameraComponent.enabled = false;
            cameraComponent.cameraType = CameraType.Preview;
#if UNITY_EDITOR
            cameraComponent.hideFlags |= HideFlags.HideInHierarchy;
            cameraComponent.gameObject.hideFlags |= HideFlags.HideInHierarchy;
#endif
        }

        foreach (AudioListener audioListener in droneObject.GetComponentsInChildren<AudioListener>(true))
        {
            audioListener.enabled = false;
#if UNITY_EDITOR
            audioListener.hideFlags |= HideFlags.HideInHierarchy;
            audioListener.gameObject.hideFlags |= HideFlags.HideInHierarchy;
#endif
        }
    }

    private static Transform FindPrimarySensorCameraTransform(GameObject droneObject)
    {
        if (droneObject == null)
        {
            return null;
        }

        Camera cameraComponent = droneObject.GetComponentInChildren<Camera>(true);
        return cameraComponent != null ? cameraComponent.transform : null;
    }

    private static void RemoveSensorCameraComponents(GameObject droneObject)
    {
        if (droneObject == null)
        {
            return;
        }

        HashSet<GameObject> removableObjects = new HashSet<GameObject>();

        foreach (Camera cameraComponent in droneObject.GetComponentsInChildren<Camera>(true))
        {
            if (cameraComponent != null && cameraComponent.gameObject != droneObject)
            {
                removableObjects.Add(cameraComponent.gameObject);
            }
        }

        foreach (AudioListener audioListener in droneObject.GetComponentsInChildren<AudioListener>(true))
        {
            if (audioListener != null && audioListener.gameObject != droneObject)
            {
                removableObjects.Add(audioListener.gameObject);
            }
        }

        foreach (GameObject removableObject in removableObjects)
        {
            Object.Destroy(removableObject);
        }
    }

    private void AttachUnitSensorCamera(RuntimeDroneUnit unit, GameObject targetObject)
    {
        if (unit == null || unit.SharedSensorCameraTransform == null || targetObject == null)
        {
            return;
        }

        Vector3 localPosition = unit.SharedSensorCameraTransform.localPosition;
        Quaternion localRotation = unit.SharedSensorCameraTransform.localRotation;
        Vector3 localScale = unit.SharedSensorCameraTransform.localScale;

        unit.SharedSensorCameraTransform.SetParent(targetObject.transform, false);
        unit.SharedSensorCameraTransform.localPosition = localPosition;
        unit.SharedSensorCameraTransform.localRotation = localRotation;
        unit.SharedSensorCameraTransform.localScale = localScale;

        if (unit.SearchTracker != null && unit.SearchObject != null)
        {
            unit.SearchTracker.SetSensorTransform(unit.SearchObject.transform);
        }

        if (unit.SearchFootprint != null && unit.SearchObject != null)
        {
            unit.SearchFootprint.SetSensorTransform(unit.SearchObject.transform);
        }

        if (unit.TrackingFootprint != null && unit.TrackingObject != null)
        {
            unit.TrackingFootprint.SetSensorTransform(unit.TrackingObject.transform);
        }

        if (unit.TrackingAgent != null)
        {
            unit.TrackingAgent.RefreshSensorReferences();
        }
    }

    private static void RemoveRecorder(GameObject droneObject)
    {
        DemonstrationRecorder recorder = droneObject.GetComponent<DemonstrationRecorder>();
        if (recorder != null)
        {
            recorder.Record = false;
            recorder.enabled = false;
            Object.Destroy(recorder);
        }
    }
}
