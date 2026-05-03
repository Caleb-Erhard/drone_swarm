using System;
using System.Text;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using System.Collections.Generic;

public class TelemetryStreamer : MonoBehaviour
{
    [Header("Telemetry")]
    public string serverUrl = "ws://localhost:8080/ws/unity";
    [Tooltip("Telemetry updates per second")]
    public float updatesPerSecond = 10f;
    public bool includeInactive = false;

    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private int lastReportedCount = -1;

    [Serializable]
    class DronePayload
    {
        public string id;
        public string label;
        public int instanceId;
        public float x;
        public float y;
        public float altitude_m;
        public float battery;
        public string state;
        public int assigned_zones;
        public int current_zone;
    }

    async void Start()
    {
        cts = new CancellationTokenSource();
        ws = new ClientWebSocket();
        try
        {
            await ws.ConnectAsync(new Uri(serverUrl), cts.Token);
            Debug.Log("Telemetry websocket connected: " + serverUrl);
            _ = StreamLoopAsync(cts.Token);
        }
        catch (Exception e)
        {
            Debug.LogError("WebSocket connect failed: " + e.Message);
        }
    }

    async Task StreamLoopAsync(CancellationToken token)
    {
        var delayMs = (int)(1000f / Mathf.Max(1f, updatesPerSecond));

        while (!token.IsCancellationRequested && ws != null && ws.State == WebSocketState.Open)
        {
            var found = FindObjectsOfType<LockedAltitudeDroneController>(false); 
            var parts = new List<string>();

            foreach (var ctrl in found)
            {
                var go = ctrl.gameObject;
                var identity = go.GetComponent<DroneIdentity>();
                var mission = go.GetComponent<DroneDemoMissionController>();
                var agent = go.GetComponent<DroneCoverageAgent>();
                var trackAgent = go.GetComponent<DroneTrackingAgent>();

                // Determine logical identity
                string label = identity != null ? identity.droneLabel : go.name;
                int id = identity != null ? identity.originalInstanceId : go.GetInstanceID();

                // Determine state
                string state = "idle";
                if (identity != null && identity.isTrackingTwin) {
                    state = "tracking";
                } else if (agent != null && agent.RuntimeSearchActive) {
                    state = "searching";
                } else if (mission != null && mission.IsMissionActive) {
                    state = "transit";
                }

                var payload = new DronePayload
                {
                    id = label, // Use label as the ID anchor
                    label = label,
                    instanceId = id, // This is the ID the video player uses
                    x = go.transform.position.x,
                    y = go.transform.position.z,
                    altitude_m = go.transform.position.y,
                    battery = 100f,
                    state = state,
                    assigned_zones = mission != null ? mission.AssignedZoneCount : 0,
                    current_zone = mission != null ? mission.CurrentZoneNumber : 0
                };
                parts.Add(JsonUtility.ToJson(payload));
            }

            if (parts.Count != lastReportedCount)
            {
                lastReportedCount = parts.Count;
                Debug.Log($"TelemetryStreamer: reporting {parts.Count} drone(s)");
            }

            string payloadJson = "[" + string.Join(",", parts) + "]";
            var bytes = Encoding.UTF8.GetBytes(payloadJson);

            try
            {
                await ws.SendAsync(new ArraySegment<byte>(bytes),
                    WebSocketMessageType.Text, true, token);
            }
            catch (Exception e)
            {
                Debug.LogError("WebSocket send failed: " + e.Message);
                break;
            }

            await Task.Delay(delayMs, token);
        }
    }

    void OnApplicationQuit()
    {
        if (cts != null) cts.Cancel();
        if (ws != null && ws.State == WebSocketState.Open)
        {
            try
            {
                ws.CloseAsync(WebSocketCloseStatus.NormalClosure,
                    "quit", CancellationToken.None).Wait();
            }
            catch { }
        }
    }
}