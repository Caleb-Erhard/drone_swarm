using System;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using UnityEngine;
using System.Collections;

/// <summary>
/// Streams camera frames to a Go backend via WebSockets.
/// Designed for seamless handover between Search and Tracking objects.
/// </summary>
[RequireComponent(typeof(Camera))]
public class DroneVideoStreamer : MonoBehaviour
{
    [Header("Capture Settings")]
    public Camera sourceCamera;
    public int width = 640;
    public int height = 360;
    [Range(10, 95)] public int jpegQuality = 50;
    public float fps = 10f;

    [Header("Server Settings")]
    public string serverBaseUrl = "ws://localhost:8080/ws/video/";
    
    [Tooltip("Forced ID by DroneDemoManager to keep Search and Tracking streams identical.")]
    public string forceStreamId = "";

    // internals
    private string streamId;
    private ClientWebSocket ws;
    private CancellationTokenSource cts;
    private RenderTexture rt;
    private Texture2D readTex;

    private readonly ConcurrentQueue<byte[]> frameQueue = new ConcurrentQueue<byte[]>();
    private readonly SemaphoreSlim queueSignal = new SemaphoreSlim(0, 100);
    private Task senderTask;
    private bool isCapturing = false;

    async void Start()
    {
        if (sourceCamera == null) sourceCamera = GetComponent<Camera>();
        
        // Ensure camera stays enabled for rendering to the RenderTexture
        sourceCamera.enabled = true;
        sourceCamera.gameObject.SetActive(true);

        // Set up render textures
        rt = new RenderTexture(width, height, 16, RenderTextureFormat.ARGB32);
        rt.Create();
        sourceCamera.targetTexture = rt;
        readTex = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Resolve Stream ID: Use forced ID if provided, otherwise parent ID
        if (!string.IsNullOrEmpty(forceStreamId))
        {
            streamId = forceStreamId;
        }
        else
        {
            var parent = GetComponentInParent<LockedAltitudeDroneController>();
            streamId = parent != null ? parent.gameObject.GetInstanceID().ToString() : gameObject.GetInstanceID().ToString();
        }

        cts = new CancellationTokenSource();
        
        // Start background sender
        senderTask = Task.Run(SenderLoop, cts.Token);
        
        // Start connection manager
        _ = ConnectWithRetry(cts.Token);

        // Start capture loop
        isCapturing = true;
        StartCoroutine(CaptureLoop());
    }

    private void OnEnable()
    {
        if (cts != null && !isCapturing)
        {
            isCapturing = true;
            StartCoroutine(CaptureLoop());
        }
    }

    private void OnDisable()
    {
        // Stop the capture loop to save CPU when inactive, 
        // but DO NOT close the WebSocket here. This allows for handover.
        isCapturing = false;
        StopAllCoroutines();
    }

    private async Task ConnectWithRetry(CancellationToken token)
    {
        var uri = new Uri(serverBaseUrl + Uri.EscapeDataString(streamId));
        
        while (!token.IsCancellationRequested)
        {
            try
            {
                ws = new ClientWebSocket();
                // Optimization: Keep buffers small for low-latency streaming
                ws.Options.SetBuffer(1024 * 32, 1024 * 32); 
                
                await ws.ConnectAsync(uri, token);
                Debug.Log($"[Video Streamer] Connected to Hub: {streamId}");
                
                // Wait here until the socket closes/errors
                while (ws.State == WebSocketState.Open && !token.IsCancellationRequested)
                {
                    await Task.Delay(1000, token);
                }
            }
            catch (Exception)
            {
                // Silent retry
            }
            finally
            {
                ws?.Dispose();
                ws = null;
            }

            // Quick retry interval for handover (500ms)
            if (!token.IsCancellationRequested) await Task.Delay(500, token);
        }
    }

    private IEnumerator CaptureLoop()
    {
        var wait = new WaitForSeconds(1f / Mathf.Max(1f, fps));

        while (isCapturing && !cts.IsCancellationRequested)
        {
            // Force the camera to render a frame to the targetTexture
            sourceCamera.Render();

            // Copy to Texture2D
            RenderTexture prev = RenderTexture.active;
            RenderTexture.active = rt;
            readTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            readTex.Apply();
            RenderTexture.active = prev;

            // Fast JPEG Encoding on main thread
            byte[] jpg = readTex.EncodeToJPG(jpegQuality);

            // Queue for background sending
            if (frameQueue.Count < 3) // Cap queue to 3 frames to prevent lag accumulation
            {
                frameQueue.Enqueue(jpg);
                queueSignal.Release();
            }

            yield return wait;
        }
    }

    private async Task SenderLoop()
    {
        try
        {
            while (!cts.Token.IsCancellationRequested)
            {
                await queueSignal.WaitAsync(cts.Token);

                if (frameQueue.TryDequeue(out byte[] frameData))
                {
                    if (ws != null && ws.State == WebSocketState.Open)
                    {
                        await ws.SendAsync(
                            new ArraySegment<byte>(frameData),
                            WebSocketMessageType.Binary,
                            true,
                            cts.Token
                        );
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Video Streamer] Sender error: {ex.Message}");
        }
    }

    private void OnDestroy()
    {
        Cleanup();
    }

    private void Cleanup()
    {
        isCapturing = false;
        if (cts != null) cts.Cancel();

        if (ws != null)
        {
            ws.Dispose();
            ws = null;
        }

        if (rt != null)
        {
            rt.Release();
            Destroy(rt);
        }

        if (readTex != null)
        {
            Destroy(readTex);
        }
    }
}