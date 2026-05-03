"use client";

import { useEffect, useRef, useState } from "react";

type Props = { instanceId?: number | string };

export default function DroneVideoPlayer({ instanceId }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const [status, setStatus] = useState<string>("disconnected");
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (instanceId == null) {
      setStatus("no-stream-id");
      setLastError("No instanceId provided");
      return;
    }

    let stopped = false;
    const hosts = ["localhost", "127.0.0.1", "::1"];

    const scheduleReconnect = (ms = 2000) => {
      if (reconnectRef.current) window.clearTimeout(reconnectRef.current);
      reconnectRef.current = window.setTimeout(() => {
        if (!stopped) connect();
      }, ms) as unknown as number;
    };

    const cleanupAndRetry = () => {
      try {
        wsRef.current?.close();
      } catch {}
      wsRef.current = null;
      scheduleReconnect(2000);
    };

    const connect = async () => {
      setStatus("connecting");
      setLastError(null);

      for (const host of hosts) {
        // bracket IPv6 hosts
        const hostPart = host.includes(":") && !host.startsWith("[")
          ? `[${host}]`
          : host;
        const url = `ws://${hostPart}:8080/ws/video-client/${encodeURIComponent(
          String(instanceId)
        )}`;

        try {
          const ws = new WebSocket(url);
          ws.binaryType = "blob";

          const openPromise = new Promise<void>((resolve, reject) => {
            const onOpen = () => {
              ws.removeEventListener("error", onError);
              ws.removeEventListener("close", onClose);
              resolve();
            };
            const onError = (ev: Event) => {
              ws.removeEventListener("open", onOpen);
              ws.removeEventListener("close", onClose);
              reject(ev);
            };
            const onClose = (ev: CloseEvent) => {
              ws.removeEventListener("open", onOpen);
              ws.removeEventListener("error", onError);
              reject(ev);
            };
            ws.addEventListener("open", onOpen);
            ws.addEventListener("error", onError);
            ws.addEventListener("close", onClose);
          });

          ws.onmessage = async (ev: MessageEvent) => {
            try {
              const blob = ev.data as Blob;
              const bitmap = await createImageBitmap(blob);
              const canvas = canvasRef.current;
              if (canvas) {
                canvas.width = bitmap.width;
                canvas.height = bitmap.height;
                const ctx = canvas.getContext("2d");
                if (ctx) ctx.drawImage(bitmap, 0, 0);
              }
              bitmap.close();
            } catch (err) {
              console.error("video draw error", err);
            }
          };

          await openPromise;

          ws.onclose = (ev) => {
            setStatus("closed");
            const reason = (ev && (ev as CloseEvent).reason) ?? "";
            setLastError(`close code=${(ev as CloseEvent).code} reason=${reason}`);
            console.warn("Video WS closed", url, ev.code, ev.reason);
            cleanupAndRetry();
          };

          ws.onerror = (ev) => {
            setStatus("error");
            setLastError("WebSocket error (see console)");
            console.error("Video WS error", url, ev);
          };

          wsRef.current = ws;
          setStatus("connected");
          console.debug("Video WS connected ->", url);
          return;
        } catch (err: any) {
          console.warn("connect attempt failed", host, err);
          setLastError(String(err));
          continue;
        }
      }

      setStatus("disconnected");
      scheduleReconnect(2000);
    };

    connect();

    return () => {
      stopped = true;
      try {
        wsRef.current?.close();
      } catch {}
      if (reconnectRef.current !== null) {
        window.clearTimeout(reconnectRef.current);
        reconnectRef.current = null;
      }
      wsRef.current = null;
    };
  }, [instanceId]);

  return (
    <div style={{ width: "100%", background: "#000" }}>
      <div
        style={{
          fontSize: 12,
          color: status === "connected" ? "#6ee7b7" : "#f87171",
        }}
      >
        {status === "connected" ? "Camera live" : "Camera disconnected"}
      </div>

      {lastError && (
        <div
          style={{
            color: "#ff7b7b",
            fontSize: 11,
            marginBottom: 6,
            whiteSpace: "pre-wrap",
          }}
        >
          {lastError}
        </div>
      )}

      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "auto", display: "block", background: "#000" }}
      />
    </div>
  );
}
