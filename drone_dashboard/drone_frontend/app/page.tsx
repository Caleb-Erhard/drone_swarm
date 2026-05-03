"use client";

import { useEffect, useRef, useState } from "react";
import DroneVideoPlayer from "../components/DroneVideoPlayer.client";

interface DroneData {
  key: string;
  id: string;
  label: string;
  instanceId?: number;
  position: { x: number; y: number; z: number };
  battery: number;
  altitude_m: number;
  state: string;
  assignedZones?: number;
  currentZone?: number;
}

type FilterState = "all" | "searching" | "tracking" | "transit";

export default function Page() {
  const [drones, setDrones] = useState<Record<string, DroneData>>({});
  const [connected, setConnected] = useState(false);
  const [filter, setFilter] = useState<FilterState>("all");
  
  // Track which videos are EXPLICITLY closed. 
  // If a key is not in here (undefined), it is considered "Open".
  const [openVideos, setOpenVideos] = useState<Record<string, boolean>>({});
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8080/ws/frontend");
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (e) => {
      try {
        const raw = JSON.parse(e.data as string);
        const arr = Array.isArray(raw) ? raw : [raw];

        setDrones((prev) => {
          const next = { ...prev };
          for (const r of arr as any[]) {
            // Use label as the unique key so handover doesn't create new cards
            const key = r.label || r.id || "drone";

            next[key] = {
              key,
              id: r.id,
              label: r.label,
              instanceId: r.instanceId,
              position: { 
                x: (r.x ?? 0), 
                y: (r.altitude_m ?? 0), 
                z: (r.y ?? 0) 
              },
              battery: (r.battery ?? 0),
              altitude_m: (r.altitude_m ?? 0),
              state: r.state ?? "unknown",
              assignedZones: r.assigned_zones ?? 0,
              currentZone: r.current_zone ?? 0,
            };
          }
          return next;
        });
      } catch (err) {
        console.error("telemetry parse error", err);
      }
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Filter drones based on button selection
  const filteredDrones = Object.values(drones).filter((d) => {
    if (filter === "all") return true;
    return d.state.toLowerCase() === filter;
  });

  // Toggle Logic: Default is open (undefined), so toggle flips it to false.
  function toggleVideo(key: string) {
    setOpenVideos((prev) => {
      const isCurrentlyOpen = prev[key] !== false;
      return {
        ...prev,
        [key]: !isCurrentlyOpen,
      };
    });
  }

  // Helper for cleaner JSX
  const isVideoOpen = (key: string) => openVideos[key] !== false;

  const btnStyle = (f: FilterState) => ({
    padding: "8px 16px",
    borderRadius: "6px",
    border: "none",
    cursor: "pointer",
    background: filter === f ? "#3b82f6" : "#1f2937",
    color: "#fff",
    fontWeight: "bold" as const,
    transition: "background 0.2s",
  });

  return (
    <div style={{ background: "#0b1020", color: "#fff", minHeight: "100vh", padding: 20 }}>
      {/* --- HEADER --- */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0 }}>Drone Swarm Telemetry</h1>
          <div style={{ color: "#9ca3af", fontSize: 14 }}>
            Viewing {filteredDrones.length} of {Object.keys(drones).length} drones
          </div>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {/* Filter Group */}
          <div style={{ display: "flex", background: "#111827", padding: 4, borderRadius: 8, gap: 4 }}>
            <button style={btnStyle("all")} onClick={() => setFilter("all")}>All</button>
            <button style={btnStyle("searching")} onClick={() => setFilter("searching")}>Searching</button>
            <button style={btnStyle("tracking")} onClick={() => setFilter("tracking")}>Tracking</button>
            <button style={btnStyle("transit")} onClick={() => setFilter("transit")}>Transit</button>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center", marginLeft: 12 }}>
            <div style={{ 
              width: 12, height: 12, borderRadius: 6, 
              background: connected ? "#22c55e" : "#ef4444" 
            }} />
            <div>{connected ? "Live" : "Offline"}</div>
          </div>
        </div>
      </div>

      {/* --- DRONE GRID --- */}
      <div style={{ 
        display: "grid", 
        gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", 
        gap: 16 
      }}>
        {filteredDrones.map((drone) => {
          const videoActive = isVideoOpen(drone.key);

          return (
            <div 
              key={drone.key} 
              style={{ 
                padding: 16, 
                background: "#0f1724", 
                borderRadius: 12, 
                border: "1px solid #1e293b",
                display: "flex",
                flexDirection: "column",
                gap: 12
              }}
            >
              {/* Card Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                  <div style={{ fontWeight: "bold", fontSize: 18 }}>{drone.label}</div>
                  <div style={{ 
                    display: "inline-block", 
                    padding: "2px 8px", 
                    borderRadius: 4, 
                    fontSize: 11, 
                    marginTop: 4,
                    background: drone.state === "tracking" ? "#7c3aed" : 
                                drone.state === "searching" ? "#2563eb" : "#4b5563",
                    textTransform: "uppercase",
                    fontWeight: "bold"
                  }}>
                    {drone.state}
                  </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                  <div style={{ fontSize: 14 }}>{Math.round(drone.battery)}% 🔋</div>
                  <button
                    onClick={() => toggleVideo(drone.key)}
                    style={{
                      background: videoActive ? "#ef4444" : "#3b82f6",
                      color: "#fff",
                      border: "none",
                      padding: "6px 12px",
                      borderRadius: 6,
                      cursor: "pointer",
                      fontSize: 12,
                      fontWeight: "bold"
                    }}
                  >
                    {videoActive ? "Close Camera" : "Open Camera"}
                  </button>
                </div>
              </div>

              {/* Stats Row */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <div style={{ background: "#1e293b", padding: 8, borderRadius: 6 }}>
                  <small style={{ color: "#9ca3af", display: "block" }}>Altitude</small>
                  <span style={{ fontSize: 15 }}>{drone.altitude_m.toFixed(2)}m</span>
                </div>
                <div style={{ background: "#1e293b", padding: 8, borderRadius: 6 }}>
                  <small style={{ color: "#9ca3af", display: "block" }}>Local Position</small>
                  <span style={{ fontSize: 15 }}>{drone.position.x.toFixed(0)}, {drone.position.z.toFixed(0)}</span>
                </div>
              </div>

              {/* Video Player Section */}
              {videoActive && drone.instanceId !== undefined && (
                <div style={{ 
                  marginTop: 4, 
                  borderRadius: 8, 
                  overflow: "hidden", 
                  border: "1px solid #334155",
                  background: "#000",
                  aspectRatio: "16/9"
                }}>
                  <DroneVideoPlayer instanceId={drone.instanceId} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Empty State */}
      {filteredDrones.length === 0 && (
        <div style={{ textAlign: "center", marginTop: 100, color: "#4b5563" }}>
          <h2>No drones currently in {filter} state</h2>
          <p>Switch filters or wait for drones to change state in Unity.</p>
        </div>
      )}
    </div>
  );
}
