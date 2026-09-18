import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";
import {
  EVENT_TYPE_LABELS,
  EVENT_TYPES,
  type EventType,
  type VisionEvent,
} from "./types";

const MAX_ROWS = 200;
const WS_RECONNECT_DELAY_MS = 2000;

// Empty string => same-origin relative paths ("/events", "/ws/live"), which
// works both in dev (via the Vite proxy configured in vite.config.ts) and in
// production (via the nginx reverse-proxy baked into the dashboard-ui image
// — see frontend/nginx.conf). Set VITE_API_BASE to call a different backend
// origin directly instead (e.g. during `vite dev` without the proxy).
const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const EVENT_TYPE_COLORS: Record<EventType, string> = {
  restricted_zone_entry: "#ff5d6c",
  line_crossing: "#4da3ff",
  dwell_time_exceeded: "#ffb454",
};

const DETECTION_FRAMES = [
  {
    src: "/screenshots/detection-frame-1.png",
    caption: "Real YOLOv8 detections + tracker IDs + zone violation (mirrored bus.jpg)",
  },
  {
    src: "/screenshots/detection-frame-2.png",
    caption: "Real YOLOv8 detections + restricted-zone overlay (bus.jpg)",
  },
  {
    src: "/screenshots/detection-frame-3.png",
    caption: "Real YOLOv8 detections + tracker IDs (mirrored zidane.jpg)",
  },
];

type ConnectionState = "connecting" | "connected" | "disconnected";

function formatTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

function formatConfidence(confidence: number): string {
  return confidence.toFixed(2);
}

function App() {
  const [events, setEvents] = useState<VisionEvent[]>([]);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | undefined>(undefined);

  const appendEvent = useCallback((event: VisionEvent) => {
    setEvents((prev) => {
      const next = [event, ...prev];
      return next.length > MAX_ROWS ? next.slice(0, MAX_ROWS) : next;
    });
  }, []);

  // Load recent events already sitting in the backend's ring buffer.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/events`);
        if (!res.ok) return;
        const data: VisionEvent[] = await res.json();
        if (!cancelled) {
          // /events returns oldest-first; the table renders newest-first.
          setEvents(data.slice(-MAX_ROWS).reverse());
        }
      } catch (err) {
        console.warn("Could not load recent events", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Live updates over WS /ws/live, with auto-reconnect.
  useEffect(() => {
    let stopped = false;

    function connect() {
      if (stopped) return;
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const base = API_BASE
        ? API_BASE.replace(/^http/, "ws")
        : `${proto}://${window.location.host}`;
      const ws = new WebSocket(`${base}/ws/live`);
      wsRef.current = ws;
      setConnectionState("connecting");

      ws.onopen = () => setConnectionState("connected");
      ws.onclose = () => {
        setConnectionState("disconnected");
        if (!stopped) {
          reconnectTimerRef.current = window.setTimeout(
            connect,
            WS_RECONNECT_DELAY_MS,
          );
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          appendEvent(JSON.parse(msg.data) as VisionEvent);
        } catch (err) {
          console.warn("Bad event payload", err);
        }
      };
    }

    connect();
    return () => {
      stopped = true;
      window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [appendEvent]);

  const counts = useMemo(() => {
    const byType: Record<EventType, number> = {
      restricted_zone_entry: 0,
      line_crossing: 0,
      dwell_time_exceeded: 0,
    };
    for (const evt of events) {
      byType[evt.event_type] = (byType[evt.event_type] ?? 0) + 1;
    }
    return byType;
  }, [events]);

  const chartData = useMemo(
    () =>
      EVENT_TYPES.map((type) => ({
        type,
        label: EVENT_TYPE_LABELS[type],
        count: counts[type],
      })),
    [counts],
  );

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <div>
          <h1>Video Analytics Dashboard</h1>
          <p className="subtitle">
            RTSP + YOLOv8 vision events, live from the analytics pipeline
          </p>
        </div>
        <span className={`status-badge status-${connectionState}`}>
          {connectionState === "connected" && "● connected"}
          {connectionState === "connecting" && "connecting…"}
          {connectionState === "disconnected" && "disconnected — retrying"}
        </span>
      </header>

      <main className="dashboard-main">
        <section className="summary-strip" aria-label="Event summary">
          <div className="summary-card total">
            <div className="summary-value">{events.length}</div>
            <div className="summary-label">Total events</div>
          </div>
          {EVENT_TYPES.map((type) => (
            <div className="summary-card" key={type}>
              <div
                className="summary-value"
                style={{ color: EVENT_TYPE_COLORS[type] }}
              >
                {counts[type]}
              </div>
              <div className="summary-label">{EVENT_TYPE_LABELS[type]}</div>
            </div>
          ))}
        </section>

        <section className="panel chart-panel" aria-label="Events by type">
          <h2>Events by type</h2>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#262b38" />
              <XAxis type="number" allowDecimals={false} stroke="#8b93a7" />
              <YAxis
                type="category"
                dataKey="label"
                width={170}
                stroke="#8b93a7"
                tick={{ fill: "#e6e8ee", fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  background: "#161923",
                  border: "1px solid #262b38",
                  borderRadius: 8,
                  color: "#e6e8ee",
                }}
              />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {chartData.map((entry) => (
                  <Cell key={entry.type} fill={EVENT_TYPE_COLORS[entry.type]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel" aria-label="Detections">
          <h2>Detections</h2>
          <p className="panel-subtitle">
            Real YOLOv8 inference through this repo's own detector, tracker,
            and zone-rule engine — bounding boxes, track IDs, and zone/line
            overlays drawn with OpenCV (see{" "}
            <code>scripts/generate_annotated_frames.py</code> for how, and
            why these use Ultralytics' bundled reference photos rather than
            the untextured synthetic demo video).
          </p>
          <div className="detections-grid">
            {DETECTION_FRAMES.map((frame) => (
              <figure key={frame.src} className="detection-frame">
                <img src={frame.src} alt={frame.caption} loading="lazy" />
                <figcaption>{frame.caption}</figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="panel" aria-label="Live event table">
          <h2>Live events</h2>
          {events.length === 0 ? (
            <div className="empty-state">Waiting for events&hellip;</div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Camera</th>
                    <th>Event</th>
                    <th>Track ID</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((evt, idx) => (
                    <tr key={`${evt.camera_id}-${evt.track_id}-${evt.timestamp}-${idx}`}>
                      <td>{formatTime(evt.timestamp)}</td>
                      <td>{evt.camera_id}</td>
                      <td>
                        <span className={`badge badge-${evt.event_type}`}>
                          {EVENT_TYPE_LABELS[evt.event_type]}
                        </span>
                      </td>
                      <td>{evt.track_id}</td>
                      <td>{formatConfidence(evt.confidence)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>

      <footer className="dashboard-footer">
        Data source: <code>GET /events</code> on load, then{" "}
        <code>WS /ws/live</code> for real-time updates.
      </footer>
    </div>
  );
}

export default App;
