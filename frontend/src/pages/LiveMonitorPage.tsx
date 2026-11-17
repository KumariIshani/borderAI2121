import React, { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Sliders,
  Eye,
  Layers,
  Maximize2,
  X,
  ShieldAlert,
  AlertTriangle,
  Radio,
  CheckCircle2,
  Info
} from 'lucide-react';
import { Camera, FrameDetection, BoxDetection } from '@/types';

export const LiveMonitorPage: React.FC = () => {
  const [showOverlays, setShowOverlays] = useState<boolean>(true);
  const [showZones, setShowZones] = useState<boolean>(true);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [selectedCam, setSelectedCam] = useState<Camera | null>(null);

  const { data: cameras, isLoading } = useQuery<Camera[]>({
    queryKey: ['cameras'],
    queryFn: async () => {
      const res = await fetch('/api/cameras');
      if (!res.ok) throw new Error('Failed to fetch cameras');
      return res.json();
    },
  });

  const fourCams = cameras?.slice(0, 4) || [];

  return (
    <div className="space-y-6">
      {/* Top Controls Bar */}
      <div className="glass-card p-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/30 text-orange-400 font-mono text-xs font-semibold">
            <Radio className="w-4 h-4 animate-pulse" />
            <span>4-WAY PERIMETER GRID</span>
          </div>
          <span className="text-xs text-[#9CA3AF] font-mono">
            Synced HTML5 Canvas Overlay &middot; REPLAY Mode
          </span>
        </div>

        {/* Filters & Toggles */}
        <div className="flex flex-wrap items-center gap-6">
          {/* Overlays toggle */}
          <label className="flex items-center gap-2 text-xs font-mono text-[#E5E7EB] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showOverlays}
              onChange={(e) => setShowOverlays(e.target.checked)}
              className="w-4 h-4 rounded border-gray-700 bg-gray-900 text-orange-500 focus:ring-0 focus:ring-offset-0"
            />
            <Eye className="w-3.5 h-3.5 text-cyan-400" />
            <span>Show AI Overlays</span>
          </label>

          {/* Zones toggle */}
          <label className="flex items-center gap-2 text-xs font-mono text-[#E5E7EB] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showZones}
              onChange={(e) => setShowZones(e.target.checked)}
              className="w-4 h-4 rounded border-gray-700 bg-gray-900 text-orange-500 focus:ring-0 focus:ring-offset-0"
            />
            <Layers className="w-3.5 h-3.5 text-emerald-400" />
            <span>Show Risk Zones</span>
          </label>

          {/* Confidence Slider */}
          <div className="flex items-center gap-2">
            <Sliders className="w-3.5 h-3.5 text-orange-400" />
            <span className="text-xs font-mono text-[#9CA3AF]">Confidence:</span>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.05}
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="w-24 accent-orange-500 cursor-pointer"
            />
            <span className="text-xs font-mono font-bold text-white w-10">
              {Math.round(confThreshold * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* 2x2 Camera Video + Synchronized Canvas Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {fourCams.map((cam) => (
          <CameraFeedCard
            key={cam.id}
            cam={cam}
            showOverlays={showOverlays}
            showZones={showZones}
            confThreshold={confThreshold}
            onExpand={() => setSelectedCam(cam)}
          />
        ))}
      </div>

      {/* Expanded Modal View when a camera is clicked */}
      {selectedCam && (
        <ExpandedCameraModal
          cam={selectedCam}
          showOverlays={showOverlays}
          showZones={showZones}
          confThreshold={confThreshold}
          onClose={() => setSelectedCam(null)}
        />
      )}
    </div>
  );
};

// -------------------------------------------------------------
// INDIVIDUAL CAMERA FEED CARD WITH REALTIME CANVAS OVERLAY
// -------------------------------------------------------------
interface FeedProps {
  cam: Camera;
  showOverlays: boolean;
  showZones: boolean;
  confThreshold: number;
  onExpand: () => void;
}

const CameraFeedCard: React.FC<FeedProps> = ({
  cam,
  showOverlays,
  showZones,
  confThreshold,
  onExpand,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [currentCounts, setCurrentCounts] = useState({ persons: 0, vehicles: 0, weapons: 0, armed: 0 });

  // Fetch detections JSON for this camera
  const { data: detections } = useQuery<FrameDetection[]>({
    queryKey: ['detections', cam.id],
    queryFn: async () => {
      const res = await fetch(`/api/detections/${cam.id}`);
      if (!res.ok) return [];
      return res.json();
    },
    staleTime: Infinity,
  });

  // Render loop syncing canvas overlay with video.currentTime
  useEffect(() => {
    let animId: number;

    const render = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && detections && detections.length > 0) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          // Sync canvas internal resolution with video
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
          }

          const w = canvas.width;
          const h = canvas.height;
          ctx.clearRect(0, 0, w, h);

          // 1. Draw horizontal security zone bands
          if (showZones) {
            // Restricted Fence Zone (bottom 35%: 0.65 to 1.00)
            ctx.fillStyle = 'rgba(239, 68, 68, 0.08)';
            ctx.fillRect(0, h * 0.65, w, h * 0.35);
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, h * 0.65);
            ctx.lineTo(w, h * 0.65);
            ctx.stroke();

            ctx.fillStyle = '#EF4444';
            ctx.font = 'bold 12px "JetBrains Mono", monospace';
            ctx.fillText('RESTRICTED FENCE ZONE (0.65 - 1.00)', 14, h * 0.65 + 18);

            // Tactical Buffer Zone (middle 30%: 0.35 to 0.65)
            ctx.fillStyle = 'rgba(249, 115, 22, 0.06)';
            ctx.fillRect(0, h * 0.35, w, h * 0.30);
            ctx.strokeStyle = 'rgba(249, 115, 22, 0.5)';
            ctx.beginPath();
            ctx.moveTo(0, h * 0.35);
            ctx.lineTo(w, h * 0.35);
            ctx.stroke();

            ctx.fillStyle = '#F97316';
            ctx.font = 'bold 12px "JetBrains Mono", monospace';
            ctx.fillText('TACTICAL BUFFER ZONE (0.35 - 0.65)', 14, h * 0.35 + 18);

            // Observation Zone (top 35%: 0.00 to 0.35)
            ctx.fillStyle = '#22D3EE';
            ctx.font = 'bold 12px "JetBrains Mono", monospace';
            ctx.fillText('OBSERVATION ZONE (0.00 - 0.35)', 14, 22);
          }

          // 2. Draw detections for the matching timestamp
          if (showOverlays) {
            const currentTime = video.currentTime;
            // Find closest frame
            let bestDiff = Infinity;
            let currentFrame: FrameDetection | null = null;
            for (const f of detections) {
              const diff = Math.abs(f.time - currentTime);
              if (diff < bestDiff) {
                bestDiff = diff;
                currentFrame = f;
              }
              if (diff > bestDiff) break;
            }

            if (currentFrame && currentFrame.boxes) {
              let pCount = 0;
              let vCount = 0;
              let wCount = 0;
              let aCount = 0;

              for (const box of currentFrame.boxes) {
                if (box.confidence < confThreshold) continue;

                const [x1, y1, x2, y2] = box.bbox;
                const bw = x2 - x1;
                const bh = y2 - y1;

                let strokeColor = '#22D3EE';
                let label = `PERSON ${Math.round(box.confidence * 100)}%`;

                if (box.armed) {
                  strokeColor = '#EF4444';
                  label = `! ARMED PERSON ${Math.round(box.confidence * 100)}%`;
                  aCount++;
                } else if (box.class === 'weapon') {
                  strokeColor = '#EAB308';
                  label = `WEAPON ${Math.round(box.confidence * 100)}%`;
                  wCount++;
                } else if (box.class === 'vehicle') {
                  strokeColor = '#22C55E';
                  label = `VEHICLE ${Math.round(box.confidence * 100)}%`;
                  vCount++;
                } else if (box.class === 'military_vehicle') {
                  strokeColor = '#F97316';
                  label = `MILITARY ${Math.round(box.confidence * 100)}%`;
                  vCount++;
                } else {
                  pCount++;
                }

                // Draw bounding box
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = box.armed ? 3 : 2;
                ctx.strokeRect(x1, y1, bw, bh);

                // Label background pill
                ctx.fillStyle = strokeColor;
                const textWidth = ctx.measureText(label).width;
                ctx.fillRect(x1, Math.max(0, y1 - 20), textWidth + 12, 20);

                ctx.fillStyle = '#000000';
                ctx.font = 'bold 11px "JetBrains Mono", monospace';
                ctx.fillText(label, x1 + 6, Math.max(14, y1 - 5));
              }

              setCurrentCounts({ persons: pCount, vehicles: vCount, weapons: wCount, armed: aCount });
            }
          }
        }
      }
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [detections, showOverlays, showZones, confThreshold]);

  return (
    <div className="glass-card overflow-hidden flex flex-col group relative">
      {/* Video Container */}
      <div className="relative aspect-video bg-black overflow-hidden">
        {/* Header HUD inside video */}
        <div className="absolute top-3 left-3 z-20 flex items-center gap-2 bg-black/70 backdrop-blur-md px-2.5 py-1 rounded-md text-xs font-mono text-white border border-white/10">
          <span className="pulse-dot-green" />
          <span className="font-bold">{cam.name}</span>
          <span className="text-[#9CA3AF]">· {cam.location}</span>
        </div>

        <div className="absolute top-3 right-3 z-20 flex items-center gap-2">
          <span className="bg-black/70 backdrop-blur-md px-2 py-0.5 rounded text-[11px] font-mono text-orange-400 border border-orange-500/30">
            {cam.mode}
          </span>
          <button
            onClick={onExpand}
            className="p-1.5 rounded bg-black/70 backdrop-blur-md text-white hover:text-orange-400 border border-white/10 transition-colors"
            title="Expand Full View"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Video stream */}
        <video
          ref={videoRef}
          src={cam.stream_url}
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-full object-cover"
        />

        {/* Synchronized Canvas Overlay */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none w-full h-full object-cover"
        />
      </div>

      {/* Footer HUD info */}
      <div className="p-3 bg-[#0F1318] flex items-center justify-between text-xs font-mono text-[#9CA3AF] border-t border-[rgba(255,255,255,0.06)]">
        <div className="flex items-center gap-3">
          <span>Threat: <strong className={cam.threat_level === 'CRITICAL' ? 'text-red-400' : 'text-emerald-400'}>{cam.threat_level}</strong></span>
          <span>FPS: <strong className="text-white">{cam.fps}</strong></span>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="text-cyan-400">P: {currentCounts.persons}</span>
          <span className="text-emerald-400">V: {currentCounts.vehicles}</span>
          <span className="text-yellow-400">W: {currentCounts.weapons}</span>
          {currentCounts.armed > 0 && <span className="text-red-400 font-bold">! ARMED: {currentCounts.armed}</span>}
        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// EXPANDED CAMERA MODAL WITH EVENT STREAM INSPECTOR
// -------------------------------------------------------------
interface ModalProps {
  cam: Camera;
  showOverlays: boolean;
  showZones: boolean;
  confThreshold: number;
  onClose: () => void;
}

const ExpandedCameraModal: React.FC<ModalProps> = ({
  cam,
  showOverlays,
  showZones,
  confThreshold,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md">
      <div className="relative w-full max-w-6xl glass-card overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-4 border-b border-[rgba(255,255,255,0.08)] bg-[#0C1014]">
          <div className="flex items-center gap-3">
            <span className="pulse-dot-green" />
            <h2 className="text-lg font-bold text-white tracking-wide">
              {cam.name} &middot; {cam.location}
            </h2>
            <span className="px-2 py-0.5 rounded text-xs font-mono bg-orange-500/10 text-orange-400 border border-orange-500/25">
              Expanded Video Stream
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#9CA3AF] hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body: Large Video + Telemetry Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-5 overflow-y-auto">
          {/* Main Video */}
          <div className="lg:col-span-2 relative rounded-xl overflow-hidden bg-black border border-[rgba(255,255,255,0.08)]">
            <video
              src={cam.annotated_url || cam.stream_url}
              autoPlay
              loop
              muted
              playsInline
              controls
              className="w-full aspect-video object-cover"
            />
          </div>

          {/* Telemetry & Specifications */}
          <div className="space-y-4">
            <div className="glass-card p-4">
              <h4 className="text-xs font-mono font-bold text-orange-400 uppercase tracking-wider mb-3">
                Telemetry &amp; Optics
              </h4>
              <table className="w-full text-xs font-mono">
                <tbody className="divide-y divide-[rgba(255,255,255,0.06)]">
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Sensor ID</td>
                    <td className="py-2 text-white font-bold text-right">{cam.id.toUpperCase()}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Sector Tag</td>
                    <td className="py-2 text-white font-bold text-right">{cam.location}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Resolution</td>
                    <td className="py-2 text-white font-bold text-right">{cam.resolution}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Frame Rate</td>
                    <td className="py-2 text-white font-bold text-right">{cam.fps} FPS</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Status</td>
                    <td className="py-2 text-emerald-400 font-bold text-right">{cam.status}</td>
                  </tr>
                  <tr>
                    <td className="py-2 text-[#9CA3AF]">Threat Posture</td>
                    <td className="py-2 text-right">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        cam.threat_level === 'CRITICAL' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      }`}>
                        {cam.threat_level}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="glass-card p-4">
              <h4 className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider mb-2">
                Operational Notice
              </h4>
              <p className="text-xs text-[#9CA3AF] leading-relaxed">
                Stream operates in REPLAY validation mode. Neural detections and spatial proximity logic are applied frame-by-frame with zero external network dependency.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

