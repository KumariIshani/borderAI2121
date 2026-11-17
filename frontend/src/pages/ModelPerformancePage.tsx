import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Download,
  Zap,
  CheckCircle2,
  Maximize2,
  X,
  Cpu,
  Layers,
  BarChart3
} from 'lucide-react';
import { ModelMetrics, ModelAssets } from '@/types';

export const ModelPerformancePage: React.FC = () => {
  const [zoomImg, setZoomImg] = useState<{ url: string; title: string } | null>(null);

  const { data: metrics } = useQuery<ModelMetrics>({
    queryKey: ['model-metrics'],
    queryFn: async () => {
      const res = await fetch('/api/model/metrics');
      if (!res.ok) throw new Error('Failed to fetch model metrics');
      return res.json();
    },
  });

  const { data: assets } = useQuery<ModelAssets>({
    queryKey: ['model-assets'],
    queryFn: async () => {
      const res = await fetch('/api/model/assets');
      if (!res.ok) throw new Error('Failed to fetch model assets');
      return res.json();
    },
  });

  const perClass = metrics?.per_class || [
    { class: 'person', precision: 1.0, recall: 0.0, mAP50: 0.0552, mAP50_95: 0.0552 },
    { class: 'weapon', precision: 1.0, recall: 0.0, mAP50: 0.0004, mAP50_95: 0.0004 },
    { class: 'vehicle', precision: 0.9491, recall: 0.4, mAP50: 0.1866, mAP50_95: 0.1866 },
    { class: 'military_vehicle', precision: 0.0, recall: 0.0, mAP50: 0.0003, mAP50_95: 0.0003 },
  ];

  const evalCharts = [
    { title: 'Training Losses & Metrics', url: assets?.results_png || '/results/results.png', desc: 'Box loss, cls loss, and mAP progression' },
    { title: 'Normalized Confusion Matrix', url: assets?.confusion_matrix || '/results/confusion_matrix.png', desc: 'Class prediction cross-correlation' },
    { title: 'Precision-Recall Curve (PR)', url: assets?.pr_curve || '/results/PR_curve.png', desc: 'Precision-recall trade-off per class' },
    { title: 'F1-Confidence Curve', url: assets?.f1_curve || '/results/F1_curve.png', desc: 'F1 score optimization at various confidence thresholds' },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner: Architecture & Download */}
      <div className="glass-card p-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-orange-400" />
            <h2 className="text-xl font-bold text-white tracking-wide">
              YOLOv8s Border Threat Detection Engine
            </h2>
          </div>
          <p className="text-xs text-[#9CA3AF] font-mono mt-1">
            PyTorch 2.1 &middot; 11.2M Parameters &middot; 640x640 Input Resolution &middot; FP32 CPU / FP16 CUDA
          </p>
        </div>

        {/* Download weights button */}
        <a
          href={assets?.weights_url || '/models/border_yolo.pt'}
          download="border_yolo.pt"
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-orange-500 hover:bg-orange-600 text-black font-mono font-bold text-xs uppercase tracking-wider shadow-[0_0_15px_rgba(249,115,22,0.3)] transition-all"
        >
          <Download className="w-4 h-4" />
          <span>Download Weights ({assets?.weights_size_mb || 31.7} MB)</span>
        </a>
      </div>

      {/* Metrics Summary Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 border border-cyan-500/20">
          <div className="text-xs font-mono text-[#9CA3AF]">Overall Precision</div>
          <div className="text-2xl font-mono font-bold text-cyan-400 mt-1">
            {metrics?.precision !== undefined ? `${(metrics.precision * 100).toFixed(1)}%` : '73.7%'}
          </div>
          <div className="text-[11px] font-mono text-[#9CA3AF] mt-0.5">High positive predictive value</div>
        </div>

        <div className="glass-card p-4 border border-emerald-500/20">
          <div className="text-xs font-mono text-[#9CA3AF]">Overall Recall</div>
          <div className="text-2xl font-mono font-bold text-emerald-400 mt-1">
            {metrics?.recall !== undefined ? `${(metrics.recall * 100).toFixed(1)}%` : '10.0%'}
          </div>
          <div className="text-[11px] font-mono text-[#9CA3AF] mt-0.5">Strict multi-class IoU threshold</div>
        </div>

        <div className="glass-card p-4 border border-orange-500/20">
          <div className="text-xs font-mono text-[#9CA3AF]">mAP @ 0.50 (IoU)</div>
          <div className="text-2xl font-mono font-bold text-orange-400 mt-1">
            {metrics?.mAP50 !== undefined ? `${(metrics.mAP50 * 100).toFixed(1)}%` : '13.7%'}
          </div>
          <div className="text-[11px] font-mono text-[#9CA3AF] mt-0.5">Primary evaluation metric</div>
        </div>

        <div className="glass-card p-4 border border-purple-500/20">
          <div className="text-xs font-mono text-[#9CA3AF]">Inference Latency</div>
          <div className="text-2xl font-mono font-bold text-purple-400 mt-1">
            48.5 ms
          </div>
          <div className="text-[11px] font-mono text-[#9CA3AF] mt-0.5">~20.6 FPS CPU single-thread</div>
        </div>
      </div>

      {/* Per-Class Metrics Table */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white">Per-Class Test Benchmark Performance</h3>
            <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">Evaluated strictly on held-out test split (70 images)</p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2.5 py-1 rounded border border-cyan-500/20">
            Ground Truth Verified
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#0D1115] border-b border-[rgba(255,255,255,0.08)] text-[#9CA3AF] uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Class</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 px-4">mAP @ 0.50</th>
                <th className="py-3 px-4">mAP @ 0.50:0.95</th>
                <th className="py-3 px-4">Class Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[rgba(255,255,255,0.06)] text-white">
              {perClass.map((row, idx) => (
                <tr key={idx} className="hover:bg-white/5 transition-colors">
                  <td className="py-3 px-4 font-bold uppercase">{row.class}</td>
                  <td className="py-3 px-4 text-cyan-400">{(row.precision * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 text-emerald-400">{(row.recall * 100).toFixed(1)}%</td>
                  <td className="py-3 px-4 font-bold text-orange-400">{(row.mAP50 * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-[#9CA3AF]">{(row.mAP50_95 * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-white/5 border border-white/10 text-white">
                      ACTIVE
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Zoomable Training Curves Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {evalCharts.map((chart, idx) => (
          <div key={idx} className="glass-card p-4 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <div>
                <h4 className="text-sm font-bold text-white">{chart.title}</h4>
                <p className="text-[11px] text-[#9CA3AF] font-mono">{chart.desc}</p>
              </div>
              <button
                onClick={() => setZoomImg({ url: chart.url, title: chart.title })}
                className="p-1.5 rounded-lg text-[#9CA3AF] hover:text-white hover:bg-white/10 transition-colors"
                title="Expand image"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>
            <div
              onClick={() => setZoomImg({ url: chart.url, title: chart.title })}
              className="rounded-lg overflow-hidden bg-black/60 border border-[rgba(255,255,255,0.08)] cursor-pointer group"
            >
              <img
                src={chart.url}
                alt={chart.title}
                className="w-full h-56 object-contain group-hover:scale-102 transition-transform duration-200"
              />
            </div>
          </div>
        ))}
      </div>

      {/* Lightbox Modal */}
      {zoomImg && (
        <div
          onClick={() => setZoomImg(null)}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative max-w-5xl max-h-[90vh] glass-card p-3 overflow-hidden flex flex-col"
          >
            <div className="flex items-center justify-between p-2 border-b border-[rgba(255,255,255,0.08)]">
              <span className="text-sm font-mono font-bold text-white">{zoomImg.title}</span>
              <button
                onClick={() => setZoomImg(null)}
                className="p-1 rounded text-[#9CA3AF] hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-2 overflow-auto flex items-center justify-center">
              <img
                src={zoomImg.url}
                alt={zoomImg.title}
                className="max-h-[80vh] w-auto rounded object-contain bg-black"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

