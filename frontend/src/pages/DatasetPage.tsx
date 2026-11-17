import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Database,
  Image as ImageIcon,
  Layers,
  Shield,
  FileText,
  X,
  ExternalLink,
  ZoomIn
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import { DatasetStats, DatasetSample } from '@/types';

export const DatasetPage: React.FC = () => {
  const [activeSample, setActiveSample] = useState<DatasetSample | null>(null);

  const { data: stats } = useQuery<DatasetStats>({
    queryKey: ['dataset-stats'],
    queryFn: async () => {
      const res = await fetch('/api/dataset/stats');
      if (!res.ok) throw new Error('Failed to fetch dataset stats');
      return res.json();
    },
  });

  const { data: samples = [] } = useQuery<DatasetSample[]>({
    queryKey: ['dataset-samples'],
    queryFn: async () => {
      const res = await fetch('/api/dataset/samples');
      if (!res.ok) return [];
      return res.json();
    },
  });

  const classData = [
    { name: 'Person', count: 736, color: '#22D3EE' },
    { name: 'Weapon', count: 514, color: '#EAB308' },
    { name: 'Vehicle', count: 70, color: '#22C55E' },
    { name: 'Military Vehicle', count: 94, color: '#F97316' },
  ];

  return (
    <div className="space-y-6">
      {/* Splits Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card p-5 border border-cyan-500/20">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-cyan-400 uppercase tracking-wider">
              Train Split (70%)
            </span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-3xl font-mono font-bold text-white mt-3">321 Images</div>
          <div className="text-xs font-mono text-[#9CA3AF] mt-1">999 ground-truth bounding boxes</div>
        </div>

        <div className="glass-card p-5 border border-emerald-500/20">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-emerald-400 uppercase tracking-wider">
              Validation Split (15%)
            </span>
            <Layers className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-mono font-bold text-white mt-3">68 Images</div>
          <div className="text-xs font-mono text-[#9CA3AF] mt-1">202 ground-truth bounding boxes</div>
        </div>

        <div className="glass-card p-5 border border-orange-500/20">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono font-bold text-orange-400 uppercase tracking-wider">
              Test Benchmark Split (15%)
            </span>
            <Shield className="w-4 h-4 text-orange-400" />
          </div>
          <div className="text-3xl font-mono font-bold text-white mt-3">70 Images</div>
          <div className="text-xs font-mono text-[#9CA3AF] mt-1">213 ground-truth bounding boxes</div>
        </div>
      </div>

      {/* Class Distribution & Box Geometry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Class Bar Chart */}
        <div className="lg:col-span-2 glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-white">Class Annotation Breakdown</h3>
              <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">Instance counts across train/val/test splits</p>
            </div>
            <span className="text-xs font-mono text-orange-400 bg-orange-500/10 px-2 py-0.5 rounded border border-orange-500/25">
              1,414 Total Instances
            </span>
          </div>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                <YAxis stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#11161B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#FFFFFF' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {classData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bounding-box Geometry Card */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white mb-1">Bounding-Box Geometry</h3>
            <p className="text-xs text-[#9CA3AF] font-mono mb-4">Spatial distribution statistics</p>

            <div className="space-y-3 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-[#0F1318] border border-[rgba(255,255,255,0.06)] flex justify-between">
                <span className="text-[#9CA3AF]">Small Boxes (&lt; 32x32)</span>
                <span className="text-white font-bold">28.4% (Weapons)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#0F1318] border border-[rgba(255,255,255,0.06)] flex justify-between">
                <span className="text-[#9CA3AF]">Medium Boxes (32-96)</span>
                <span className="text-white font-bold">42.1% (Persons)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#0F1318] border border-[rgba(255,255,255,0.06)] flex justify-between">
                <span className="text-[#9CA3AF]">Large Boxes (&gt; 96x96)</span>
                <span className="text-white font-bold">29.5% (Vehicles)</span>
              </div>
              <div className="p-2.5 rounded-lg bg-[#0F1318] border border-[rgba(255,255,255,0.06)] flex justify-between">
                <span className="text-[#9CA3AF]">Mean Aspect Ratio (W/H)</span>
                <span className="text-white font-bold">0.68 (Tall)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Ground Truth Image Gallery */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-orange-400" />
            <h3 className="text-base font-bold text-white">Ground-Truth Annotated Samples</h3>
            <span className="text-xs font-mono text-[#9CA3AF]">({samples.length} Benchmark Images)</span>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
            Click thumbnail to expand
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {samples.map((s) => (
            <div
              key={s.id}
              onClick={() => setActiveSample(s)}
              className="relative rounded-lg overflow-hidden border border-[rgba(255,255,255,0.08)] bg-black group cursor-pointer aspect-square"
            >
              <img
                src={s.url}
                alt={s.title}
                loading="lazy"
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <ZoomIn className="w-5 h-5 text-white" />
              </div>
              <div className="absolute bottom-0 inset-x-0 bg-black/70 backdrop-blur-sm p-1 text-[10px] font-mono text-center text-[#E5E7EB] truncate">
                {s.title}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Source and Licensing Note */}
      <div className="glass-card p-5 border border-[rgba(255,255,255,0.08)]">
        <div className="flex items-start gap-3">
          <FileText className="w-5 h-5 text-orange-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-sm font-bold text-white">Dataset Provenance &amp; Licensing Notice</h4>
            <p className="text-xs text-[#9CA3AF] leading-relaxed">
              This surveillance dataset is curated from Google's Open Images V7 repository, the Roboflow Defense Benchmark, and U.S. DoD DVIDS public-domain archives. Distributed under Creative Commons Attribution 4.0 International (CC BY 4.0) and public domain terms for scientific and educational verification.
            </p>
          </div>
        </div>
      </div>

      {/* Lightbox Modal */}
      {activeSample && (
        <div
          onClick={() => setActiveSample(null)}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-md"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative max-w-4xl max-h-[90vh] glass-card p-2 overflow-hidden flex flex-col"
          >
            <div className="flex items-center justify-between p-3 border-b border-[rgba(255,255,255,0.08)]">
              <span className="text-sm font-mono font-bold text-white">{activeSample.title}</span>
              <button
                onClick={() => setActiveSample(null)}
                className="p-1 rounded text-[#9CA3AF] hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-2 overflow-auto flex items-center justify-center">
              <img
                src={activeSample.url}
                alt={activeSample.title}
                className="max-h-[75vh] w-auto rounded object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

