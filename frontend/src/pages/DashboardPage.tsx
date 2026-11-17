import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Users,
  Crosshair,
  Truck,
  ShieldAlert,
  AlertTriangle,
  Zap,
  Activity,
  Layers,
  Radio,
  ArrowUpRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend
} from 'recharts';
import { DashboardStats, Camera } from '@/types';
import { formatNumber } from '@/lib/utils';
import { Link } from 'react-router-dom';

export const DashboardPage: React.FC = () => {
  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ['stats'],
    queryFn: async () => {
      const res = await fetch('/api/stats');
      if (!res.ok) throw new Error('Failed to fetch dashboard stats');
      return res.json();
    },
    refetchInterval: 5000,
  });

  const { data: cameras } = useQuery<Camera[]>({
    queryKey: ['cameras'],
    queryFn: async () => {
      const res = await fetch('/api/cameras');
      if (!res.ok) throw new Error('Failed to fetch cameras');
      return res.json();
    },
  });

  const kpis = stats?.kpis || {
    total_detections: 261,
    persons: 66,
    weapons: 77,
    vehicles: 26,
    military_vehicles: 92,
    armed_persons: 0,
    active_critical_alerts: 0,
    total_alerts: 9,
    model_mAP50: 0.812,
    avg_inference_ms: 48.5,
  };

  const kpiCards = [
    { label: 'Total Detections', value: formatNumber(kpis.total_detections), sub: 'Across 4 cameras', icon: Layers, color: 'text-orange-400', border: 'border-orange-500/20' },
    { label: 'Persons', value: formatNumber(kpis.persons), sub: 'Perimeter entities', icon: Users, color: 'text-cyan-400', border: 'border-cyan-500/20' },
    { label: 'Vehicles', value: formatNumber(kpis.vehicles), sub: 'Logistics / Transport', icon: Truck, color: 'text-emerald-400', border: 'border-emerald-500/20' },
    { label: 'Weapons', value: formatNumber(kpis.weapons), sub: 'Firearms & hardware', icon: Crosshair, color: 'text-yellow-400', border: 'border-yellow-500/20' },
    { label: 'Armed Persons', value: formatNumber(kpis.armed_persons), sub: 'Spatial proximity links', icon: ShieldAlert, color: 'text-red-400', border: 'border-red-500/20' },
    { label: 'Critical Alerts', value: formatNumber(kpis.active_critical_alerts), sub: 'Restricted fence breach', icon: AlertTriangle, color: 'text-rose-400', border: 'border-rose-500/20' },
    { label: 'Model mAP@50', value: `${(kpis.model_mAP50 * 100).toFixed(1)}%`, sub: 'YOLOv8s benchmark', icon: Activity, color: 'text-purple-400', border: 'border-purple-500/20' },
    { label: 'Inference Latency', value: `${kpis.avg_inference_ms.toFixed(1)} ms`, sub: '~20 FPS CPU realtime', icon: Zap, color: 'text-green-400', border: 'border-green-500/20' },
  ];

  const trendData = stats?.trend_data || [
    { time: '00:00', persons: 14, vehicles: 4, alerts: 2 },
    { time: '04:00', persons: 22, vehicles: 2, alerts: 5 },
    { time: '08:00', persons: 18, vehicles: 8, alerts: 1 },
    { time: '12:00', persons: 9, vehicles: 12, alerts: 0 },
    { time: '16:00', persons: 16, vehicles: 9, alerts: 2 },
    { time: '20:00', persons: 25, vehicles: 5, alerts: 4 },
    { time: '23:00', persons: 19, vehicles: 3, alerts: 3 },
  ];

  const classDist = stats?.class_distribution || [
    { name: 'Persons', value: 66, color: '#22D3EE' },
    { name: 'Vehicles', value: 26, color: '#22C55E' },
    { name: 'Weapons', value: 77, color: '#EAB308' },
    { name: 'Military Vehicles', value: 92, color: '#F97316' },
  ];

  const priorityData = stats?.priority_counts || [
    { priority: 'CRITICAL', count: 0, color: '#EF4444' },
    { priority: 'HIGH', count: 4, color: '#F97316' },
    { priority: 'MEDIUM', count: 5, color: '#EAB308' },
    { priority: 'LOW', count: 0, color: '#22D3EE' },
  ];

  const recentAlerts = stats?.recent_alerts || [];

  return (
    <div className="space-y-6">
      {/* 8 Top KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {kpiCards.map((kpi, idx) => {
          const Icon = kpi.icon;
          return (
            <div
              key={idx}
              className={`glass-card p-4 flex flex-col justify-between border ${kpi.border} hover:scale-[1.02] transition-transform`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-medium text-[#9CA3AF] tracking-wide uppercase">
                  {kpi.label}
                </span>
                <Icon className={`w-4 h-4 ${kpi.color}`} />
              </div>
              <div className="mt-3">
                <div className="text-2xl font-mono font-bold text-white tracking-tight">
                  {kpi.value}
                </div>
                <div className="text-[11px] text-[#9CA3AF] font-mono mt-0.5">
                  {kpi.sub}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Priority Feeds Strip */}
      <div className="glass-card p-5">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-orange-400 animate-pulse" />
            <h3 className="text-base font-bold text-white">Priority Surveillance Feeds</h3>
            <span className="text-xs font-mono text-orange-400 bg-orange-500/10 border border-orange-500/20 px-2 py-0.5 rounded">
              REPLAY STREAM
            </span>
          </div>
          <Link
            to="/monitor"
            className="flex items-center gap-1 text-xs font-medium text-orange-400 hover:text-orange-300 font-mono transition-colors"
          >
            <span>Open All 4 Cameras</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {cameras?.slice(0, 2).map((cam) => (
            <div key={cam.id} className="relative rounded-xl overflow-hidden bg-black border border-[rgba(255,255,255,0.08)]">
              <div className="absolute top-3 left-3 z-10 flex items-center gap-2 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-md text-xs font-mono text-white border border-white/10">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="font-bold">{cam.name}</span>
                <span className="text-[#9CA3AF]">· {cam.location}</span>
              </div>
              <div className="absolute top-3 right-3 z-10 bg-black/60 backdrop-blur-md px-2 py-0.5 rounded text-[11px] font-mono text-orange-400 border border-orange-500/30">
                {cam.mode}
              </div>
              <video
                src={cam.annotated_url || cam.stream_url}
                autoPlay
                loop
                muted
                playsInline
                className="w-full aspect-video object-cover"
              />
              <div className="p-3 bg-[#0F1318] flex items-center justify-between text-xs font-mono text-[#9CA3AF] border-t border-[rgba(255,255,255,0.06)]">
                <span>Threat: <strong className={cam.threat_level === 'CRITICAL' ? 'text-red-400' : 'text-emerald-400'}>{cam.threat_level}</strong></span>
                <span>FPS: {cam.fps}</span>
                <span>Res: {cam.resolution}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Detection Trend Area Chart */}
        <div className="lg:col-span-2 glass-card p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-white">Detection Trend (24h)</h3>
                <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">Timeline of personnel, vehicles, and alert triggers</p>
              </div>
              <span className="text-[11px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                Hourly Timeline
              </span>
            </div>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorPersons" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22D3EE" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22D3EE" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="colorAlerts" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#F97316" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#F97316" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                  <YAxis stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#11161B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#FFFFFF' }}
                  />
                  <Area type="monotone" dataKey="persons" stroke="#22D3EE" fillOpacity={1} fill="url(#colorPersons)" strokeWidth={2} name="Persons" />
                  <Area type="monotone" dataKey="alerts" stroke="#F97316" fillOpacity={1} fill="url(#colorAlerts)" strokeWidth={2} name="Alerts" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Donut Chart: Class Breakdown */}
        <div className="glass-card p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white mb-1">Class Distribution</h3>
            <p className="text-xs text-[#9CA3AF] font-mono mb-4">Neural ontology partition</p>
            <div className="h-48 w-full flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={classDist}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={75}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {classDist.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} stroke="rgba(0,0,0,0.5)" />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#11161B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#FFFFFF' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[rgba(255,255,255,0.06)] text-xs font-mono">
            {classDist.map((item, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
                <span className="text-[#9CA3AF] truncate">{item.name}:</span>
                <span className="text-white font-bold">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Alerts by Priority Bar Chart & Recent Alerts Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Bar chart */}
        <div className="glass-card p-5">
          <h3 className="text-base font-bold text-white mb-1">Alerts by Severity</h3>
          <p className="text-xs text-[#9CA3AF] font-mono mb-4">Threat level classification</p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={priorityData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="priority" stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 10 }} />
                <YAxis stroke="#6B7280" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#11161B', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px', color: '#FFFFFF' }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {priorityData.map((entry, index) => (
                    <Cell key={`bar-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Alerts Feed */}
        <div className="lg:col-span-2 glass-card p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base font-bold text-white">Recent Security Alerts</h3>
            <Link to="/alerts" className="text-xs font-mono text-orange-400 hover:text-orange-300">
              View All Logs &rarr;
            </Link>
          </div>
          <div className="divide-y divide-[rgba(255,255,255,0.06)] overflow-y-auto max-h-56">
            {recentAlerts.length === 0 ? (
              <p className="text-sm text-[#9CA3AF] py-6 text-center font-mono">No active alerts recorded.</p>
            ) : (
              recentAlerts.map((alt) => (
                <div key={alt.id} className="py-2.5 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-3">
                    <span
                      className={`px-2 py-0.5 rounded font-mono font-bold text-[10px] ${
                        alt.priority === 'CRITICAL'
                          ? 'bg-red-500/15 text-red-400 border border-red-500/30'
                          : alt.priority === 'HIGH'
                          ? 'bg-orange-500/15 text-orange-400 border border-orange-500/30'
                          : 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/30'
                      }`}
                    >
                      {alt.priority}
                    </span>
                    <div>
                      <p className="text-white font-medium">{alt.message}</p>
                      <p className="text-[#9CA3AF] font-mono text-[11px]">
                        {alt.camera} &middot; {alt.zone}
                      </p>
                    </div>
                  </div>
                  <span className="font-mono text-[#9CA3AF] shrink-0">{alt.time}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

