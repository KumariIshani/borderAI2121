import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  Download,
  Filter,
  Search,
  CheckCircle,
  Clock,
  MapPin,
  Eye,
  X
} from 'lucide-react';
import { Alert } from '@/types';

export const AlertsPage: React.FC = () => {
  const [priorityFilter, setPriorityFilter] = useState<string>('ALL');
  const [cameraFilter, setCameraFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  const { data: alerts = [], isLoading } = useQuery<Alert[]>({
    queryKey: ['alerts', priorityFilter, cameraFilter],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (priorityFilter !== 'ALL') params.append('priority', priorityFilter);
      if (cameraFilter !== 'ALL') params.append('cam', cameraFilter);
      const res = await fetch(`/api/alerts?${params.toString()}`);
      if (!res.ok) return [];
      return res.json();
    },
    refetchInterval: 5000,
  });

  const filteredAlerts = alerts.filter((a) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return (
      a.id.toLowerCase().includes(q) ||
      a.message.toLowerCase().includes(q) ||
      a.camera.toLowerCase().includes(q) ||
      a.zone.toLowerCase().includes(q)
    );
  });

  const exportCSV = () => {
    if (filteredAlerts.length === 0) return;
    const headers = ['id', 'time', 'camera', 'class', 'confidence', 'zone', 'priority', 'message'];
    const rows = filteredAlerts.map((a) => [
      a.id,
      a.time,
      `"${a.camera}"`,
      a.class,
      a.confidence,
      `"${a.zone}"`,
      a.priority,
      `"${a.message.replace(/"/g, '""')}"`,
    ]);
    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sentinel_alerts_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Controls Bar */}
      <div className="glass-card p-4 flex flex-wrap items-center justify-between gap-4">
        {/* Search */}
        <div className="relative min-w-[260px] flex-1 max-w-md">
          <Search className="w-4 h-4 text-[#9CA3AF] absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search alerts by ID, camera, or description..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-[#0F1318] border border-[rgba(255,255,255,0.1)] text-white text-xs font-mono placeholder:text-[#6B7280] focus:border-orange-500 focus:outline-none"
          />
        </div>

        {/* Priority & Camera Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-orange-400" />
            <select
              value={priorityFilter}
              onChange={(e) => setPriorityFilter(e.target.value)}
              className="bg-[#0F1318] border border-[rgba(255,255,255,0.1)] rounded-xl px-3 py-1.5 text-xs font-mono text-white focus:outline-none"
            >
              <option value="ALL">All Priorities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
          </div>

          <select
            value={cameraFilter}
            onChange={(e) => setCameraFilter(e.target.value)}
            className="bg-[#0F1318] border border-[rgba(255,255,255,0.1)] rounded-xl px-3 py-1.5 text-xs font-mono text-white focus:outline-none"
          >
            <option value="ALL">All Cameras</option>
            <option value="cam01">CAM-01 (North Fence)</option>
            <option value="cam02">CAM-02 (Gate Road)</option>
            <option value="cam03">CAM-03 (Checkpoint)</option>
            <option value="cam04">CAM-04 (Open Field)</option>
            <option value="weapon_test">TEST-01 (Weapon Test)</option>
          </select>

          {/* Export CSV button */}
          <button
            onClick={exportCSV}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white font-mono text-xs font-semibold transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-orange-400" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#0D1115] border-b border-[rgba(255,255,255,0.08)] text-[#9CA3AF] uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Alert ID</th>
                <th className="py-3 px-4">Priority</th>
                <th className="py-3 px-4">Camera</th>
                <th className="py-3 px-4">Entity</th>
                <th className="py-3 px-4">Zone</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[rgba(255,255,255,0.06)] text-white">
              {filteredAlerts.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-[#9CA3AF]">
                    No alerts match the selected criteria.
                  </td>
                </tr>
              ) : (
                filteredAlerts.map((alt) => (
                  <tr
                    key={alt.id}
                    onClick={() => setSelectedAlert(alt)}
                    className="hover:bg-white/5 cursor-pointer transition-colors"
                  >
                    <td className="py-3 px-4 font-bold text-orange-400">{alt.id}</td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          alt.priority === 'CRITICAL'
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : alt.priority === 'HIGH'
                            ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                            : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                        }`}
                      >
                        {alt.priority}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-semibold">{alt.camera}</td>
                    <td className="py-3 px-4 uppercase">{alt.class}</td>
                    <td className="py-3 px-4 text-[#9CA3AF]">{alt.zone}</td>
                    <td className="py-3 px-4">{Math.round(alt.confidence * 100)}%</td>
                    <td className="py-3 px-4 text-[#9CA3AF]">{alt.time}</td>
                    <td className="py-3 px-4 text-right">
                      <button className="text-orange-400 hover:text-orange-300 font-bold">
                        Inspect &rarr;
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Alert Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="glass-card max-w-lg w-full p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[rgba(255,255,255,0.08)]">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-orange-400" />
                <h3 className="text-base font-bold text-white">{selectedAlert.id} Incident Details</h3>
              </div>
              <button
                onClick={() => setSelectedAlert(null)}
                className="p-1 rounded text-[#9CA3AF] hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs font-mono">
              <div className="p-3 bg-[#0C1014] rounded-lg border border-[rgba(255,255,255,0.06)]">
                <p className="text-white text-sm font-sans font-medium mb-1">{selectedAlert.message}</p>
                <p className="text-[#9CA3AF]">Logged by autonomous rule evaluator.</p>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[#9CA3AF]">
                <div>Camera: <strong className="text-white">{selectedAlert.camera}</strong></div>
                <div>Priority: <strong className="text-orange-400">{selectedAlert.priority}</strong></div>
                <div>Security Band: <strong className="text-white">{selectedAlert.zone}</strong></div>
                <div>Confidence: <strong className="text-white">{Math.round(selectedAlert.confidence * 100)}%</strong></div>
                <div>Timestamp: <strong className="text-white">{selectedAlert.time}</strong></div>
                {selectedAlert.bbox && (
                  <div>Box: <strong className="text-white">[{selectedAlert.bbox.map((v) => Math.round(v)).join(', ')}]</strong></div>
                )}
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedAlert(null)}
                className="px-4 py-2 rounded-lg bg-orange-500 hover:bg-orange-600 text-black font-bold text-xs font-mono"
              >
                Acknowledge &amp; Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

