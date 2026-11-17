import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Clock } from 'lucide-react';

const titleMap: Record<string, { title: string; subtitle: string }> = {
  '/': { title: 'Dashboard', subtitle: 'Surveillance Analytics & Priority Feeds' },
  '/monitor': { title: 'Live Monitor', subtitle: 'Synchronized Multi-Camera Perimeter Canvas' },
  '/analyze': { title: 'Upload & Analyze', subtitle: 'Direct Inference on External Media' },
  '/alerts': { title: 'Tactical Alerts', subtitle: 'Prioritized Perimeter Incident Log' },
  '/dataset': { title: 'Dataset Explorer', subtitle: 'Open Images V7 & Defense Benchmarks' },
  '/model': { title: 'Model Performance', subtitle: 'YOLOv8 Metrics, Precision & Confusion Matrix' },
};

export const Header: React.FC = () => {
  const location = useLocation();
  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString(undefined, { hour12: false }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const meta = titleMap[location.pathname] || { title: 'Surveillance', subtitle: 'Border Intelligence' };

  return (
    <header className="flex items-center justify-between h-16 px-6 border-b border-[rgba(255,255,255,0.08)] bg-[#0A0D10]/80 backdrop-blur-md z-20 shrink-0">
      {/* Title & Subtitle */}
      <div>
        <h1 className="text-xl font-bold tracking-tight text-white leading-none">
          {meta.title}
        </h1>
        <p className="text-xs text-[#9CA3AF] mt-1 font-mono tracking-wide">
          {meta.subtitle}
        </p>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-4">
        {/* Status Chip */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-xs font-mono font-medium shadow-[0_0_10px_rgba(34,197,94,0.1)]">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>System Online</span>
        </div>

        {/* Local Clock */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-[#11161B] border border-[rgba(255,255,255,0.08)] text-white text-xs font-mono font-semibold">
          <Clock className="w-3.5 h-3.5 text-orange-400" />
          <span>{timeStr || '--:--:--'}</span>
        </div>
      </div>
    </header>
  );
};

