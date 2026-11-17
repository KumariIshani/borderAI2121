import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  ShieldCheck,
  LayoutDashboard,
  Radio,
  UploadCloud,
  AlertTriangle,
  Database,
  Activity,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  name: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { name: 'Dashboard', path: '/', icon: LayoutDashboard },
  { name: 'Live Monitor', path: '/monitor', icon: Radio },
  { name: 'Upload & Analyze', path: '/analyze', icon: UploadCloud },
  { name: 'Alerts', path: '/alerts', icon: AlertTriangle },
  { name: 'Dataset', path: '/dataset', icon: Database },
  { name: 'Model Performance', path: '/model', icon: Activity },
];

export const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        'relative flex flex-col h-screen bg-[#090C0F] border-r border-[rgba(255,255,255,0.08)] transition-all duration-300 z-30 select-none',
        collapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Brand Header */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-[rgba(255,255,255,0.08)]">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-orange-500/10 border border-orange-500/30 text-orange-400 shrink-0 shadow-[0_0_12px_rgba(249,115,22,0.2)]">
            <ShieldCheck className="w-6 h-6" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="font-bold tracking-wider text-white text-base leading-tight">
                SENTINEL
              </span>
              <span className="text-[11px] font-mono tracking-widest text-[#9CA3AF] uppercase">
                AI Border Watch
              </span>
            </div>
          )}
        </div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-lg text-[#9CA3AF] hover:text-white hover:bg-white/5 transition-colors"
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all group',
                  isActive
                    ? 'bg-orange-500/15 text-orange-400 border border-orange-500/40 shadow-[0_0_14px_rgba(249,115,22,0.15)] font-semibold'
                    : 'text-[#9CA3AF] hover:text-white hover:bg-white/5 border border-transparent'
                )
              }
              title={collapsed ? item.name : undefined}
            >
              <Icon className="w-5 h-5 shrink-0 transition-transform group-hover:scale-105" />
              {!collapsed && <span className="truncate">{item.name}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer Status */}
      <div className="p-3.5 border-t border-[rgba(255,255,255,0.08)] bg-[#07090C]">
        <div className="flex items-center gap-3 px-2 py-1.5 rounded-lg">
          <span className="pulse-dot-green shrink-0" />
          {!collapsed && (
            <div className="flex flex-col">
              <span className="text-xs font-semibold text-white">System Online</span>
              <span className="text-[10px] font-mono text-[#9CA3AF]">YOLOv8s · Ready</span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

