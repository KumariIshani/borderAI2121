import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const AppLayout: React.FC = () => {
  const location = useLocation();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0A0D10] text-[#E5E7EB]">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 h-screen overflow-hidden bg-grid-pattern relative">
        {/* Subtle radial ambient glow */}
        <div className="pointer-events-none absolute top-0 right-1/4 w-[600px] h-[300px] bg-orange-500/5 rounded-full blur-[140px] z-0" />
        <div className="pointer-events-none absolute bottom-0 left-1/3 w-[500px] h-[300px] bg-cyan-500/5 rounded-full blur-[140px] z-0" />

        <Header />

        <main className="flex-1 overflow-y-auto p-6 z-10">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              className="max-w-[1600px] mx-auto"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
};

