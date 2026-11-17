import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';
import { AppLayout } from '@/components/layout/AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { LiveMonitorPage } from '@/pages/LiveMonitorPage';
import { UploadAnalyzePage } from '@/pages/UploadAnalyzePage';
import { AlertsPage } from '@/pages/AlertsPage';
import { DatasetPage } from '@/pages/DatasetPage';
import { ModelPerformancePage } from '@/pages/ModelPerformancePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<AppLayout />}>
            <Route index element={<DashboardPage />} />
            <Route path="monitor" element={<LiveMonitorPage />} />
            <Route path="analyze" element={<UploadAnalyzePage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="dataset" element={<DatasetPage />} />
            <Route path="model" element={<ModelPerformancePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="bottom-right" theme="dark" richColors />
    </QueryClientProvider>
  );
};

export default App;

