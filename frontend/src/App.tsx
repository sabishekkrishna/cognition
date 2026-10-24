import React, { useState, useEffect } from 'react';
import { Navigation } from './components/Navigation';
import { Header } from './components/Header';
import { OverviewPage } from './pages/OverviewPage';
import { ImageLabPage } from './pages/ImageLabPage';
import { AudioLabPage } from './pages/AudioLabPage';
import { LiveMonitorPage } from './pages/LiveMonitorPage';
import { LLMConsolePage } from './pages/LLMConsolePage';
import { BaselineManagerPage } from './pages/BaselineManagerPage';
import { DatasetsModelsPage } from './pages/DatasetsModelsPage';
import { EventHistoryPage } from './pages/EventHistoryPage';
import { SettingsPage } from './pages/SettingsPage';
import { SystemOverallStatus } from './types';
import { api } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [systemStatus, setSystemStatus] = useState<SystemOverallStatus | null>(null);
  const [activeProviderName, setActiveProviderName] = useState<string>('Ollama / Local');

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      const statusData = await api.getStatus();
      setSystemStatus(statusData);

      const provData = await api.getLLMProviders();
      const current = provData.providers?.find((p: any) => p.id === provData.active_provider);
      if (current) {
        setActiveProviderName(current.name.split(' ')[0]);
      }
    } catch (err) {
      // Backend may be starting or offline
    }
  };

  const renderActivePage = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <OverviewPage
            status={systemStatus}
            onNavigate={(tab) => setActiveTab(tab)}
            onRefresh={fetchStatus}
          />
        );
      case 'image-lab':
        return <ImageLabPage />;
      case 'audio-lab':
        return <AudioLabPage />;
      case 'live-monitor':
        return <LiveMonitorPage />;
      case 'llm-console':
        return (
          <LLMConsolePage
            currentInputRiskScore={
              systemStatus ? (100 - systemStatus.overall_reliability_score) : 0
            }
            currentInputSubsystem={
              systemStatus?.image_state === 'warning' ? 'image' : systemStatus?.audio_state === 'warning' ? 'audio' : 'sensor'
            }
          />
        );
      case 'baselines':
        return <BaselineManagerPage />;
      case 'datasets':
        return <DatasetsModelsPage />;
      case 'events':
        return <EventHistoryPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return (
          <OverviewPage
            status={systemStatus}
            onNavigate={(tab) => setActiveTab(tab)}
            onRefresh={fetchStatus}
          />
        );
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0b0f19] text-slate-100 font-sans selection:bg-blue-600/30">
      {/* Sidebar Navigation */}
      <Navigation
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        overallState={systemStatus?.overall_state || 'normal'}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Header */}
        <Header
          status={systemStatus}
          activeProviderName={activeProviderName}
        />

        {/* Scrollable Viewport */}
        <main className="flex-1 overflow-y-auto p-6 tech-grid">
          <div className="max-w-7xl mx-auto pb-12">
            {renderActivePage()}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
