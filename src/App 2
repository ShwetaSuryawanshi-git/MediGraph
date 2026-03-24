// File: MediGraph/src/App.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/App.jsx

/**
 * Main Application Component
 * Root component that handles routing and global providers
 */

import React, { useState } from 'react';
import { AuthProvider, useAuth } from './hooks/useAuth';
import { DataProvider } from './hooks/useFirestoreData';
import AuthPage from './components/Auth/AuthPage';
import Navbar from './components/Layout/Navbar';
import Dashboard from './components/Dashboard/Dashboard';
import GraphView from './components/Graph/GraphView';
import RepurposingView from './components/Repurposing/RepurposingView';
import SavedGraphsView from './components/SavedGraphs/SavedGraphsView';
import './styles/index.css';

/**
 * Main App Content - Handles view routing based on auth state
 */
const AppContent = () => {
    const { isAuthenticated, loading } = useAuth();
    const [currentView, setCurrentView] = useState('dashboard');

    // Show loading spinner while checking auth
    if (loading) {
        return (
            <div
                style={{
                    minHeight: '100vh',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}
            >
                <div className="loading-spinner"></div>
            </div>
        );
    }

    // Show auth page if not logged in
    if (!isAuthenticated) {
        return <AuthPage />;
    }

    // Render main app with navigation
    const renderView = () => {
        switch (currentView) {
            case 'dashboard':
                return <Dashboard />;
            case 'graph':
                return <GraphView />;
            case 'repurposing':
                return <RepurposingView />;
            case 'saved':
                return <SavedGraphsView />;
            default:
                return <Dashboard />;
        }
    };

    return (
        <DataProvider>
            <div style={{ minHeight: '100vh' }}>
                <Navbar currentView={currentView} onNavigate={setCurrentView} />
                <main>{renderView()}</main>
            </div>
        </DataProvider>
    );
};

/**
 * App Component - Wraps everything in AuthProvider
 */
const App = () => {
    return (
        <AuthProvider>
            <AppContent />
        </AuthProvider>
    );
};

export default App;
