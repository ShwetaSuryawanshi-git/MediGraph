// File: MediGraph/src/components/Layout/Navbar.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Layout/Navbar.jsx

/**
 * Navigation Bar Component
 * Provides app-wide navigation and user actions
 */

import React from 'react';
import { useAuth } from '../../hooks/useAuth';

const Navbar = ({ currentView, onNavigate }) => {
    const { user, logout } = useAuth();

    const navItems = [
        { id: 'dashboard', label: 'Dashboard', icon: '📊' },
        { id: 'graph', label: 'Graph', icon: '🔗' },
        { id: 'repurposing', label: 'Repurposing', icon: '💊' },
        { id: 'saved', label: 'Saved Graphs', icon: '💾' }
    ];

    const handleLogout = async () => {
        try {
            await logout();
        } catch (err) {
            console.error('Logout failed:', err);
        }
    };

    return (
        <nav
            style={{
                position: 'sticky',
                top: 0,
                zIndex: 50,
                background: 'rgba(15, 15, 35, 0.9)',
                backdropFilter: 'blur(12px)',
                borderBottom: '1px solid var(--color-border)',
                padding: 'var(--spacing-md) var(--spacing-lg)'
            }}
        >
            <div
                className="container"
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: 'var(--spacing-md)'
                }}
            >
                {/* Left Side: Logo and Links */}
                <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--spacing-xl)', flex: 1 }}>
                    
                    {/* Logo */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                        <span style={{ fontSize: '1.5rem' }}>🧬</span>
                        <span
                            style={{
                                fontSize: '1.25rem',
                                fontWeight: 600,
                                background: 'linear-gradient(135deg, var(--color-primary), var(--color-disease))',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                whiteSpace: 'nowrap'
                            }}
                        >
                            MediGraph
                        </span>
                    </div>

                    {/* Navigation Links */}
                    <div className="overflow-x-auto" style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
                        {navItems.map((item) => (
                            <button
                                key={item.id}
                                onClick={() => onNavigate(item.id)}
                                className={`btn ${currentView === item.id ? 'btn-primary' : 'btn-ghost'}`}
                                style={{ fontSize: '0.875rem', flexShrink: 0 }}
                            >
                                <span>{item.icon}</span>
                                <span>{item.label}</span>
                            </button>
                        ))}
                    </div>
                </div>

                {/* Right Side: User Actions */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                    <span className="text-secondary" style={{ fontSize: '0.875rem', display: window.innerWidth > 768 ? 'block' : 'none' }}>
                        {user?.email}
                    </span>
                    <button
                        onClick={handleLogout}
                        className="btn btn-secondary"
                        style={{ fontSize: '0.875rem', padding: '0.25rem 0.75rem' }}
                    >
                        Logout
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
