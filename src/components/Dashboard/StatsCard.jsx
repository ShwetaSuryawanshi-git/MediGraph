// File: MediGraph/src/components/Dashboard/StatsCard.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Dashboard/StatsCard.jsx

/**
 * Stats Card Component
 * Displays a single statistic with icon and color
 */

import React, { useState } from 'react';

const StatsCard = ({ icon, label, value, color, loading, inference }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div 
            className="glass-card" 
            style={{ 
                padding: 'var(--spacing-lg)',
                cursor: inference ? 'pointer' : 'default',
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center'
            }}
            onClick={() => inference && setIsExpanded(!isExpanded)}
        >
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)' }}>
                <div
                    style={{
                        width: '48px',
                        height: '48px',
                        borderRadius: 'var(--radius-lg)',
                        background: `${color}20`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '1.5rem'
                    }}
                >
                    {icon}
                </div>
                <div>
                    <p className="text-secondary" style={{ fontSize: '0.875rem', marginBottom: '2px' }}>
                        {label}
                    </p>
                    {loading ? (
                        <div
                            className="skeleton"
                            style={{ width: '60px', height: '28px' }}
                        />
                    ) : (
                        <p
                            style={{
                                fontSize: '1.75rem',
                                fontWeight: 600,
                                color: color
                            }}
                        >
                            {value.toLocaleString()}
                        </p>
                    )}
                </div>
            </div>

            {/* Inference Expansion Animation */}
            {inference && (
                <div 
                    style={{
                        maxHeight: isExpanded ? '1000px' : '0px',
                        opacity: isExpanded ? 1 : 0,
                        overflow: 'hidden',
                        transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                        marginTop: isExpanded ? 'var(--spacing-md)' : '0px'
                    }}
                >
                    <div style={{
                        paddingTop: 'var(--spacing-sm)',
                        borderTop: `1px solid ${color}40`,
                        color: 'var(--color-text-secondary)',
                        fontSize: '0.85rem',
                        lineHeight: '1.5'
                    }}>
                        <strong style={{ color: color, display: 'block', marginBottom: '2px' }}>AI Inference Guide:</strong>
                        {inference}
                    </div>
                </div>
            )}
        </div>
    );
};

export default StatsCard;
