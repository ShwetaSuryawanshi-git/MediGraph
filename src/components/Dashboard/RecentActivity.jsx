// File: MediGraph/src/components/Dashboard/RecentActivity.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Dashboard/RecentActivity.jsx

/**
 * Recent Activity Component
 * Displays a list of recent items with badges
 */

import React from 'react';

const RecentActivity = ({ title, items, badgeClass, emptyMessage, onItemClick }) => {
    return (
        <div className="glass-card">
            <h4 style={{ marginBottom: 'var(--spacing-md)' }}>{title}</h4>

            {items.length === 0 ? (
                <div className="empty-state" style={{ padding: 'var(--spacing-lg)' }}>
                    <p>{emptyMessage}</p>
                </div>
            ) : (
                <ul style={{ listStyle: 'none' }}>
                    {items.map((item) => (
                        <li
                            key={item.id}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                padding: 'var(--spacing-sm) var(--spacing-xs)',
                                borderBottom: '1px solid var(--color-border)',
                                cursor: onItemClick ? 'pointer' : 'default',
                                borderRadius: 'var(--radius-sm)'
                            }}
                            onClick={() => onItemClick && onItemClick(item.id)}
                            onMouseOver={(e) => onItemClick && (e.currentTarget.style.backgroundColor = 'var(--color-bg-tertiary)')}
                            onMouseOut={(e) => onItemClick && (e.currentTarget.style.backgroundColor = 'transparent')}
                        >
                            <div>
                                <p style={{ fontWeight: 500 }}>{item.label}</p>
                                <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                                    {item.sublabel}
                                </p>
                            </div>
                            <span className={`badge ${badgeClass}`}>{item.badge}</span>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default RecentActivity;
