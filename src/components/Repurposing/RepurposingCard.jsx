// File: MediGraph/src/components/Repurposing/RepurposingCard.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Repurposing/RepurposingCard.jsx

/**
 * Repurposing Card Component
 * Displays a single repurposing candidate with key information
 */

import React from 'react';

const RepurposingCard = ({ candidate, onClick }) => {
    const { drug, target, proposedDisease, repurposing_score, status } = candidate;

    const getStatusBadgeClass = (status) => {
        switch (status) {
            case 'validated': return 'badge-validated';
            case 'rejected': return 'badge-rejected';
            case 'reviewed': return 'badge-target';
            default: return 'badge-pending';
        }
    };

    const scorePercent = Math.round((repurposing_score || 0) * 100);

    return (
        <div
            className="glass-card"
            onClick={onClick}
            style={{
                cursor: 'pointer',
                transition: 'all var(--transition-base)'
            }}
        >
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 'var(--spacing-md)' }}>
                <div>
                    <h4 style={{ marginBottom: 'var(--spacing-xs)' }}>
                        {drug?.drug_name || 'Unknown Drug'}
                    </h4>
                    <p className="text-secondary" style={{ fontSize: '0.875rem' }}>
                        → {proposedDisease?.disease_name || 'Unknown Disease'}
                    </p>
                </div>
                <span className={`badge ${getStatusBadgeClass(status)}`}>
                    {status}
                </span>
            </div>

            {/* Via Target */}
            <div
                style={{
                    padding: 'var(--spacing-sm)',
                    background: 'rgba(59, 130, 246, 0.1)',
                    borderRadius: 'var(--radius-md)',
                    marginBottom: 'var(--spacing-md)'
                }}
            >
                <p className="text-muted" style={{ fontSize: '0.75rem', marginBottom: '2px' }}>
                    Via Target
                </p>
                <p style={{ fontSize: '0.875rem', color: 'var(--color-target-light)' }}>
                    🎯 {target?.target_name || 'Unknown Target'}
                </p>
            </div>

            {/* Confidence Score */}
            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-xs)' }}>
                    <span className="text-muted" style={{ fontSize: '0.75rem' }}>
                        Repurposing Score
                    </span>
                    <span style={{ fontWeight: 600, color: 'var(--color-primary)' }}>
                        {scorePercent}%
                    </span>
                </div>
                <div
                    style={{
                        width: '100%',
                        height: '6px',
                        background: 'var(--color-bg-tertiary)',
                        borderRadius: 'var(--radius-full)',
                        overflow: 'hidden'
                    }}
                >
                    <div
                        style={{
                            width: `${scorePercent}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, var(--color-primary), var(--color-disease))',
                            borderRadius: 'var(--radius-full)',
                            transition: 'width var(--transition-base)'
                        }}
                    />
                </div>
            </div>

            {/* Original Indication */}
            {drug?.approved_indication && (
                <div style={{ marginTop: 'var(--spacing-md)', paddingTop: 'var(--spacing-md)', borderTop: '1px solid var(--color-border)' }}>
                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                        Originally approved for: {drug.approved_indication}
                    </p>
                </div>
            )}

            {/* Click hint */}
            <p className="text-muted" style={{ fontSize: '0.75rem', marginTop: 'var(--spacing-md)', textAlign: 'center' }}>
                Click to view evidence chain →
            </p>
        </div>
    );
};

export default RepurposingCard;
