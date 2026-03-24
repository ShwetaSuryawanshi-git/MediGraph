// File: MediGraph/src/components/SavedGraphs/SavedGraphsView.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/SavedGraphs/SavedGraphsView.jsx

/**
 * Saved Graphs View Component
 * Displays user's saved graph configurations
 */

import React from 'react';
import { useFirestoreData } from '../../hooks/useFirestoreData';
import { deleteGraphMetadata } from '../../services/firestoreService';

const SavedGraphsView = ({ onViewGraph }) => {
    const { graphsMetadata, loadingState } = useFirestoreData();

    const handleDelete = async (graphId) => {
        if (window.confirm('Are you sure you want to delete this saved graph?')) {
            try {
                await deleteGraphMetadata(graphId);
            } catch (error) {
                console.error('Failed to delete graph:', error);
            }
        }
    };

    const formatDate = (timestamp) => {
        if (!timestamp) return 'Unknown';
        const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    if (loadingState.graphsMetadata === 'loading') {
        return (
            <div className="container" style={{ padding: 'var(--spacing-xl)' }}>
                <div style={{ display: 'flex', justifyContent: 'center', padding: 'var(--spacing-2xl)' }}>
                    <div className="loading-spinner"></div>
                </div>
            </div>
        );
    }

    const savedGraphs = graphsMetadata.allIds.map(id => graphsMetadata.byId[id]);

    return (
        <div className="container animate-fade-in" style={{ padding: 'var(--spacing-xl)' }}>
            <h2 style={{ marginBottom: 'var(--spacing-lg)' }}>Saved Graphs</h2>

            {savedGraphs.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">💾</div>
                    <h3>No Saved Graphs</h3>
                    <p className="text-secondary">
                        Save graphs from the Graph view to access them here.
                    </p>
                </div>
            ) : (
                <div
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                        gap: 'var(--spacing-lg)'
                    }}
                >
                    {savedGraphs.map((graph) => (
                        <div key={graph.id} style={{
                            backgroundColor: 'var(--color-bg-secondary)',
                            border: '1px solid var(--color-border)',
                            borderRadius: 'var(--radius-md)',
                            padding: 'var(--spacing-lg)',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                            display: 'flex',
                            flexDirection: 'column'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-md)', borderBottom: '1px solid var(--color-border)', paddingBottom: 'var(--spacing-sm)' }}>
                                <h4 style={{ margin: 0, color: 'var(--color-text-primary)' }}>{graph.graph_name || 'Unnamed Graph'}</h4>
                                <div style={{ display: 'flex', gap: '8px' }}>
                                    {onViewGraph && (
                                        <button
                                            className="btn btn-primary"
                                            onClick={() => onViewGraph(graph)}
                                            style={{ padding: '4px 12px', fontSize: '0.85rem' }}
                                            title="View Graph"
                                        >
                                            👁️ View
                                        </button>
                                    )}
                                    <button
                                        className="btn btn-outline"
                                        onClick={() => handleDelete(graph.id)}
                                        style={{ padding: '4px 12px', borderColor: 'var(--color-error)', color: 'var(--color-error)' }}
                                        title="Delete Graph"
                                    >
                                        🗑️ Delete
                                    </button>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: 'var(--spacing-xl)', marginBottom: 'var(--spacing-lg)' }}>
                                <div>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Nodes</p>
                                    <p style={{ fontWeight: 'bold', fontSize: '1.2rem', margin: 0, color: 'var(--color-text-primary)' }}>{graph.node_count || 0}</p>
                                </div>
                                <div>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Edges</p>
                                    <p style={{ fontWeight: 'bold', fontSize: '1.2rem', margin: 0, color: 'var(--color-text-primary)' }}>{graph.edge_count || 0}</p>
                                </div>
                            </div>

                            <p style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', marginBottom: 'var(--spacing-sm)' }}>
                                <strong>Created:</strong> {formatDate(graph.created_at)}
                            </p>

                            {graph.filters_applied && Object.keys(graph.filters_applied).length > 0 && (
                                <div style={{ marginTop: 'auto', paddingTop: 'var(--spacing-sm)' }}>
                                    <p style={{ fontSize: '0.85rem', fontWeight: 600, margin: '0 0 8px 0', color: 'var(--color-text-primary)' }}>Filters Applied</p>
                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--spacing-xs)', marginTop: 'var(--spacing-xs)' }}>
                                        {Object.entries(graph.filters_applied).map(([key, value]) => (
                                            <span key={key} className="badge badge-target">
                                                {key}: {String(value)}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default SavedGraphsView;
