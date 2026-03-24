// File: MediGraph/src/components/SavedGraphs/SavedGraphsView.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/SavedGraphs/SavedGraphsView.jsx

/**
 * Saved Graphs View Component
 * Displays user's saved graph configurations
 */

import React from 'react';
import { useFirestoreData } from '../../hooks/useFirestoreData';
import { deleteGraphMetadata } from '../../services/firestoreService';

const SavedGraphsView = () => {
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
                        <div key={graph.id} className="glass-card">
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 'var(--spacing-md)' }}>
                                <h4>{graph.graph_name || 'Unnamed Graph'}</h4>
                                <button
                                    className="btn btn-ghost"
                                    onClick={() => handleDelete(graph.id)}
                                    style={{ color: 'var(--color-error)', padding: 'var(--spacing-xs)' }}
                                >
                                    🗑️
                                </button>
                            </div>

                            <div style={{ display: 'flex', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-md)' }}>
                                <div>
                                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>Nodes</p>
                                    <p style={{ fontWeight: 600 }}>{graph.node_count || 0}</p>
                                </div>
                                <div>
                                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>Edges</p>
                                    <p style={{ fontWeight: 600 }}>{graph.edge_count || 0}</p>
                                </div>
                            </div>

                            <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                                Created: {formatDate(graph.created_at)}
                            </p>

                            {graph.filters_applied && Object.keys(graph.filters_applied).length > 0 && (
                                <div style={{ marginTop: 'var(--spacing-sm)' }}>
                                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>Filters Applied</p>
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
