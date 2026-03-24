// File: MediGraph/src/components/Graph/GraphFilters.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/GraphFilters.jsx

/**
 * Graph Filters Component
 * Provides filter controls for the graph visualization
 */

import React from 'react';

const GraphFilters = ({ filters, onUpdateFilters, onReset }) => {
    return (
        <div
            className="glass-card"
            style={{
                padding: 'var(--spacing-md)',
                marginBottom: 'var(--spacing-lg)',
                display: 'flex',
                flexWrap: 'wrap',
                gap: 'var(--spacing-lg)',
                alignItems: 'center'
            }}
        >
            {/* Confidence Slider */}
            <div style={{ flex: '1', minWidth: '200px' }}>
                <label className="form-label" style={{ marginBottom: 'var(--spacing-xs)', display: 'block' }}>
                    Min Confidence: {(filters.minConfidence * 100).toFixed(0)}%
                </label>
                <input
                    type="range"
                    min="0"
                    max="100"
                    value={filters.minConfidence * 100}
                    onChange={(e) => onUpdateFilters({ minConfidence: e.target.value / 100 })}
                    style={{ width: '100%' }}
                />
            </div>

            {/* Model Filter */}
            <div style={{ minWidth: '150px' }}>
                <label className="form-label" style={{ marginBottom: 'var(--spacing-xs)', display: 'block' }}>
                    Model
                </label>
                <select
                    className="form-input"
                    value={filters.modelUsed || ''}
                    onChange={(e) => onUpdateFilters({ modelUsed: e.target.value || null })}
                    style={{ padding: 'var(--spacing-xs) var(--spacing-sm)' }}
                >
                    <option value="">All Models</option>
                    <option value="random_forest">Random Forest</option>
                    <option value="logistic_regression">Logistic Regression</option>
                    <option value="svm">SVM</option>
                </select>
            </div>

            {/* Show Diseases Toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
                <input
                    type="checkbox"
                    id="showDiseases"
                    checked={filters.showDiseases}
                    onChange={(e) => onUpdateFilters({ showDiseases: e.target.checked })}
                />
                <label htmlFor="showDiseases" className="form-label" style={{ margin: 0 }}>
                    Show Diseases
                </label>
            </div>

            {/* Reset Button */}
            <button className="btn btn-secondary" onClick={onReset}>
                Reset Filters
            </button>
        </div>
    );
};

export default GraphFilters;
