// File: MediGraph/src/components/Graph/GraphFilters.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/GraphFilters.jsx

/**
 * Graph Filters Component
 * Provides filter controls for the graph visualization
 */

import React from 'react';

const GraphFilters = ({ filters, onUpdateFilters, onReset, onSaveGraph, drugsList, diseasesList }) => {
    const handleAddDisease = (e) => {
        const id = e.target.value;
        if (id && !filters.selectedDiseaseIds.includes(id)) {
            onUpdateFilters({ selectedDiseaseIds: [...filters.selectedDiseaseIds, id] });
        }
        e.target.value = ""; // reset select
    };

    const handleRemoveDisease = (id) => {
        onUpdateFilters({ selectedDiseaseIds: filters.selectedDiseaseIds.filter(dId => dId !== id) });
    };

    const handleAddDrug = (e) => {
        const id = e.target.value;
        if (id && !filters.selectedDrugIds.includes(id)) {
            onUpdateFilters({ selectedDrugIds: [...filters.selectedDrugIds, id] });
        }
        e.target.value = ""; // reset select
    };

    const handleRemoveDrug = (id) => {
        onUpdateFilters({ selectedDrugIds: filters.selectedDrugIds.filter(dId => dId !== id) });
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
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

            {/* Advanced Filters */}
            <div style={{ minWidth: '150px' }}>
                <label className="form-label" style={{ marginBottom: 'var(--spacing-xs)', display: 'block' }}>
                    Filter by Disease(s)
                </label>
                <select
                    className="form-input"
                    defaultValue=""
                    onChange={handleAddDisease}
                    style={{ padding: 'var(--spacing-xs) var(--spacing-sm)' }}
                >
                    <option value="" disabled>Add a Disease...</option>
                    {diseasesList && diseasesList.filter(d => !filters.selectedDiseaseIds.includes(d.id)).map(d => (
                        <option key={d.id} value={d.id}>{d.disease_name}</option>
                    ))}
                </select>
                
                {/* Selected Disease Badges */}
                {filters.selectedDiseaseIds.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: 'var(--spacing-sm)' }}>
                        {filters.selectedDiseaseIds.map(id => {
                            const name = diseasesList?.find(d => d.id === id)?.disease_name || id;
                            return (
                                <span key={id} className="badge badge-target" style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }} onClick={() => handleRemoveDisease(id)}>
                                    {name} <span style={{ fontSize: '0.7rem' }}>✕</span>
                                </span>
                            );
                        })}
                    </div>
                )}
            </div>

            <div style={{ minWidth: '150px' }}>
                <label className="form-label" style={{ marginBottom: 'var(--spacing-xs)', display: 'block' }}>
                    Filter by Other Drug(s)
                </label>
                <select
                    className="form-input"
                    defaultValue=""
                    onChange={handleAddDrug}
                    style={{ padding: 'var(--spacing-xs) var(--spacing-sm)' }}
                >
                    <option value="" disabled>Add a Drug...</option>
                    {drugsList && drugsList.filter(d => !filters.selectedDrugIds.includes(d.id)).map(d => (
                        <option key={d.id} value={d.id}>{d.drug_name}</option>
                    ))}
                </select>

                {/* Selected Drug Badges */}
                {filters.selectedDrugIds.length > 0 && (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: 'var(--spacing-sm)' }}>
                        {filters.selectedDrugIds.map(id => {
                            const name = drugsList?.find(d => d.id === id)?.drug_name || id;
                            return (
                                <span key={id} className="badge" style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', backgroundColor: 'var(--color-bg-tertiary)' }} onClick={() => handleRemoveDrug(id)}>
                                    {name} <span style={{ fontSize: '0.7rem' }}>✕</span>
                                </span>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-sm)' }}>
                <button className="btn btn-primary" onClick={onSaveGraph} style={{ flex: 1, display: 'flex', justifyContent: 'center', gap: '8px' }}>
                    <span>💾</span> Save Graph
                </button>
                <button className="btn btn-secondary" onClick={onReset} style={{ flex: 1 }}>
                    Reset
                </button>
            </div>
        </div>
    );
};

export default GraphFilters;
