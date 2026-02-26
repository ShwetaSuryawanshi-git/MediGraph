// File: MediGraph/src/components/Repurposing/RepurposingView.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Repurposing/RepurposingView.jsx

/**
 * Repurposing View Component
 * Displays drug repurposing candidates with evidence traceability
 */

import React, { useState } from 'react';
import { useFirestoreData } from '../../hooks/useFirestoreData';
import RepurposingCard from './RepurposingCard';
import EvidencePanel from './EvidencePanel';

const RepurposingView = () => {
    const {
        repurposingCandidates,
        drugs,
        targets,
        diseases,
        interactions,
        loadingState,
        getDrugById,
        getTargetById,
        getDiseaseById,
        getInteractionById
    } = useFirestoreData();

    const [statusFilter, setStatusFilter] = useState('all');
    const [selectedCandidate, setSelectedCandidate] = useState(null);

    // Filter candidates by status
    const filteredCandidates = repurposingCandidates.allIds
        .map(id => repurposingCandidates.byId[id])
        .filter(candidate => {
            if (statusFilter === 'all') return true;
            return candidate.status === statusFilter;
        })
        .sort((a, b) => (b.repurposing_score || 0) - (a.repurposing_score || 0));

    // Handle candidate click
    const handleCandidateClick = (candidate) => {
        setSelectedCandidate(candidate);
    };

    // Close evidence panel
    const handleCloseEvidence = () => {
        setSelectedCandidate(null);
    };

    // Get enriched candidate data
    const getEnrichedCandidate = (candidate) => {
        return {
            ...candidate,
            drug: getDrugById(candidate.drug_id),
            target: getTargetById(candidate.target_id),
            proposedDisease: getDiseaseById(candidate.proposed_disease_id),
            originalDisease: candidate.original_disease_id
                ? getDiseaseById(candidate.original_disease_id)
                : null
        };
    };

    if (loadingState.repurposingCandidates === 'loading') {
        return (
            <div className="container" style={{ padding: 'var(--spacing-xl)' }}>
                <div style={{ display: 'flex', justifyContent: 'center', padding: 'var(--spacing-2xl)' }}>
                    <div className="loading-spinner"></div>
                </div>
            </div>
        );
    }

    return (
        <div className="container animate-fade-in" style={{ padding: 'var(--spacing-xl)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-lg)' }}>
                <div>
                    <h2 style={{ marginBottom: 'var(--spacing-xs)' }}>Drug Repurposing Candidates</h2>
                    <p className="text-secondary">
                        {filteredCandidates.length} candidates found
                    </p>
                </div>
            </div>

            {/* Status Filter */}
            <div
                className="glass-card"
                style={{
                    padding: 'var(--spacing-md)',
                    marginBottom: 'var(--spacing-lg)',
                    display: 'flex',
                    gap: 'var(--spacing-sm)',
                    flexWrap: 'wrap'
                }}
            >
                {['all', 'pending', 'reviewed', 'validated', 'rejected'].map((status) => (
                    <button
                        key={status}
                        className={`btn ${statusFilter === status ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setStatusFilter(status)}
                    >
                        {status.charAt(0).toUpperCase() + status.slice(1)}
                    </button>
                ))}
            </div>

            {/* Candidates List */}
            {filteredCandidates.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">💊</div>
                    <h3>No Repurposing Candidates</h3>
                    <p className="text-secondary">
                        {statusFilter === 'all'
                            ? 'Run the ML pipeline to identify repurposing opportunities.'
                            : `No candidates with status "${statusFilter}".`}
                    </p>
                </div>
            ) : (
                <div
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
                        gap: 'var(--spacing-lg)'
                    }}
                >
                    {filteredCandidates.map((candidate) => (
                        <RepurposingCard
                            key={candidate.id}
                            candidate={getEnrichedCandidate(candidate)}
                            onClick={() => handleCandidateClick(candidate)}
                        />
                    ))}
                </div>
            )}

            {/* Evidence Panel */}
            {selectedCandidate && (
                <EvidencePanel
                    candidate={getEnrichedCandidate(selectedCandidate)}
                    getInteractionById={getInteractionById}
                    getDrugById={getDrugById}
                    getTargetById={getTargetById}
                    onClose={handleCloseEvidence}
                />
            )}
        </div>
    );
};

export default RepurposingView;
