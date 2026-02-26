// File: MediGraph/src/components/Repurposing/EvidencePanel.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Repurposing/EvidencePanel.jsx

/**
 * Evidence Panel Component
 * Displays the full evidence chain for a repurposing candidate
 * Traces pathway through Firestore data without any recomputation
 */

import React from 'react';

const EvidencePanel = ({
    candidate,
    getInteractionById,
    getDrugById,
    getTargetById,
    onClose
}) => {
    const { drug, target, proposedDisease, originalDisease, pathway, rationale } = candidate;

    // Build evidence chain from pathway
    const evidenceChain = (pathway || []).map(interactionId => {
        const interaction = getInteractionById(interactionId);
        if (!interaction) {
            return { id: interactionId, available: false };
        }

        const interactionDrug = getDrugById(interaction.drug_id);
        const interactionTarget = getTargetById(interaction.target_id);

        return {
            id: interactionId,
            available: true,
            drug: interactionDrug,
            target: interactionTarget,
            interaction
        };
    });

    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                right: 0,
                width: '500px',
                height: '100vh',
                background: 'var(--color-bg-secondary)',
                borderLeft: '1px solid var(--color-border)',
                padding: 'var(--spacing-lg)',
                overflowY: 'auto',
                zIndex: 100,
                animation: 'slideIn 0.3s ease-out'
            }}
        >
            <style>{`
        @keyframes slideIn {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>

            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-xl)' }}>
                <h3>Evidence Chain</h3>
                <button className="btn btn-ghost" onClick={onClose}>✕</button>
            </div>

            {/* Summary */}
            <div
                className="glass-card"
                style={{ marginBottom: 'var(--spacing-lg)' }}
            >
                <h4 style={{ marginBottom: 'var(--spacing-md)', color: 'var(--color-primary)' }}>
                    Repurposing Hypothesis
                </h4>
                <p style={{ lineHeight: 1.8 }}>
                    <span className="badge badge-drug" style={{ marginRight: 'var(--spacing-xs)' }}>
                        {drug?.drug_name || 'Drug'}
                    </span>
                    may be repurposed for
                    <span className="badge badge-disease" style={{ margin: '0 var(--spacing-xs)' }}>
                        {proposedDisease?.disease_name || 'Disease'}
                    </span>
                </p>
                {originalDisease && (
                    <p className="text-muted" style={{ fontSize: '0.875rem', marginTop: 'var(--spacing-sm)' }}>
                        Original indication: {originalDisease.disease_name}
                    </p>
                )}
            </div>

            {/* Evidence Steps */}
            <div style={{ marginBottom: 'var(--spacing-lg)' }}>
                <h4 style={{ marginBottom: 'var(--spacing-md)' }}>Pathway Evidence</h4>

                {/* Step 1: Drug-Target Interaction */}
                <div
                    style={{
                        position: 'relative',
                        paddingLeft: 'var(--spacing-xl)',
                        marginBottom: 'var(--spacing-lg)'
                    }}
                >
                    {/* Timeline dot */}
                    <div
                        style={{
                            position: 'absolute',
                            left: 0,
                            top: '4px',
                            width: '12px',
                            height: '12px',
                            background: 'var(--color-drug)',
                            borderRadius: '50%',
                            boxShadow: '0 0 10px var(--color-drug)'
                        }}
                    />
                    {/* Timeline line */}
                    <div
                        style={{
                            position: 'absolute',
                            left: '5px',
                            top: '16px',
                            width: '2px',
                            height: 'calc(100% + var(--spacing-lg) - 16px)',
                            background: 'var(--color-border)'
                        }}
                    />

                    <div className="card">
                        <p className="text-muted" style={{ fontSize: '0.75rem', marginBottom: 'var(--spacing-xs)' }}>
                            Step 1: Predicted Interaction
                        </p>
                        <p style={{ marginBottom: 'var(--spacing-sm)' }}>
                            <strong>{drug?.drug_name}</strong> is predicted to interact with{' '}
                            <strong>{target?.target_name}</strong>
                        </p>
                        {evidenceChain.length > 0 && evidenceChain[0].available && (
                            <div style={{ fontSize: '0.875rem' }}>
                                <p>
                                    • Confidence: {Math.round((evidenceChain[0].interaction?.confidence_score || 0) * 100)}%
                                </p>
                                <p>
                                    • Model: {evidenceChain[0].interaction?.model_used || 'Unknown'}
                                </p>
                                <p>
                                    • Prediction Score: {Math.round((evidenceChain[0].interaction?.prediction_score || 0) * 100)}%
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Step 2: Target-Disease Association */}
                <div
                    style={{
                        position: 'relative',
                        paddingLeft: 'var(--spacing-xl)',
                        marginBottom: 'var(--spacing-lg)'
                    }}
                >
                    {/* Timeline dot */}
                    <div
                        style={{
                            position: 'absolute',
                            left: 0,
                            top: '4px',
                            width: '12px',
                            height: '12px',
                            background: 'var(--color-target)',
                            borderRadius: '50%',
                            boxShadow: '0 0 10px var(--color-target)'
                        }}
                    />
                    {/* Timeline line */}
                    <div
                        style={{
                            position: 'absolute',
                            left: '5px',
                            top: '16px',
                            width: '2px',
                            height: 'calc(100% + var(--spacing-lg) - 16px)',
                            background: 'var(--color-border)'
                        }}
                    />

                    <div className="card">
                        <p className="text-muted" style={{ fontSize: '0.75rem', marginBottom: 'var(--spacing-xs)' }}>
                            Step 2: Target-Disease Link
                        </p>
                        <p>
                            <strong>{target?.target_name}</strong> is associated with{' '}
                            <strong>{proposedDisease?.disease_name}</strong>
                        </p>
                        {target?.gene_name && (
                            <p className="text-muted" style={{ fontSize: '0.875rem', marginTop: 'var(--spacing-xs)' }}>
                                Gene: {target.gene_name}
                            </p>
                        )}
                    </div>
                </div>

                {/* Step 3: Conclusion */}
                <div
                    style={{
                        position: 'relative',
                        paddingLeft: 'var(--spacing-xl)'
                    }}
                >
                    {/* Timeline dot */}
                    <div
                        style={{
                            position: 'absolute',
                            left: 0,
                            top: '4px',
                            width: '12px',
                            height: '12px',
                            background: 'var(--color-disease)',
                            borderRadius: '50%',
                            boxShadow: '0 0 10px var(--color-disease)'
                        }}
                    />

                    <div className="card" style={{ background: 'rgba(139, 92, 246, 0.1)', border: '1px solid var(--color-disease)' }}>
                        <p className="text-muted" style={{ fontSize: '0.75rem', marginBottom: 'var(--spacing-xs)' }}>
                            Conclusion
                        </p>
                        <p>
                            Repurposing candidate identified with score:{' '}
                            <strong style={{ color: 'var(--color-primary)' }}>
                                {Math.round((candidate.repurposing_score || 0) * 100)}%
                            </strong>
                        </p>
                    </div>
                </div>
            </div>

            {/* Rationale */}
            {rationale && (
                <div className="card" style={{ marginBottom: 'var(--spacing-lg)' }}>
                    <h4 style={{ marginBottom: 'var(--spacing-sm)' }}>Rationale</h4>
                    <p className="text-secondary">{rationale}</p>
                </div>
            )}

            {/* Traceability Note */}
            <p className="text-muted" style={{ fontSize: '0.75rem', textAlign: 'center' }}>
                All evidence is derived from Firestore data without recomputation.
                <br />
                Pathway IDs: {(pathway || []).join(', ') || 'N/A'}
            </p>
        </div>
    );
};

export default EvidencePanel;
