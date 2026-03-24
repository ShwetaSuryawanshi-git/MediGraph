// File: MediGraph/src/components/Graph/NodeDetail.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/NodeDetail.jsx

/**
 * Node Detail Component
 * Displays detailed information about a selected node
 */

import React from 'react';
import { NODE_TYPES, EDGE_TYPES } from '../../hooks/useGraphData';

const WIKI_SUMMARIES = {
    "Malaria": "Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes.",
    "Tuberculosis": "Tuberculosis (TB) is a contagious infection that usually attacks the lungs. It can also spread to other parts of the body, like the brain and spine.",
    "Dengue": "Dengue fever is a mosquito-borne viral disease occurring in tropical and subtropical areas. Symptoms include high fever, rash, and muscle and joint pain.",
    "Typhoid": "Typhoid is a bacterial infection that can lead to a high fever, diarrhea, and vomiting. It is caused by the bacteria Salmonella typhimurium.",
    "Asthma": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing.",
    "Pain and Fever": "Pain and fever are common symptoms of many medical conditions, including infections, inflammation, and injury. Fever is a temporary increase in your body temperature.",
    "Hypertension": "High blood pressure (hypertension) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems.",
    "Ibuprofen": "Ibuprofen is a medication in the nonsteroidal anti-inflammatory drug (NSAID) class that is used for treating pain, fever, and inflammation.",
    "Amoxicillin": "Amoxicillin is an antibiotic often used for the treatment of a number of bacterial infections. These include middle ear infections, strep throat, pneumonia, skin infections, and urinary tract infections.",
    "Cetirizine": "Cetirizine, sold under the brand name Zyrtec among others, is a second-generation antihistamine used to treat allergic rhinitis, dermatitis, and urticaria.",
    "Salbutamol": "Salbutamol, also known as albuterol, is a medication that opens up the medium and large airways in the lungs. It is used to treat asthma, including asthma attacks.",
    "Losartan": "Losartan is a medication used to treat high blood pressure (hypertension) and diabetic kidney disease, heart failure, and left ventricular enlargement.",
    "Amlodipine": "Amlodipine, sold under the brand name Norvasc among others, is a calcium channel blocker medication used to treat high blood pressure and coronary artery disease.",
    "Isoniazid": "Isoniazid is an antibiotic used for the treatment of tuberculosis. For active tuberculosis it is often used together with rifampicin and pyrazinamide.",
    "Hydroxychloroquine": "Hydroxychloroquine (HCQ), sold under the brand name Plaquenil among others, is a medication used to prevent and treat malaria in areas where malaria remains sensitive to chloroquine.",
    "Ofloxacin": "Ofloxacin is a quinolone antibiotic useful for the treatment of a number of bacterial infections.",
    "Dolo-650 (Paracetamol)": "Paracetamol, also known as acetaminophen, is a medication used to treat fever and mild to moderate pain. Dolo-650 is a popular brand name in India.",
    "Paracetamol": "Paracetamol, also known as acetaminophen, is a medication used to treat fever and mild to moderate pain."
};

const NodeDetail = ({ nodeType, nodeData, edges, nodes, inline, onClose }) => {
    if (!nodeData) return null;

    const getEntityName = () => {
        if (nodeType === NODE_TYPES.DRUG) return nodeData.drug_name;
        if (nodeType === NODE_TYPES.TARGET) return nodeData.target_name;
        if (nodeType === NODE_TYPES.DISEASE) return nodeData.disease_name;
        return "";
    };

    const getSummary = () => {
        const name = getEntityName();
        if (WIKI_SUMMARIES[name]) return WIKI_SUMMARIES[name];
        
        // Fallback templates based on type
        if (nodeType === NODE_TYPES.DRUG && nodeData.approved_indication) {
            return `${name} is a pharmaceutical compound primarily indicated for the treatment of ${nodeData.approved_indication.toLowerCase()}.`;
        }
        if (nodeType === NODE_TYPES.TARGET) {
            return `${name} is a biological target protein or macromolecule that is often involved in signaling pathways or disease mechanisms.`;
        }
        if (nodeType === NODE_TYPES.DISEASE) {
            return `${name} is a medical condition or disease state currently being investigated for novel drug repurposing candidates.`;
        }
        return `Detailed information about ${name} is currently limited in the database.`;
    };

    const renderDrugDetails = () => (
        <>
            <div className="detail-row">
                <span className="detail-label">Drug Name</span>
                <span className="detail-value">{nodeData.drug_name}</span>
            </div>
            <div className="detail-row">
                <span className="detail-label">SMILES</span>
                <span className="detail-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    {nodeData.smiles}
                </span>
            </div>
            {nodeData.drugbank_id && (
                <div className="detail-row">
                    <span className="detail-label">DrugBank ID</span>
                    <span className="detail-value">{nodeData.drugbank_id}</span>
                </div>
            )}
            {nodeData.approved_indication && (
                <div className="detail-row">
                    <span className="detail-label">Approved For</span>
                    <span className="detail-value">{nodeData.approved_indication}</span>
                </div>
            )}
            {nodeData.molecular_weight && (
                <div className="detail-row">
                    <span className="detail-label">Molecular Weight</span>
                    <span className="detail-value">{nodeData.molecular_weight} g/mol</span>
                </div>
            )}
            
            {edges && nodes && nodeType === NODE_TYPES.DRUG && (
                (() => {
                    const drugInteractions = edges.filter(e => e.type === EDGE_TYPES.INTERACTION && e.source === nodeData.id);
                    if (drugInteractions.length === 0) return null;
                    return (
                        <div className="detail-row" style={{ marginTop: 'var(--spacing-md)' }}>
                            <span className="detail-label" style={{ fontWeight: 'bold', color: 'var(--color-primary)', fontSize: '0.85rem' }}>🤖 AI Rationales</span>
                            <div style={{ marginTop: 'var(--spacing-sm)' }}>
                                {drugInteractions.map(interaction => {
                                    const targetNode = nodes.find(n => n.id === interaction.target);
                                    const confPercent = (interaction.weight * 100).toFixed(1);
                                    return (
                                        <div key={interaction.id} style={{ 
                                            background: 'var(--color-bg-tertiary)', 
                                            padding: 'var(--spacing-sm)', 
                                            borderRadius: 'var(--radius-sm)',
                                            marginBottom: 'var(--spacing-sm)',
                                            fontSize: '0.8rem',
                                            borderLeft: '3px solid var(--color-primary)'
                                        }}>
                                            <div style={{ fontWeight: '600', marginBottom: '2px' }}>{targetNode?.label || 'Target'}</div>
                                            <div style={{ color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
                                                Predicted with <strong style={{color: 'var(--color-text-primary)'}}>{confPercent}%</strong> confidence. 
                                                {interaction.data?.interaction_type === 'validated' 
                                                    ? ' This is a known, clinically validated interaction.'
                                                    : ' This is a novel AI prediction based on structural affinity.'}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })()
            )}

            {edges && nodes && nodeType === NODE_TYPES.DRUG && (
                (() => {
                    // Find all targets connected to this drug
                    const drugTargetEdges = edges.filter(e => e.type === EDGE_TYPES.INTERACTION && e.source === nodeData.id);
                    const targetIds = drugTargetEdges.map(e => e.target);
                    
                    // Find all diseases connected to those targets
                    const targetDiseaseEdges = edges.filter(e => e.type === EDGE_TYPES.ASSOCIATION && targetIds.includes(e.source));
                    
                    // Deduplicate diseases
                    const uniqueDiseaseIds = [...new Set(targetDiseaseEdges.map(e => e.target))];
                    const connectedDiseases = uniqueDiseaseIds.map(id => nodes.find(n => n.id === id)).filter(Boolean);

                    if (connectedDiseases.length === 0) return null;
                    return (
                        <div className="detail-row" style={{ marginTop: 'var(--spacing-md)' }}>
                            <span className="detail-label" style={{ fontWeight: 'bold', color: 'var(--color-primary)', fontSize: '0.85rem' }}>🦠 Linked Diseases (via Targets)</span>
                            <div style={{ marginTop: 'var(--spacing-sm)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                {connectedDiseases.map(disease => (
                                    <span key={disease.id} className="badge badge-target" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>
                                        {disease.label}
                                    </span>
                                ))}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '8px', fontStyle: 'italic' }}>
                                {getEntityName()} interacts with targets that are biologically associated with these diseases.
                            </div>
                        </div>
                    );
                })()
            )}
        </>
    );

    const renderTargetDetails = () => (
        <>
            <div className="detail-row">
                <span className="detail-label">Target Name</span>
                <span className="detail-value">{nodeData.target_name}</span>
            </div>
            {nodeData.uniprot_id && (
                <div className="detail-row">
                    <span className="detail-label">UniProt ID</span>
                    <span className="detail-value">{nodeData.uniprot_id}</span>
                </div>
            )}
            {nodeData.gene_name && (
                <div className="detail-row">
                    <span className="detail-label">Gene Name</span>
                    <span className="detail-value">{nodeData.gene_name}</span>
                </div>
            )}
            {nodeData.organism && (
                <div className="detail-row">
                    <span className="detail-label">Organism</span>
                    <span className="detail-value">{nodeData.organism}</span>
                </div>
            )}
            {nodeData.associated_disease_ids?.length > 0 && (
                <div className="detail-row">
                    <span className="detail-label">Linked Diseases</span>
                    <span className="detail-value">{nodeData.associated_disease_ids.length} diseases</span>
                </div>
            )}
        </>
    );

    const renderDiseaseDetails = () => (
        <>
            <div className="detail-row">
                <span className="detail-label">Disease Name</span>
                <span className="detail-value">{nodeData.disease_name}</span>
            </div>
            {nodeData.disease_category && (
                <div className="detail-row">
                    <span className="detail-label">Category</span>
                    <span className="detail-value">{nodeData.disease_category}</span>
                </div>
            )}
            {nodeData.disease_id_external && (
                <div className="detail-row">
                    <span className="detail-label">External ID</span>
                    <span className="detail-value">{nodeData.disease_id_external}</span>
                </div>
            )}
            {nodeData.description && (
                <div className="detail-row">
                    <span className="detail-label">Description</span>
                    <span className="detail-value">{nodeData.description}</span>
                </div>
            )}

            {edges && nodes && nodeType === NODE_TYPES.DISEASE && (
                (() => {
                    // Find all targets connected to this disease
                    const diseaseTargetEdges = edges.filter(e => e.type === EDGE_TYPES.ASSOCIATION && e.target === nodeData.id);
                    const targetIds = diseaseTargetEdges.map(e => e.source);
                    
                    // Find all drugs connected to those targets
                    const targetDrugEdges = edges.filter(e => e.type === EDGE_TYPES.INTERACTION && targetIds.includes(e.target));
                    
                    // Deduplicate drugs
                    const uniqueDrugIds = [...new Set(targetDrugEdges.map(e => e.source))];
                    const connectedDrugs = uniqueDrugIds.map(id => nodes.find(n => n.id === id)).filter(Boolean);

                    if (connectedDrugs.length === 0) return null;
                    return (
                        <div className="detail-row" style={{ marginTop: 'var(--spacing-md)' }}>
                            <span className="detail-label" style={{ fontWeight: 'bold', color: 'var(--color-primary)', fontSize: '0.85rem' }}>💊 Linked Drugs (via Targets)</span>
                            <div style={{ marginTop: 'var(--spacing-sm)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                                {connectedDrugs.map(drug => (
                                    <span key={drug.id} className="badge" style={{ backgroundColor: 'var(--color-bg-tertiary)', color: 'var(--color-text-primary)' }}>
                                        {drug.label}
                                    </span>
                                ))}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '8px', fontStyle: 'italic' }}>
                                These pharmaceutical compounds interact with biological targets associated with {getEntityName()}.
                            </div>
                        </div>
                    );
                })()
            )}
        </>
    );

    const getTitle = () => {
        switch (nodeType) {
            case NODE_TYPES.DRUG: return '💊 Drug Details';
            case NODE_TYPES.TARGET: return '🎯 Target Details';
            case NODE_TYPES.DISEASE: return '🏥 Disease Details';
            default: return 'Details';
        }
    };

    const getBadgeClass = () => {
        switch (nodeType) {
            case NODE_TYPES.DRUG: return 'badge-drug';
            case NODE_TYPES.TARGET: return 'badge-target';
            case NODE_TYPES.DISEASE: return 'badge-disease';
            default: return '';
        }
    };

    return (
        <div
            style={inline ? {
                width: '100%',
                height: '100%',
                animation: 'fadeIn 0.3s ease-out'
            } : {
                position: 'fixed',
                top: 0,
                right: 0,
                width: '400px',
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
        .detail-row {
          display: flex;
          flex-direction: column;
          padding: var(--spacing-sm) 0;
          border-bottom: 1px solid var(--color-border);
        }
        .detail-label {
          font-size: 0.75rem;
          color: var(--color-text-muted);
          margin-bottom: 2px;
        }
        .detail-value {
          font-size: 0.875rem;
          color: var(--color-text-primary);
          word-break: break-word;
        }
      `}</style>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-md)' }}>
                <h3 style={{ fontSize: inline ? '1.5rem' : '1.25rem', margin: 0 }}>{getEntityName()}</h3>
                {!inline && <button className="btn btn-ghost" onClick={onClose}>✕</button>}
            </div>

            <div style={{ marginBottom: 'var(--spacing-md)' }}>
                <span className={`badge ${getBadgeClass()}`} style={{ marginRight: '8px' }}>
                    {nodeType.toUpperCase()}
                </span>
            </div>

            {/* Google-Style Wikipedia Summary */}
            <div style={{ 
                marginBottom: 'var(--spacing-lg)', 
                lineHeight: '1.5',
                color: 'var(--color-text-secondary)',
                fontSize: '0.95rem',
                paddingBottom: 'var(--spacing-md)',
                borderBottom: '1px solid var(--color-border)'
            }}>
                {getSummary()}
            </div>

            {nodeType === NODE_TYPES.DRUG && renderDrugDetails()}
            {nodeType === NODE_TYPES.TARGET && renderTargetDetails()}
            {nodeType === NODE_TYPES.DISEASE && renderDiseaseDetails()}
        </div>
    );
};

export default NodeDetail;
