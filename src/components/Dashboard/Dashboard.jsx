// File: MediGraph/src/components/Dashboard/Dashboard.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Dashboard/Dashboard.jsx

/**
 * Dashboard Component
 * Displays overview statistics and quick access to data
 */

import React, { useState } from 'react';
import { useFirestoreData } from '../../hooks/useFirestoreData';
import StatsCard from './StatsCard';
import RecentActivity from './RecentActivity';
import SearchBar from './SearchBar';

const Dashboard = ({ onNavigateToGraph }) => {
    const {
        drugs,
        targets,
        interactions,
        diseases,
        repurposingCandidates,
        loadingState,
        isLoading
    } = useFirestoreData();

    // Calculate statistics
    const stats = {
        totalDrugs: drugs.allIds.length,
        totalTargets: targets.allIds.length,
        totalInteractions: interactions.allIds.length,
        totalDiseases: diseases.allIds.length,
        pendingRepurposing: repurposingCandidates.allIds.filter(
            id => repurposingCandidates.byId[id]?.status === 'pending'
        ).length,
        validatedRepurposing: repurposingCandidates.allIds.filter(
            id => repurposingCandidates.byId[id]?.status === 'validated'
        ).length
    };

    if (isLoading) {
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
            <h2 style={{ marginBottom: 'var(--spacing-lg)' }}>Dashboard</h2>

            {/* Search Bar */}
            <SearchBar
                drugs={drugs}
                targets={targets}
                diseases={diseases}
                onResultClick={(result) => {
                    console.log('Search result clicked:', result);
                    if (onNavigateToGraph) {
                        onNavigateToGraph(result);
                    }
                }}
            />

            {/* Stats Grid */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 140px), 1fr))',
                    gap: 'var(--spacing-md)',
                    marginBottom: 'var(--spacing-xl)',
                    alignItems: 'start'
                }}
            >
                <StatsCard
                    icon="💊"
                    label="Drugs"
                    value={stats.totalDrugs}
                    color="var(--color-drug)"
                    loading={loadingState.drugs === 'loading'}
                    inference={`The number ${stats.totalDrugs} physically represents the total common pharmaceutical compounds (e.g., Paracetamol, Aspirin) loaded into the MediGraph pipeline specifically for the Indian medical context.`}
                />
                <StatsCard
                    icon="🎯"
                    label="Targets"
                    value={stats.totalTargets}
                    color="var(--color-target)"
                    loading={loadingState.targets === 'loading'}
                    inference={`There are currently ${stats.totalTargets} critical biological receptors (like ABL1 Kinase) that the custom Random Forest model analyzes to predict exact binding affinities in the human body.`}
                />
                <StatsCard
                    icon="🔗"
                    label="Interactions"
                    value={stats.totalInteractions}
                    color="var(--color-primary)"
                    loading={loadingState.interactions === 'loading'}
                    inference={`The AI has formed ${stats.totalInteractions} structural pathways mathematically predicting the bonding confidence between the mapped drugs and targets.`}
                />
                <StatsCard
                    icon="🏥"
                    label="Diseases"
                    value={stats.totalDiseases}
                    color="var(--color-disease)"
                    loading={loadingState.diseases === 'loading'}
                    inference={`There are ${stats.totalDiseases} pathological conditions (including Tuberculosis and Dengue) mathematically isolated linked to the targets, allowing the graph engine to triangulate novel repurposing opportunities.`}
                />
                <StatsCard
                    icon="⏳"
                    label="Pending Review"
                    value={stats.pendingRepurposing}
                    color="var(--color-warning)"
                    loading={loadingState.repurposingCandidates === 'loading'}
                    inference={`These ${stats.pendingRepurposing} counts represent newly discovered drug-repurposing hypotheses automatically generated by the AI's graph inference algorithm that currently require human clinical validation.`}
                />
                <StatsCard
                    icon="✅"
                    label="Validated"
                    value={stats.validatedRepurposing}
                    color="var(--color-success)"
                    loading={loadingState.repurposingCandidates === 'loading'}
                    inference={`This number indicates that ${stats.validatedRepurposing} confirmed interactions have been transferred to this secure counter by researchers verifying the AI's 1st-degree pathway predictions.`}
                />
            </div>

            {/* Recent Activity */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 280px), 1fr))',
                    gap: 'var(--spacing-md)'
                }}
            >
                <RecentActivity
                    title="Recent Drugs"
                    items={drugs.allIds.slice(0, 5).map(id => ({
                        id,
                        label: drugs.byId[id]?.drug_name || 'Unknown',
                        sublabel: drugs.byId[id]?.approved_indication || 'No indication',
                        badge: 'Drug'
                    }))}
                    badgeClass="badge-drug"
                    emptyMessage="No drugs in database yet"
                    onItemClick={(id) => onNavigateToGraph && onNavigateToGraph({ id, type: 'drug', label: drugs.byId[id]?.drug_name })}
                />
                <RecentActivity
                    title="Recent Targets"
                    items={targets.allIds.slice(0, 5).map(id => ({
                        id,
                        label: targets.byId[id]?.target_name || 'Unknown',
                        sublabel: targets.byId[id]?.gene_name || 'No gene',
                        badge: 'Target'
                    }))}
                    badgeClass="badge-target"
                    emptyMessage="No targets in database yet"
                    onItemClick={(id) => onNavigateToGraph && onNavigateToGraph({ id, type: 'target', label: targets.byId[id]?.target_name })}
                />
            </div>
        </div>
    );
};

export default Dashboard;
