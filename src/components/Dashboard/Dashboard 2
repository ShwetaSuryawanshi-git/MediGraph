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

const Dashboard = () => {
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
                    // You could navigate to graph view or show a modal here
                    alert(`Selected: ${result.name} (${result.type})`);
                }}
            />

            {/* Stats Grid */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: 'var(--spacing-lg)',
                    marginBottom: 'var(--spacing-xl)'
                }}
            >
                <StatsCard
                    icon="💊"
                    label="Drugs"
                    value={stats.totalDrugs}
                    color="var(--color-drug)"
                    loading={loadingState.drugs === 'loading'}
                />
                <StatsCard
                    icon="🎯"
                    label="Targets"
                    value={stats.totalTargets}
                    color="var(--color-target)"
                    loading={loadingState.targets === 'loading'}
                />
                <StatsCard
                    icon="🔗"
                    label="Interactions"
                    value={stats.totalInteractions}
                    color="var(--color-primary)"
                    loading={loadingState.interactions === 'loading'}
                />
                <StatsCard
                    icon="🏥"
                    label="Diseases"
                    value={stats.totalDiseases}
                    color="var(--color-disease)"
                    loading={loadingState.diseases === 'loading'}
                />
                <StatsCard
                    icon="⏳"
                    label="Pending Review"
                    value={stats.pendingRepurposing}
                    color="var(--color-warning)"
                    loading={loadingState.repurposingCandidates === 'loading'}
                />
                <StatsCard
                    icon="✅"
                    label="Validated"
                    value={stats.validatedRepurposing}
                    color="var(--color-success)"
                    loading={loadingState.repurposingCandidates === 'loading'}
                />
            </div>

            {/* Recent Activity */}
            <div
                style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
                    gap: 'var(--spacing-lg)'
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
                />
            </div>
        </div>
    );
};

export default Dashboard;
