// File: MediGraph/src/components/Dashboard/SearchBar.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Dashboard/SearchBar.jsx

/**
 * SearchBar Component
 * Search for drugs, targets, and diseases in the database
 */

import React, { useState, useMemo } from 'react';

const SearchBar = ({ drugs, targets, diseases, onResultClick }) => {
    const [query, setQuery] = useState('');
    const [isOpen, setIsOpen] = useState(false);

    // Search across all entities
    const searchResults = useMemo(() => {
        if (!query.trim() || query.length < 2) return [];

        const q = query.toLowerCase();
        const results = [];

        // Search drugs
        drugs.allIds.forEach(id => {
            const drug = drugs.byId[id];
            if (drug) {
                const name = drug.name || drug.drug_name || '';
                const smiles = drug.smiles || '';
                if (name.toLowerCase().includes(q) || smiles.toLowerCase().includes(q)) {
                    results.push({
                        id,
                        type: 'drug',
                        name: name || 'Unknown Drug',
                        detail: drug.approved_indications?.join(', ') || drug.approved_indication || 'No indication',
                        icon: '💊',
                        badgeClass: 'badge-drug'
                    });
                }
            }
        });

        // Search targets
        targets.allIds.forEach(id => {
            const target = targets.byId[id];
            if (target) {
                const name = target.name || target.target_name || '';
                const gene = target.gene_symbol || target.gene_name || '';
                if (name.toLowerCase().includes(q) || gene.toLowerCase().includes(q)) {
                    results.push({
                        id,
                        type: 'target',
                        name: name || 'Unknown Target',
                        detail: gene || 'No gene symbol',
                        icon: '🎯',
                        badgeClass: 'badge-target'
                    });
                }
            }
        });

        // Search diseases
        diseases.allIds.forEach(id => {
            const disease = diseases.byId[id];
            if (disease) {
                const name = disease.name || disease.disease_name || '';
                const category = disease.category || '';
                if (name.toLowerCase().includes(q) || category.toLowerCase().includes(q)) {
                    results.push({
                        id,
                        type: 'disease',
                        name: name || 'Unknown Disease',
                        detail: category || 'No category',
                        icon: '🏥',
                        badgeClass: 'badge-disease'
                    });
                }
            }
        });

        return results.slice(0, 10); // Limit to 10 results
    }, [query, drugs, targets, diseases]);

    const handleResultClick = (result) => {
        setQuery('');
        setIsOpen(false);
        if (onResultClick) {
            onResultClick(result);
        }
    };

    return (
        <div style={{ position: 'relative', marginBottom: 'var(--spacing-lg)' }}>
            {/* Search Input */}
            <div style={{ position: 'relative' }}>
                <span style={{
                    position: 'absolute',
                    left: 'var(--spacing-md)',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    fontSize: '1.2rem'
                }}>
                    🔍
                </span>
                <input
                    type="text"
                    className="form-input"
                    placeholder="Search drugs, targets, diseases..."
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        setIsOpen(e.target.value.length >= 2);
                    }}
                    onFocus={() => query.length >= 2 && setIsOpen(true)}
                    onBlur={() => setTimeout(() => setIsOpen(false), 200)}
                    style={{
                        paddingLeft: 'var(--spacing-2xl)',
                        width: '100%',
                        maxWidth: '500px',
                        fontSize: '1rem'
                    }}
                />
            </div>

            {/* Search Results Dropdown */}
            {isOpen && searchResults.length > 0 && (
                <div
                    className="glass-card"
                    style={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        right: 0,
                        maxWidth: '500px',
                        marginTop: 'var(--spacing-xs)',
                        zIndex: 'var(--z-dropdown)',
                        padding: 'var(--spacing-sm)',
                        maxHeight: '400px',
                        overflowY: 'auto'
                    }}
                >
                    {searchResults.map((result) => (
                        <div
                            key={`${result.type}-${result.id}`}
                            onClick={() => handleResultClick(result)}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 'var(--spacing-md)',
                                padding: 'var(--spacing-sm) var(--spacing-md)',
                                borderRadius: 'var(--radius-md)',
                                cursor: 'pointer',
                                transition: 'background var(--transition-fast)'
                            }}
                            onMouseEnter={(e) => e.target.style.background = 'var(--glass-bg)'}
                            onMouseLeave={(e) => e.target.style.background = 'transparent'}
                        >
                            <span style={{ fontSize: '1.5rem' }}>{result.icon}</span>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontWeight: 500 }}>{result.name}</div>
                                <div className="text-muted" style={{ fontSize: '0.875rem' }}>
                                    {result.detail}
                                </div>
                            </div>
                            <span className={`badge ${result.badgeClass}`}>
                                {result.type}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* No results message */}
            {isOpen && query.length >= 2 && searchResults.length === 0 && (
                <div
                    className="glass-card"
                    style={{
                        position: 'absolute',
                        top: '100%',
                        left: 0,
                        right: 0,
                        maxWidth: '500px',
                        marginTop: 'var(--spacing-xs)',
                        zIndex: 'var(--z-dropdown)',
                        padding: 'var(--spacing-lg)',
                        textAlign: 'center'
                    }}
                >
                    <p className="text-muted">No results found for "{query}"</p>
                </div>
            )}
        </div>
    );
};

export default SearchBar;
