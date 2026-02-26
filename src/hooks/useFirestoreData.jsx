// File: MediGraph/src/hooks/useFirestoreData.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/hooks/useFirestoreData.js

/**
 * Custom React Hook for Real-Time Firestore Data
 * Provides normalized data state and loading indicators for all collections
 */

import { useState, useEffect, useCallback, createContext, useContext } from 'react';
import {
    subscribeToDrugs,
    subscribeToTargets,
    subscribeToInteractions,
    subscribeToDiseases,
    subscribeToRepurposingCandidates,
    subscribeToUserGraphs
} from '../services/firestoreService';
import { useAuth } from './useAuth';

// Create Data Context
const DataContext = createContext(null);

/**
 * Normalize array of documents to { byId, allIds } structure
 */
const normalizeData = (documents) => {
    const byId = {};
    const allIds = [];

    documents.forEach(doc => {
        byId[doc.id] = doc;
        allIds.push(doc.id);
    });

    return { byId, allIds };
};

/**
 * Data Provider Component
 * Subscribes to all Firestore collections and provides normalized data
 */
export const DataProvider = ({ children }) => {
    const { user, isAuthenticated } = useAuth();

    // Loading states per collection
    const [loadingState, setLoadingState] = useState({
        drugs: 'idle',
        targets: 'idle',
        interactions: 'idle',
        diseases: 'idle',
        repurposingCandidates: 'idle',
        graphsMetadata: 'idle'
    });

    // Error states per collection
    const [errors, setErrors] = useState({});

    // Normalized data stores
    const [drugs, setDrugs] = useState({ byId: {}, allIds: [] });
    const [targets, setTargets] = useState({ byId: {}, allIds: [] });
    const [interactions, setInteractions] = useState({ byId: {}, allIds: [] });
    const [diseases, setDiseases] = useState({ byId: {}, allIds: [] });
    const [repurposingCandidates, setRepurposingCandidates] = useState({ byId: {}, allIds: [] });
    const [graphsMetadata, setGraphsMetadata] = useState({ byId: {}, allIds: [] });

    // Subscribe to collections when authenticated
    useEffect(() => {
        if (!isAuthenticated) {
            // Clear all data on logout
            setDrugs({ byId: {}, allIds: [] });
            setTargets({ byId: {}, allIds: [] });
            setInteractions({ byId: {}, allIds: [] });
            setDiseases({ byId: {}, allIds: [] });
            setRepurposingCandidates({ byId: {}, allIds: [] });
            setGraphsMetadata({ byId: {}, allIds: [] });
            setLoadingState({
                drugs: 'idle',
                targets: 'idle',
                interactions: 'idle',
                diseases: 'idle',
                repurposingCandidates: 'idle',
                graphsMetadata: 'idle'
            });
            return;
        }

        // Set all to loading
        setLoadingState({
            drugs: 'loading',
            targets: 'loading',
            interactions: 'loading',
            diseases: 'loading',
            repurposingCandidates: 'loading',
            graphsMetadata: 'loading'
        });

        // Unsubscribe functions
        const unsubscribers = [];

        // Subscribe to drugs
        unsubscribers.push(
            subscribeToDrugs(
                (data) => {
                    setDrugs(normalizeData(data));
                    setLoadingState(prev => ({ ...prev, drugs: 'loaded' }));
                },
                (error) => {
                    setErrors(prev => ({ ...prev, drugs: error.message }));
                    setLoadingState(prev => ({ ...prev, drugs: 'error' }));
                }
            )
        );

        // Subscribe to targets
        unsubscribers.push(
            subscribeToTargets(
                (data) => {
                    setTargets(normalizeData(data));
                    setLoadingState(prev => ({ ...prev, targets: 'loaded' }));
                },
                (error) => {
                    setErrors(prev => ({ ...prev, targets: error.message }));
                    setLoadingState(prev => ({ ...prev, targets: 'error' }));
                }
            )
        );

        // Subscribe to interactions (positive only for graph)
        unsubscribers.push(
            subscribeToInteractions(
                (data) => {
                    setInteractions(normalizeData(data));
                    setLoadingState(prev => ({ ...prev, interactions: 'loaded' }));
                },
                (error) => {
                    setErrors(prev => ({ ...prev, interactions: error.message }));
                    setLoadingState(prev => ({ ...prev, interactions: 'error' }));
                }
            )
        );

        // Subscribe to diseases
        unsubscribers.push(
            subscribeToDiseases(
                (data) => {
                    setDiseases(normalizeData(data));
                    setLoadingState(prev => ({ ...prev, diseases: 'loaded' }));
                },
                (error) => {
                    setErrors(prev => ({ ...prev, diseases: error.message }));
                    setLoadingState(prev => ({ ...prev, diseases: 'error' }));
                }
            )
        );

        // Subscribe to repurposing candidates
        unsubscribers.push(
            subscribeToRepurposingCandidates(
                (data) => {
                    setRepurposingCandidates(normalizeData(data));
                    setLoadingState(prev => ({ ...prev, repurposingCandidates: 'loaded' }));
                },
                (error) => {
                    setErrors(prev => ({ ...prev, repurposingCandidates: error.message }));
                    setLoadingState(prev => ({ ...prev, repurposingCandidates: 'error' }));
                }
            )
        );

        // Subscribe to user's graphs
        if (user?.uid) {
            unsubscribers.push(
                subscribeToUserGraphs(
                    user.uid,
                    (data) => {
                        setGraphsMetadata(normalizeData(data));
                        setLoadingState(prev => ({ ...prev, graphsMetadata: 'loaded' }));
                    },
                    (error) => {
                        setErrors(prev => ({ ...prev, graphsMetadata: error.message }));
                        setLoadingState(prev => ({ ...prev, graphsMetadata: 'error' }));
                    }
                )
            );
        }

        // Cleanup: unsubscribe all on unmount or auth change
        return () => {
            unsubscribers.forEach(unsub => unsub());
        };
    }, [isAuthenticated, user?.uid]);

    // Check if core collections are ready for graph rendering
    const canRenderGraph =
        loadingState.drugs === 'loaded' &&
        loadingState.targets === 'loaded' &&
        loadingState.interactions === 'loaded';

    // Check if all data is loaded
    const isFullyLoaded = Object.values(loadingState).every(s => s === 'loaded');

    // Check if any collection is still loading
    const isLoading = Object.values(loadingState).some(s => s === 'loading');

    // Get drug by ID with null safety
    const getDrugById = useCallback((drugId) => {
        return drugs.byId[drugId] || null;
    }, [drugs.byId]);

    // Get target by ID with null safety
    const getTargetById = useCallback((targetId) => {
        return targets.byId[targetId] || null;
    }, [targets.byId]);

    // Get disease by ID with null safety
    const getDiseaseById = useCallback((diseaseId) => {
        return diseases.byId[diseaseId] || null;
    }, [diseases.byId]);

    // Get interaction by ID with null safety
    const getInteractionById = useCallback((interactionId) => {
        return interactions.byId[interactionId] || null;
    }, [interactions.byId]);

    const value = {
        // Normalized data
        drugs,
        targets,
        interactions,
        diseases,
        repurposingCandidates,
        graphsMetadata,

        // State indicators
        loadingState,
        errors,
        isLoading,
        isFullyLoaded,
        canRenderGraph,

        // Lookup helpers
        getDrugById,
        getTargetById,
        getDiseaseById,
        getInteractionById
    };

    return (
        <DataContext.Provider value={value}>
            {children}
        </DataContext.Provider>
    );
};

/**
 * Custom hook to access data context
 * Must be used within DataProvider
 */
export const useFirestoreData = () => {
    const context = useContext(DataContext);
    if (!context) {
        throw new Error('useFirestoreData must be used within a DataProvider');
    }
    return context;
};

export default useFirestoreData;
