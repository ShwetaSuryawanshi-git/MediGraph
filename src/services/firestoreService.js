// File: MediGraph/src/services/firestoreService.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/services/firestoreService.js

/**
 * Firestore Service for MediGraph
 * Handles real-time data subscriptions for all collections
 */

import {
    collection,
    doc,
    query,
    where,
    onSnapshot,
    addDoc,
    updateDoc,
    deleteDoc,
    serverTimestamp
} from 'firebase/firestore';
import { db } from '../../firebase/firebase';

// ═══════════════════════════════════════════════════════════════
// COLLECTION NAMES (as per schema contract)
// ═══════════════════════════════════════════════════════════════
export const COLLECTIONS = {
    USERS: 'users',
    DRUGS: 'drugs',
    TARGETS: 'targets',
    DRUG_TARGET_INTERACTIONS: 'drug_target_interactions',
    DISEASES: 'diseases',
    REPURPOSING_CANDIDATES: 'repurposing_candidates',
    GRAPHS_METADATA: 'graphs_metadata'
};

// ═══════════════════════════════════════════════════════════════
// DRUGS LISTENER
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to all drugs in real-time
 * 
 * @param {Function} onData - Callback with array of drug documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToDrugs = (onData, onError) => {
    const drugsRef = collection(db, COLLECTIONS.DRUGS);

    return onSnapshot(drugsRef,
        (snapshot) => {
            const drugs = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(drugs);
        },
        (error) => {
            console.error('Drugs listener error:', error);
            onError(error);
        }
    );
};

// ═══════════════════════════════════════════════════════════════
// TARGETS LISTENER
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to all targets in real-time
 * 
 * @param {Function} onData - Callback with array of target documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToTargets = (onData, onError) => {
    const targetsRef = collection(db, COLLECTIONS.TARGETS);

    return onSnapshot(targetsRef,
        (snapshot) => {
            const targets = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(targets);
        },
        (error) => {
            console.error('Targets listener error:', error);
            onError(error);
        }
    );
};

// ═══════════════════════════════════════════════════════════════
// DRUG-TARGET INTERACTIONS LISTENER
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to positive drug-target interactions in real-time
 * Only fetches interactions where is_positive == true
 * 
 * @param {Function} onData - Callback with array of interaction documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToInteractions = (onData, onError) => {
    const interactionsRef = collection(db, COLLECTIONS.DRUG_TARGET_INTERACTIONS);
    const q = query(interactionsRef, where('is_positive', '==', true));

    return onSnapshot(q,
        (snapshot) => {
            const interactions = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(interactions);
        },
        (error) => {
            console.error('Interactions listener error:', error);
            onError(error);
        }
    );
};

/**
 * Subscribe to all interactions (including negative predictions)
 * Used for detailed analysis views
 * 
 * @param {Function} onData - Callback with array of interaction documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToAllInteractions = (onData, onError) => {
    const interactionsRef = collection(db, COLLECTIONS.DRUG_TARGET_INTERACTIONS);

    return onSnapshot(interactionsRef,
        (snapshot) => {
            const interactions = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(interactions);
        },
        (error) => {
            console.error('All interactions listener error:', error);
            onError(error);
        }
    );
};

// ═══════════════════════════════════════════════════════════════
// DISEASES LISTENER
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to all diseases in real-time
 * 
 * @param {Function} onData - Callback with array of disease documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToDiseases = (onData, onError) => {
    const diseasesRef = collection(db, COLLECTIONS.DISEASES);

    return onSnapshot(diseasesRef,
        (snapshot) => {
            const diseases = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(diseases);
        },
        (error) => {
            console.error('Diseases listener error:', error);
            onError(error);
        }
    );
};

// ═══════════════════════════════════════════════════════════════
// REPURPOSING CANDIDATES LISTENER
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to all repurposing candidates in real-time
 * 
 * @param {Function} onData - Callback with array of candidate documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToRepurposingCandidates = (onData, onError) => {
    const candidatesRef = collection(db, COLLECTIONS.REPURPOSING_CANDIDATES);

    return onSnapshot(candidatesRef,
        (snapshot) => {
            const candidates = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(candidates);
        },
        (error) => {
            console.error('Repurposing candidates listener error:', error);
            onError(error);
        }
    );
};

/**
 * Subscribe to repurposing candidates filtered by status
 * 
 * @param {string} status - Status filter ('pending', 'reviewed', 'validated', 'rejected')
 * @param {Function} onData - Callback with array of candidate documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToRepurposingByStatus = (status, onData, onError) => {
    const candidatesRef = collection(db, COLLECTIONS.REPURPOSING_CANDIDATES);
    const q = query(candidatesRef, where('status', '==', status));

    return onSnapshot(q,
        (snapshot) => {
            const candidates = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(candidates);
        },
        (error) => {
            console.error('Repurposing candidates (filtered) listener error:', error);
            onError(error);
        }
    );
};

// ═══════════════════════════════════════════════════════════════
// GRAPHS METADATA LISTENER & CRUD
// ═══════════════════════════════════════════════════════════════

/**
 * Subscribe to current user's saved graphs
 * 
 * @param {string} userId - Current user's UID
 * @param {Function} onData - Callback with array of graph metadata documents
 * @param {Function} onError - Error callback
 * @returns {Function} - Unsubscribe function
 */
export const subscribeToUserGraphs = (userId, onData, onError) => {
    const graphsRef = collection(db, COLLECTIONS.GRAPHS_METADATA);
    const q = query(graphsRef, where('user_id', '==', userId));

    return onSnapshot(q,
        (snapshot) => {
            const graphs = snapshot.docs.map(doc => ({
                id: doc.id,
                ...doc.data()
            }));
            onData(graphs);
        },
        (error) => {
            console.error('Graphs metadata listener error:', error);
            onError(error);
        }
    );
};

/**
 * Create a new saved graph for the current user
 * 
 * @param {string} userId - Current user's UID
 * @param {Object} graphData - Graph metadata to save
 * @returns {Promise<string>} - Created document ID
 */
export const createGraphMetadata = async (userId, graphData) => {
    const graphsRef = collection(db, COLLECTIONS.GRAPHS_METADATA);

    const docRef = await addDoc(graphsRef, {
        user_id: userId,
        graph_name: graphData.graphName,
        node_count: graphData.nodeCount,
        edge_count: graphData.edgeCount,
        filters_applied: graphData.filtersApplied || {},
        included_drug_ids: graphData.includedDrugIds || [],
        included_target_ids: graphData.includedTargetIds || [],
        included_disease_ids: graphData.includedDiseaseIds || [],
        layout_config: graphData.layoutConfig || {},
        created_at: serverTimestamp(),
        updated_at: serverTimestamp()
    });

    return docRef.id;
};

/**
 * Update an existing saved graph
 * 
 * @param {string} graphId - Document ID of graph to update
 * @param {Object} updates - Fields to update
 * @returns {Promise<void>}
 */
export const updateGraphMetadata = async (graphId, updates) => {
    const graphRef = doc(db, COLLECTIONS.GRAPHS_METADATA, graphId);

    await updateDoc(graphRef, {
        ...updates,
        updated_at: serverTimestamp()
    });
};

/**
 * Delete a saved graph
 * 
 * @param {string} graphId - Document ID of graph to delete
 * @returns {Promise<void>}
 */
export const deleteGraphMetadata = async (graphId) => {
    const graphRef = doc(db, COLLECTIONS.GRAPHS_METADATA, graphId);
    await deleteDoc(graphRef);
};
