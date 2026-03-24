// File: MediGraph/src/hooks/useGraphData.js
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/hooks/useGraphData.js

/**
 * Custom React Hook for Graph Data Derivation
 * Transforms normalized Firestore data into graph nodes and edges
 */

import { useMemo, useState, useCallback } from 'react';
import { useFirestoreData } from './useFirestoreData';

/**
 * Node type constants
 */
export const NODE_TYPES = {
    DRUG: 'drug',
    TARGET: 'target',
    DISEASE: 'disease'
};

/**
 * Edge type constants
 */
export const EDGE_TYPES = {
    INTERACTION: 'interaction',
    ASSOCIATION: 'association',
    REPURPOSING: 'repurposing'
};

/**
 * Hook for deriving graph data from Firestore collections
 */
export const useGraphData = () => {
    const {
        drugs,
        targets,
        interactions,
        diseases,
        canRenderGraph,
        getDrugById,
        getTargetById,
        getDiseaseById
    } = useFirestoreData();

    // Filter state
    const [filters, setFilters] = useState({
        minConfidence: 0,
        modelUsed: null, // null = all models
        showDiseases: true,
        selectedDrugIds: [], // empty = all drugs
        selectedTargetIds: [] // empty = all targets
    });

    // Derive nodes from drugs, targets, and diseases
    const nodes = useMemo(() => {
        if (!canRenderGraph) return [];

        const nodeList = [];

        // Add drug nodes
        drugs.allIds.forEach(drugId => {
            const drug = drugs.byId[drugId];
            // Apply drug filter if set
            if (filters.selectedDrugIds.length > 0 && !filters.selectedDrugIds.includes(drugId)) {
                return;
            }
            nodeList.push({
                id: drugId,
                type: NODE_TYPES.DRUG,
                label: drug.drug_name || 'Unknown Drug',
                data: drug
            });
        });

        // Add target nodes
        targets.allIds.forEach(targetId => {
            const target = targets.byId[targetId];
            // Apply target filter if set
            if (filters.selectedTargetIds.length > 0 && !filters.selectedTargetIds.includes(targetId)) {
                return;
            }
            nodeList.push({
                id: targetId,
                type: NODE_TYPES.TARGET,
                label: target.target_name || 'Unknown Target',
                data: target
            });
        });

        // Add disease nodes if enabled
        if (filters.showDiseases) {
            diseases.allIds.forEach(diseaseId => {
                const disease = diseases.byId[diseaseId];
                nodeList.push({
                    id: diseaseId,
                    type: NODE_TYPES.DISEASE,
                    label: disease.disease_name || 'Unknown Disease',
                    data: disease
                });
            });
        }

        return nodeList;
    }, [drugs, targets, diseases, canRenderGraph, filters]);

    // Create a set of node IDs for edge validation
    const nodeIds = useMemo(() => {
        return new Set(nodes.map(n => n.id));
    }, [nodes]);

    // Derive edges from interactions and target-disease associations
    const edges = useMemo(() => {
        if (!canRenderGraph) return [];

        const edgeList = [];

        // Add Drug → Target edges from interactions
        interactions.allIds.forEach(interactionId => {
            const interaction = interactions.byId[interactionId];

            // Apply confidence filter
            if (interaction.confidence_score < filters.minConfidence) {
                return;
            }

            // Apply model filter
            if (filters.modelUsed && interaction.model_used !== filters.modelUsed) {
                return;
            }

            // Validate both nodes exist (handle delayed data)
            if (!nodeIds.has(interaction.drug_id) || !nodeIds.has(interaction.target_id)) {
                return;
            }

            edgeList.push({
                id: interactionId,
                source: interaction.drug_id,
                target: interaction.target_id,
                type: EDGE_TYPES.INTERACTION,
                weight: interaction.confidence_score,
                opacity: interaction.prediction_score,
                style: interaction.interaction_type === 'validated' ? 'solid' : 'dashed',
                data: interaction
            });
        });

        // Add Target → Disease edges from associations
        if (filters.showDiseases) {
            targets.allIds.forEach(targetId => {
                const target = targets.byId[targetId];
                const diseaseIds = target.associated_disease_ids || [];

                diseaseIds.forEach(diseaseId => {
                    // Validate both nodes exist
                    if (!nodeIds.has(targetId) || !nodeIds.has(diseaseId)) {
                        return;
                    }

                    edgeList.push({
                        id: `${targetId}-${diseaseId}`,
                        source: targetId,
                        target: diseaseId,
                        type: EDGE_TYPES.ASSOCIATION,
                        weight: 1.0,
                        opacity: 1.0,
                        style: 'solid',
                        data: null
                    });
                });
            });
        }

        return edgeList;
    }, [interactions, targets, canRenderGraph, nodeIds, filters]);

    // Update filters
    const updateFilters = useCallback((newFilters) => {
        setFilters(prev => ({ ...prev, ...newFilters }));
    }, []);

    // Reset filters to default
    const resetFilters = useCallback(() => {
        setFilters({
            minConfidence: 0,
            modelUsed: null,
            showDiseases: true,
            selectedDrugIds: [],
            selectedTargetIds: []
        });
    }, []);

    // Get graph statistics
    const stats = useMemo(() => {
        const drugNodes = nodes.filter(n => n.type === NODE_TYPES.DRUG).length;
        const targetNodes = nodes.filter(n => n.type === NODE_TYPES.TARGET).length;
        const diseaseNodes = nodes.filter(n => n.type === NODE_TYPES.DISEASE).length;
        const interactionEdges = edges.filter(e => e.type === EDGE_TYPES.INTERACTION).length;
        const associationEdges = edges.filter(e => e.type === EDGE_TYPES.ASSOCIATION).length;

        return {
            totalNodes: nodes.length,
            totalEdges: edges.length,
            drugNodes,
            targetNodes,
            diseaseNodes,
            interactionEdges,
            associationEdges
        };
    }, [nodes, edges]);

    return {
        nodes,
        edges,
        filters,
        updateFilters,
        resetFilters,
        stats,
        canRenderGraph,
        // Lookup helpers for detail views
        getDrugById,
        getTargetById,
        getDiseaseById
    };
};

export default useGraphData;
