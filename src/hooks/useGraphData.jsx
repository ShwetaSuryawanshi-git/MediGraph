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
    REPURPOSING: 'repurposing',
    DDI: 'ddi'
};

/**
 * Hook for deriving graph data from Firestore collections
 */
export const useGraphData = () => {
    const {
        drugs,
        targets,
        interactions,
        drugDrugInteractions,
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
        selectedTargetIds: [],
        selectedDiseaseIds: [] // empty = all diseases
    });

    // Derive nodes from drugs, targets, and diseases
    const nodes = useMemo(() => {
        if (!canRenderGraph) return [];

        const nodeList = [];
        
        // 1. Calculate Valid Entity Sub-Graphs Based on Filters
        let validTargetIds = new Set(targets.allIds);
        let validDrugIds = new Set(drugs.allIds);
        let validDiseaseIds = new Set(diseases.allIds);

        const hasDiseaseFilter = filters.selectedDiseaseIds.length > 0;
        const hasDrugFilter = filters.selectedDrugIds.length > 0;

        if (hasDiseaseFilter || hasDrugFilter) {
            let resultDrugIds = new Set();
            let resultTargetIds = new Set();
            let resultDiseaseIds = new Set();

            // 1. Process Drug Filter Expansion (Drug -> Targets -> Diseases)
            if (hasDrugFilter) {
                filters.selectedDrugIds.forEach(dId => resultDrugIds.add(dId));
                
                // Find all targets interacting with these drugs
                interactions.allIds.forEach(iid => {
                    const interaction = interactions.byId[iid];
                    if (filters.selectedDrugIds.includes(interaction.drug_id) &&
                        interaction.confidence_score >= filters.minConfidence &&
                        (!filters.modelUsed || interaction.model_used === filters.modelUsed)) {
                        resultTargetIds.add(interaction.target_id);
                    }
                });

                // Find all drugs interacting with these drugs (DDI)
                drugDrugInteractions?.allIds?.forEach(iid => {
                    const ddi = drugDrugInteractions.byId[iid];
                    if (ddi.confidence_score >= filters.minConfidence) {
                        if (filters.selectedDrugIds.includes(ddi.drug_a_id)) {
                            resultDrugIds.add(ddi.drug_b_id);
                        } else if (filters.selectedDrugIds.includes(ddi.drug_b_id)) {
                            resultDrugIds.add(ddi.drug_a_id);
                        }
                    }
                });

                // Find all diseases mapped to those specific targets
                Array.from(resultTargetIds).forEach(tid => {
                    const t = targets.byId[tid];
                    t?.associated_disease_ids?.forEach(dId => resultDiseaseIds.add(dId));
                });
            }

            // 2. Process Disease Filter Expansion (Disease -> Targets -> Drugs)
            if (hasDiseaseFilter) {
                let tempDiseaseTargets = new Set();
                filters.selectedDiseaseIds.forEach(dId => resultDiseaseIds.add(dId));

                // Find all targets mapping to these diseases
                targets.allIds.forEach(tid => {
                    const t = targets.byId[tid];
                    if (t?.associated_disease_ids?.some(dId => filters.selectedDiseaseIds.includes(dId))) {
                        resultTargetIds.add(tid);
                        tempDiseaseTargets.add(tid);
                    }
                });

                // Find all drugs mapped to those specific targets
                interactions.allIds.forEach(iid => {
                    const interaction = interactions.byId[iid];
                    if (tempDiseaseTargets.has(interaction.target_id) &&
                        interaction.confidence_score >= filters.minConfidence &&
                        (!filters.modelUsed || interaction.model_used === filters.modelUsed)) {
                        resultDrugIds.add(interaction.drug_id);
                    }
                });
            }
            
            validTargetIds = resultTargetIds;
            validDrugIds = resultDrugIds;
            validDiseaseIds = resultDiseaseIds;
        }

        // Add drug nodes
        drugs.allIds.forEach(drugId => {
            if (!validDrugIds.has(drugId)) return;
            const drug = drugs.byId[drugId];
            nodeList.push({
                id: drugId,
                type: NODE_TYPES.DRUG,
                label: drug.drug_name || 'Unknown Drug',
                data: drug
            });
        });

        // Add target nodes
        targets.allIds.forEach(targetId => {
            if (!validTargetIds.has(targetId)) return;
            if (filters.selectedTargetIds.length > 0 && !filters.selectedTargetIds.includes(targetId)) return;
            const target = targets.byId[targetId];
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
                if (!validDiseaseIds.has(diseaseId)) return;
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
    }, [drugs, targets, interactions, diseases, canRenderGraph, filters]);

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

        // Add Drug → Drug edges from DDIs
        drugDrugInteractions?.allIds?.forEach(ddiId => {
            const ddi = drugDrugInteractions.byId[ddiId];

            if (ddi.confidence_score < filters.minConfidence) {
                return;
            }

            if (!nodeIds.has(ddi.drug_a_id) || !nodeIds.has(ddi.drug_b_id)) {
                return;
            }

            edgeList.push({
                id: ddiId,
                source: ddi.drug_a_id,
                target: ddi.drug_b_id,
                type: EDGE_TYPES.DDI,
                weight: ddi.confidence_score,
                opacity: ddi.prediction_score,
                style: 'dashed',
                data: ddi
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
            selectedTargetIds: [],
            selectedDiseaseIds: []
        });
    }, []);

    // Get graph statistics
    const stats = useMemo(() => {
        const drugNodes = nodes.filter(n => n.type === NODE_TYPES.DRUG).length;
        const targetNodes = nodes.filter(n => n.type === NODE_TYPES.TARGET).length;
        const diseaseNodes = nodes.filter(n => n.type === NODE_TYPES.DISEASE).length;
        const interactionEdges = edges.filter(e => e.type === EDGE_TYPES.INTERACTION).length;
        const associationEdges = edges.filter(e => e.type === EDGE_TYPES.ASSOCIATION).length;
        const ddiEdges = edges.filter(e => e.type === EDGE_TYPES.DDI).length;

        return {
            totalNodes: nodes.length,
            totalEdges: edges.length,
            drugNodes,
            targetNodes,
            diseaseNodes,
            interactionEdges,
            associationEdges,
            ddiEdges
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
        getDiseaseById,
        rawDrugs: drugs,
        rawDiseases: diseases
    };
};

export default useGraphData;
