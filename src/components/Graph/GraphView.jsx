// File: MediGraph/src/components/Graph/GraphView.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/GraphView.jsx

/**
 * Graph View Component
 * Main container for the drug-target-disease network visualization
 */

import React, { useState, useEffect } from 'react';
import { useGraphData, NODE_TYPES } from '../../hooks/useGraphData';
import { useAuth } from '../../hooks/useAuth';
import { createGraphMetadata } from '../../services/firestoreService';
import GraphCanvas from './GraphCanvas';
import GraphFilters from './GraphFilters';
import GraphLegend from './GraphLegend';
import NodeDetail from './NodeDetail';

const GraphView = ({ initialEntity, initialFilters, onClearInitial }) => {
    const { user } = useAuth();
    const {
        nodes,
        edges,
        filters,
        updateFilters,
        resetFilters,
        stats,
        canRenderGraph,
        getDrugById,
        getTargetById,
        getDiseaseById,
        rawDrugs,
        rawDiseases
    } = useGraphData();

    const [selectedNode, setSelectedNode] = useState(null);
    const [focusTooltipVisible, setFocusTooltipVisible] = useState(false);
    const [hasSeenTooltip, setHasSeenTooltip] = useState(false);

    useEffect(() => {
        if (selectedNode && !hasSeenTooltip) {
            setFocusTooltipVisible(true);
            setHasSeenTooltip(true);
            const timer = setTimeout(() => {
                setFocusTooltipVisible(false);
            }, 6000);
            return () => clearTimeout(timer);
        } else if (!selectedNode) {
            setFocusTooltipVisible(false);
        }
    }, [selectedNode, hasSeenTooltip]);
    const [activeTab, setActiveTab] = useState('filters');
    const [aiInference, setAiInference] = useState(null);
    const [isGeneratingAI, setIsGeneratingAI] = useState(false);

    // Watch for activeEntity or savedFilters coming from Dashboard or Saved Graphs
    useEffect(() => {
        let cleared = false;
        
        if (initialFilters) {
            updateFilters(initialFilters);
            cleared = true;
        }

        if (initialEntity) {
            handleNodeClick(initialEntity.id, initialEntity.type);
            
            // Auto update filters to constrain to this exact node
            if (initialEntity.type === NODE_TYPES.DRUG) {
                updateFilters({ selectedDrugIds: [initialEntity.id] });
            } else if (initialEntity.type === NODE_TYPES.DISEASE) {
                updateFilters({ selectedDiseaseIds: [initialEntity.id], showDiseases: true });
            }
            
            cleared = true;
        }

        if (cleared && onClearInitial) {
            onClearInitial(); // clear so we don't infinitely re-run
        }
    }, [initialEntity, initialFilters]);

    // Handle Generate AI Inference
    const handleGenerateAIInference = async () => {
        setIsGeneratingAI(true);
        setAiInference(null);
        try {
            // pass mapped lightweight representation
            const payload = {
                nodes: nodes.map(n => ({ id: n.id, label: n.label, type: n.type })),
                edges: edges.map(e => ({ source: e.source, target: e.target }))
            };
            const response = await fetch('http://localhost:8000/api/explain-graph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (data.rationale) {
                setAiInference(data.rationale);
            } else {
                setAiInference("Failed to generate AI reasoning. Please ensure backend is running.");
            }
        } catch (error) {
            console.error("AI Generation Error: ", error);
            setAiInference("Error connecting to the local AI backend (http://localhost:8000). Ensure the FastAPI server is running.");
        } finally {
            setIsGeneratingAI(false);
        }
    };

    // Handle node click
    const handleNodeClick = (nodeId, nodeType) => {
        let nodeData = null;

        switch (nodeType) {
            case NODE_TYPES.DRUG:
                nodeData = { type: nodeType, data: getDrugById(nodeId) };
                break;
            case NODE_TYPES.TARGET:
                nodeData = { type: nodeType, data: getTargetById(nodeId) };
                break;
            case NODE_TYPES.DISEASE:
                nodeData = { type: nodeType, data: getDiseaseById(nodeId) };
                break;
            default:
                break;
        }

        setSelectedNode(nodeData);
        setActiveTab('inspector'); // Auto-switch to inspector when an entity is clicked
    };

    // Close detail panel
    const handleCloseDetail = () => {
        setSelectedNode(null);
    };

    // Save Graph function
    const handleSaveGraph = async () => {
        if (!user) {
            alert('Please login to save graphs');
            return;
        }
        const graphName = prompt('Enter a name for this saved graph:');
        if (!graphName) return;

        try {
            await createGraphMetadata(user.uid, {
                graphName,
                nodeCount: stats.totalNodes,
                edgeCount: stats.totalEdges,
                filtersApplied: filters,
                includedDrugIds: nodes.filter(n => n.type === NODE_TYPES.DRUG).map(n => n.id),
                includedTargetIds: nodes.filter(n => n.type === NODE_TYPES.TARGET).map(n => n.id),
                includedDiseaseIds: nodes.filter(n => n.type === NODE_TYPES.DISEASE).map(n => n.id)
            });
            alert('Graph saved successfully! View it in the "Saved Graphs" tab.');
            resetFilters();
            setSelectedNode(null);
        } catch (err) {
            console.error('Save Graph Error:', err);
            alert('Failed to save graph.');
        }
    };

    // Calculate Global Graph Inference
    const getGraphInference = () => {
        if (nodes.length === 0) return null;

        // Is there an active filter rendering this sub-graph?
        const isGraphFiltered = filters.selectedDiseaseIds.length > 0 || filters.selectedDrugIds.length > 0;

        if (!isGraphFiltered) {
            return (
                <div style={{ padding: 'var(--spacing-lg)' }}>
                    <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: '1.5' }}>
                        Your graph is completely empty. Please utilize the Explorer Filters to isolate specific drugs or diseases.
                    </p>
                </div>
            );
        }

        // Generate dynamically natural sentence summarizing the current visible graph
        let naturalSummary = "";
        
        let visibleDrugNames = nodes.filter(n => n.type === NODE_TYPES.DRUG).map(n => n.label);
        let visibleDiseaseNames = nodes.filter(n => n.type === NODE_TYPES.DISEASE).map(n => n.label);
        
        if (filters.selectedDrugIds.length > 0 && filters.selectedDiseaseIds.length > 0) {
            // Both drug and disease were explicitly filtered
            // Check if there is a path (simply if nodes contains both and edges > 0 is a proxy, but accurately: does graph have edges?)
            if (edges.length === 0) {
                naturalSummary = `There is absolutely no biological connection predicted between ${filters.selectedDrugIds.length > 1 ? 'these drugs' : visibleDrugNames[0] || 'this drug'} and ${filters.selectedDiseaseIds.length > 1 ? 'these diseases' : visibleDiseaseNames[0] || 'this disease'}. They exist as isolated points with zero structural affinity overlap.`;
            } else {
                naturalSummary = `The AI model successfully mapped a biological bridge indicating that ${filters.selectedDrugIds.length > 1 ? 'the selected drugs' : visibleDrugNames[0] || 'the drug'} can interact with mechanisms related to ${filters.selectedDiseaseIds.length > 1 ? 'the selected diseases' : visibleDiseaseNames[0] || 'the disease'}.`;
            }
        } else if (filters.selectedDrugIds.length > 0) {
            if (visibleDiseaseNames.length > 0) {
                const diseaseList = visibleDiseaseNames.slice(0, 2).join(" and ") + (visibleDiseaseNames.length > 2 ? ` and ${visibleDiseaseNames.length - 2} others` : "");
                naturalSummary = `The drug ${visibleDrugNames[0] || 'compound'}, which affects its primary targets, also mathematically networks into potential interactions with ${diseaseList}, suggesting novel repurposing mechanics.`;
            } else {
                naturalSummary = `The drug ${visibleDrugNames[0] || 'compound'} interacts with isolated targets but currently possesses no known bridges to the diseases tracked in this dataset.`;
            }
        } else if (filters.selectedDiseaseIds.length > 0) {
            if (visibleDrugNames.length > 0) {
                const drugList = visibleDrugNames.slice(0, 2).join(" and ") + (visibleDrugNames.length > 2 ? ` and ${visibleDrugNames.length - 2} others` : "");
                naturalSummary = `The disease ${visibleDiseaseNames[0] || 'condition'} is structurally influenced by targets that can be directly modulated by ${drugList}.`;
            } else {
                naturalSummary = `The selected disease pathway presents no known target locks for any drugs currently loaded in the active model pipeline.`;
            }
        }

        // Calculate Network Hub
        const nodeDegrees = {};
        edges.forEach(edge => {
            nodeDegrees[edge.source] = (nodeDegrees[edge.source] || 0) + 1;
            nodeDegrees[edge.target] = (nodeDegrees[edge.target] || 0) + 1;
        });

        let hubNodeId = null;
        let maxDegree = 0;
        Object.entries(nodeDegrees).forEach(([id, degree]) => {
            if (degree > maxDegree) {
                maxDegree = degree;
                hubNodeId = id;
            }
        });

        const hubNode = nodes.find(n => n.id === hubNodeId);

        return (
            <div style={{ padding: 'var(--spacing-sm)' }}>
                <h4 style={{ marginBottom: 'var(--spacing-md)', fontSize: '1rem', borderBottom: '1px solid var(--color-border)', paddingBottom: 'var(--spacing-sm)' }}>
                    Sub-Graph Topology Inference
                </h4>
                
                <div style={{ 
                    background: 'var(--color-bg-tertiary)', 
                    padding: 'var(--spacing-md)', 
                    borderRadius: 'var(--radius-md)',
                    marginBottom: 'var(--spacing-md)',
                    borderLeft: '4px solid var(--color-primary)'
                }}>
                    <p style={{ margin: '0 0 var(--spacing-sm) 0', fontSize: '0.9rem', lineHeight: '1.5' }}>
                        This network isolated view contains <strong>{stats.totalNodes} entities</strong> connected by <strong>{stats.interactionEdges} predicted interactions</strong> and {stats.associationEdges} known associations.
                    </p>
                    <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: '1.5', fontStyle: 'italic', fontWeight: 500 }}>
                        {naturalSummary}
                    </p>
                </div>

                {hubNode && (
                    <div style={{ 
                        background: 'rgba(16, 185, 129, 0.1)', 
                        padding: 'var(--spacing-md)', 
                        borderRadius: 'var(--radius-md)',
                        borderLeft: '4px solid #10b981'
                    }}>
                        <h4 style={{ margin: '0 0 var(--spacing-xs) 0', color: '#10b981', fontSize: '0.9rem' }}>Network Hub</h4>
                        <p style={{ margin: 0, fontSize: '0.85rem' }}>
                            The most connected entity in this view is <strong>{hubNode.label}</strong> ({hubNode.type}) with {maxDegree} structural connections. Targeting or altering this node will have the highest cascade effect on this specific biological network.
                        </p>
                    </div>
                )}
                
                {/* AI RAG INFERENCE BUTTON & DISPLAY */}
                <div style={{ marginTop: 'var(--spacing-lg)' }}>
                    <button 
                        className="btn" 
                        onClick={handleGenerateAIInference}
                        disabled={isGeneratingAI}
                        style={{ 
                            width: '100%', 
                            display: 'flex', 
                            justifyContent: 'center', 
                            alignItems: 'center', 
                            gap: '8px',
                            background: 'var(--color-primary)',
                            color: 'white',
                            border: 'none',
                            padding: 'var(--spacing-sm)',
                            borderRadius: 'var(--radius-md)',
                            cursor: isGeneratingAI ? 'not-allowed' : 'pointer',
                            opacity: isGeneratingAI ? 0.7 : 1,
                            fontWeight: '500'
                        }}
                    >
                        {isGeneratingAI ? <span>⏳</span> : <span>🤖</span>}
                        {isGeneratingAI ? 'Retrieving Scientific Literature...' : 'Generate AI Insight (RAG Workflow)'}
                    </button>
                    
                    {aiInference && (
                        <div className="animate-fade-in" style={{
                            marginTop: 'var(--spacing-md)',
                            padding: 'var(--spacing-md)',
                            background: 'var(--color-bg-primary)',
                            border: '1px solid var(--color-border)',
                            borderRadius: 'var(--radius-md)',
                            fontSize: '0.85rem',
                            lineHeight: '1.6',
                            maxHeight: '400px',
                            overflowY: 'auto',
                            boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.05)'
                        }}>
                            <h4 style={{ margin: '0 0 var(--spacing-sm) 0', color: 'var(--color-primary)' }}>RAG AI Analysis</h4>
                            <div style={{ whiteSpace: 'pre-wrap' }}>{aiInference}</div>
                        </div>
                    )}
                </div>
                
                <div style={{ marginTop: 'var(--spacing-xl)', textAlign: 'center', opacity: 0.6 }}>
                    <div className="empty-state-icon" style={{ fontSize: '2rem', marginBottom: 'var(--spacing-sm)' }}>🖱️</div>
                    <p className="text-secondary" style={{ fontSize: '0.8rem', margin: 0 }}>Select a specific node above to view its proprietary AI properties.</p>
                </div>
            </div>
        );
    };

    if (!canRenderGraph) {
        return (
            <div className="container" style={{ padding: 'var(--spacing-xl)' }}>
                <div className="empty-state">
                    <div className="empty-state-icon">⏳</div>
                    <h3>Loading Graph Data</h3>
                    <p className="text-secondary">
                        Waiting for drugs, targets, and interactions to load...
                    </p>
                    <div className="loading-spinner" style={{ marginTop: 'var(--spacing-lg)' }}></div>
                </div>
            </div>
        );
    }

    // Check for empty data
    if (nodes.length === 0) {
        return (
            <div className="container" style={{ padding: 'var(--spacing-xl)' }}>
                <div className="empty-state">
                    <div className="empty-state-icon">📊</div>
                    <h3>No Graph Data Available</h3>
                    <p className="text-secondary">
                        Run the ML pipeline to generate drug-target interactions.
                    </p>
                </div>
            </div>
        );
    }

    // Check if graph is completely unfiltered (Empty Build State)
    const isGraphFiltered = filters.selectedDiseaseIds.length > 0 || filters.selectedDrugIds.length > 0;

    return (
        <div className="animate-fade-in" style={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
            {/* Left Side: Graph Canvas (70%) */}
            <div style={{ flex: '1', position: 'relative', backgroundColor: 'var(--color-bg-primary)', display: 'flex', flexDirection: 'column' }}>
                {!isGraphFiltered ? (
                    <div style={{ 
                        flex: '1', 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        opacity: 0.6,
                        padding: 'var(--spacing-2xl)'
                    }}>
                        <div style={{ fontSize: '4rem', marginBottom: 'var(--spacing-md)' }}>🧬</div>
                        <h2 style={{ marginBottom: 'var(--spacing-sm)' }}>Your graph will be displayed here</h2>
                        <p style={{ textAlign: 'center', maxWidth: '400px', lineHeight: '1.6' }}>
                            Start building your network map by adding Diseases or Drugs from the Explorer Filters pane on the right. Related biological entity nodes will automatically spawn and interlock into position.
                        </p>
                    </div>
                ) : (
                    <>
                        <div style={{ position: 'absolute', top: 'var(--spacing-md)', left: 'var(--spacing-md)', zIndex: 10, background: 'var(--glass-bg)', padding: 'var(--spacing-sm) var(--spacing-md)', borderRadius: 'var(--radius-md)', backdropFilter: 'blur(8px)' }}>
                            <h2 style={{ marginBottom: 0, fontSize: '1.25rem' }}>Network Graph</h2>
                            <p className="text-secondary" style={{ fontSize: '0.875rem' }}>
                                {stats.totalNodes} nodes • {stats.totalEdges} edges
                            </p>
                        </div>
                        
                        <div style={{ width: '100%', height: '100%' }}>
                            <GraphCanvas
                                nodes={nodes}
                                edges={edges}
                                selectedNodeId={selectedNode ? selectedNode.data.id : null}
                                onNodeClick={handleNodeClick}
                            />
                            
                            {/* Focus Mode Node Clarification Tooltip */}
                            {focusTooltipVisible && (
                                <div className="animate-fade-in" style={{ 
                                    position: 'absolute', 
                                    bottom: 'var(--spacing-xl)', 
                                    left: '50%', 
                                    transform: 'translateX(-50%)',
                                    backgroundColor: 'var(--color-bg-secondary)',
                                    color: 'var(--color-text-primary)',
                                    padding: 'var(--spacing-sm) var(--spacing-lg)',
                                    borderRadius: '50px',
                                    boxShadow: '0 4px 15px rgba(0,0,0,0.1)',
                                    zIndex: 100,
                                    border: '1px solid var(--color-border)',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '8px',
                                    fontSize: '0.9rem',
                                    fontWeight: '500'
                                }}>
                                    <span>🔦</span> Focus Mode Active: Click on the faded background nodes to reveal their hidden interactions.
                                    <button 
                                        onClick={() => setFocusTooltipVisible(false)}
                                        style={{ background: 'none', border: 'none', cursor: 'pointer', marginLeft: '8px', lineHeight: '1', fontSize: '1.2rem', opacity: 0.6 }}
                                    >
                                        &times;
                                    </button>
                                </div>
                            )}

                            <GraphLegend />
                        </div>
                    </>
                )}
            </div>

            {/* Right Side: Inspector Sidebar (30%) */}
            <div style={{ 
                width: '400px', 
                minWidth: '350px',
                borderLeft: '1px solid var(--color-border)', 
                display: 'flex', 
                flexDirection: 'column', 
                backgroundColor: 'var(--color-bg-secondary)',
                boxShadow: '-4px 0 15px rgba(0, 0, 0, 0.05)',
                zIndex: 20
            }}>
                {/* macOS Tabs Header */}
                <div style={{ padding: 'var(--spacing-md) var(--spacing-lg) 0 var(--spacing-lg)' }}>
                    <div className="macos-tabs">
                        <div 
                            className={`macos-tab ${activeTab === 'filters' ? 'active' : ''}`}
                            onClick={() => setActiveTab('filters')}
                        >
                            <span style={{ fontSize: '1.2rem' }}>🔬</span> Filters
                        </div>
                        <div 
                            className={`macos-tab ${activeTab === 'inspector' ? 'active' : ''}`}
                            onClick={() => setActiveTab('inspector')}
                        >
                            <span style={{ fontSize: '1.2rem' }}>📋</span> Inspector
                        </div>
                    </div>
                </div>

                {/* Tab Content Area */}
                <div style={{ flex: '1', overflowY: 'auto', padding: 'var(--spacing-lg)', borderTop: '1px solid var(--color-border)' }}>
                    {activeTab === 'filters' ? (
                        <div className="animate-fade-in">
                            <GraphFilters
                                filters={filters}
                                onUpdateFilters={updateFilters}
                                onReset={resetFilters}
                                onSaveGraph={handleSaveGraph}
                                drugsList={rawDrugs?.allIds.map(id => rawDrugs.byId[id])}
                                diseasesList={rawDiseases?.allIds.map(id => rawDiseases.byId[id])}
                            />
                        </div>
                    ) : (
                        <div className="animate-fade-in" style={{ height: '100%' }}>
                            {selectedNode ? (
                                <NodeDetail
                                    nodeType={selectedNode.type}
                                    nodeData={selectedNode.data}
                                    edges={edges}
                                    nodes={nodes}
                                    inline={true}
                                    onClose={handleCloseDetail}
                                />
                            ) : (
                                getGraphInference()
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default GraphView;
