// File: MediGraph/src/components/Graph/GraphCanvas.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/GraphCanvas.jsx

/**
 * Graph Canvas Component
 * Renders the actual network visualization using HTML5 Canvas
 * Note: In production, consider using a library like react-force-graph, vis-network, or cytoscape
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { NODE_TYPES, EDGE_TYPES } from '../../hooks/useGraphData';

// Node colors matching CSS design system
const NODE_COLORS = {
    [NODE_TYPES.DRUG]: '#ef4444',
    [NODE_TYPES.TARGET]: '#3b82f6',
    [NODE_TYPES.DISEASE]: '#10b981'
};

// Node shapes
const NODE_RADIUS = {
    [NODE_TYPES.DRUG]: 20,
    [NODE_TYPES.TARGET]: 18,
    [NODE_TYPES.DISEASE]: 16
};

const GraphCanvas = ({ nodes, edges, selectedNodeId, onNodeClick }) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
    const [nodePositions, setNodePositions] = useState({});
    const [hoveredNode, setHoveredNode] = useState(null);

    // Initialize node positions using force-directed-like placement
    useEffect(() => {
        if (nodes.length === 0) return;

        const positions = {};
        const centerX = dimensions.width / 2;
        const centerY = dimensions.height / 2;
        const radius = Math.min(dimensions.width, dimensions.height) * 0.40;

        // Group nodes by type
        const drugNodes = nodes.filter(n => n.type === NODE_TYPES.DRUG);
        const targetNodes = nodes.filter(n => n.type === NODE_TYPES.TARGET);
        const diseaseNodes = nodes.filter(n => n.type === NODE_TYPES.DISEASE);

        // Position drugs on the left arc
        drugNodes.forEach((node, i) => {
            const angle = (-Math.PI / 2) + (Math.PI * (i / Math.max(drugNodes.length - 1, 1)));
            positions[node.id] = {
                x: centerX - radius * 0.8 + Math.cos(angle) * radius * 0.5,
                y: centerY + Math.sin(angle) * radius * 1.0
            };
        });

        // Position targets in the center
        targetNodes.forEach((node, i) => {
            const cols = Math.ceil(Math.sqrt(targetNodes.length));
            const row = Math.floor(i / cols);
            const col = i % cols;
            const spacing = 100;
            positions[node.id] = {
                x: centerX + (col - cols / 2) * spacing,
                y: centerY + (row - Math.ceil(targetNodes.length / cols) / 2) * spacing
            };
        });

        // Position diseases on the right arc
        diseaseNodes.forEach((node, i) => {
            const angle = (-Math.PI / 2) + (Math.PI * (i / Math.max(diseaseNodes.length - 1, 1)));
            positions[node.id] = {
                x: centerX + radius * 0.8 + Math.cos(angle) * radius * 0.5,
                y: centerY + Math.sin(angle) * radius * 1.0
            };
        });

        setNodePositions(positions);
    }, [nodes, dimensions]);

    // Handle resize
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };

        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Draw the graph
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, dimensions.width, dimensions.height);

        // Determine connected nodes if one is selected
        const connectedNodeIds = new Set();
        if (selectedNodeId) {
            connectedNodeIds.add(selectedNodeId);
            edges.forEach(edge => {
                if (edge.source === selectedNodeId) connectedNodeIds.add(edge.target);
                if (edge.target === selectedNodeId) connectedNodeIds.add(edge.source);
            });
        }

        // Draw edges
        edges.forEach(edge => {
            const sourcePos = nodePositions[edge.source];
            const targetPos = nodePositions[edge.target];
            if (!sourcePos || !targetPos) return;

            const isConnectedToSelected = selectedNodeId ? (edge.source === selectedNodeId || edge.target === selectedNodeId) : true;
            const edgeOpacity = selectedNodeId && !isConnectedToSelected ? 0.05 : 1.0;

            ctx.beginPath();
            ctx.moveTo(sourcePos.x, sourcePos.y);
            ctx.lineTo(targetPos.x, targetPos.y);

            // Edge styling based on type
            if (edge.type === EDGE_TYPES.INTERACTION) {
                ctx.strokeStyle = `rgba(99, 102, 241, ${(0.3 + edge.opacity * 0.5) * edgeOpacity})`;
                ctx.lineWidth = 1 + edge.weight * 2;
                if (edge.style === 'dashed') {
                    ctx.setLineDash([5, 5]);
                } else {
                    ctx.setLineDash([]);
                }
            } else if (edge.type === EDGE_TYPES.DDI) {
                // Style DDI as red, dashed line to symbolize conflict/adverse reactions
                ctx.strokeStyle = `rgba(239, 68, 68, ${(0.4 + edge.opacity * 0.6) * edgeOpacity})`; // Red-500
                ctx.lineWidth = 2 + edge.weight * 2;
                ctx.setLineDash([6, 4]); // Different dash pattern to distinguish from interaction
            } else {
                ctx.strokeStyle = `rgba(16, 185, 129, ${0.3 * edgeOpacity})`;
                ctx.lineWidth = 1;
                ctx.setLineDash([]);
            }

            ctx.stroke();
            ctx.setLineDash([]);
        });

        // Draw nodes
        nodes.forEach(node => {
            const pos = nodePositions[node.id];
            if (!pos) return;

            const radius = NODE_RADIUS[node.type];
            const color = NODE_COLORS[node.type];
            const isHovered = hoveredNode === node.id;
            const isSelected = selectedNodeId === node.id;
            const isConnected = selectedNodeId ? connectedNodeIds.has(node.id) : true;
            
            const nodeOpacity = selectedNodeId && !isConnected ? 0.2 : 1.0;

            // Node shadow
            if (nodeOpacity > 0.5) {
                ctx.shadowColor = color;
                ctx.shadowBlur = isHovered || isSelected ? 20 : 10;
            } else {
                ctx.shadowBlur = 0;
            }

            // Node circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + (isHovered || isSelected ? 3 : 0), 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.globalAlpha = nodeOpacity;
            ctx.fill();

            // Reset shadow
            ctx.shadowBlur = 0;

            // Node border
            ctx.strokeStyle = `rgba(0, 0, 0, ${0.1 * nodeOpacity})`;
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = '#0f172a';
            ctx.globalAlpha = nodeOpacity;
            ctx.font = isSelected ? 'bold 13px Inter, sans-serif' : '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';

            // Truncate label if too long
            const label = node.label.length > 25
                ? node.label.substring(0, 23) + '...'
                : node.label;
            ctx.fillText(label, pos.x, pos.y + radius + 8);
            ctx.globalAlpha = 1.0; // reset for next operations
        });
    }, [nodes, edges, nodePositions, dimensions, hoveredNode, selectedNodeId]);

    // Handle mouse move for hover
    const handleMouseMove = useCallback((e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        let found = null;
        for (const node of nodes) {
            const pos = nodePositions[node.id];
            if (!pos) continue;

            const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
            if (distance < NODE_RADIUS[node.type] + 5) {
                found = node.id;
                break;
            }
        }

        setHoveredNode(found);
        canvas.style.cursor = found ? 'pointer' : 'default';
    }, [nodes, nodePositions]);

    // Handle click
    const handleClick = useCallback((e) => {
        if (hoveredNode) {
            const node = nodes.find(n => n.id === hoveredNode);
            if (node && onNodeClick) {
                onNodeClick(node.id, node.type);
            }
        }
    }, [hoveredNode, nodes, onNodeClick]);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%' }}>
            <canvas
                ref={canvasRef}
                width={dimensions.width}
                height={dimensions.height}
                onMouseMove={handleMouseMove}
                onClick={handleClick}
                style={{ display: 'block' }}
            />
        </div>
    );
};

export default GraphCanvas;
