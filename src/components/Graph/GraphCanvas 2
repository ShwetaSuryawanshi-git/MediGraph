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
    [NODE_TYPES.DISEASE]: '#8b5cf6'
};

// Node shapes
const NODE_RADIUS = {
    [NODE_TYPES.DRUG]: 20,
    [NODE_TYPES.TARGET]: 18,
    [NODE_TYPES.DISEASE]: 16
};

const GraphCanvas = ({ nodes, edges, onNodeClick }) => {
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
        const radius = Math.min(dimensions.width, dimensions.height) * 0.35;

        // Group nodes by type
        const drugNodes = nodes.filter(n => n.type === NODE_TYPES.DRUG);
        const targetNodes = nodes.filter(n => n.type === NODE_TYPES.TARGET);
        const diseaseNodes = nodes.filter(n => n.type === NODE_TYPES.DISEASE);

        // Position drugs on the left arc
        drugNodes.forEach((node, i) => {
            const angle = (-Math.PI / 2) + (Math.PI * (i / Math.max(drugNodes.length - 1, 1)));
            positions[node.id] = {
                x: centerX - radius * 0.6 + Math.cos(angle) * radius * 0.4,
                y: centerY + Math.sin(angle) * radius * 0.8
            };
        });

        // Position targets in the center
        targetNodes.forEach((node, i) => {
            const cols = Math.ceil(Math.sqrt(targetNodes.length));
            const row = Math.floor(i / cols);
            const col = i % cols;
            const spacing = 60;
            positions[node.id] = {
                x: centerX + (col - cols / 2) * spacing,
                y: centerY + (row - Math.ceil(targetNodes.length / cols) / 2) * spacing
            };
        });

        // Position diseases on the right arc
        diseaseNodes.forEach((node, i) => {
            const angle = (-Math.PI / 2) + (Math.PI * (i / Math.max(diseaseNodes.length - 1, 1)));
            positions[node.id] = {
                x: centerX + radius * 0.6 + Math.cos(angle) * radius * 0.4,
                y: centerY + Math.sin(angle) * radius * 0.8
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

        // Draw edges
        edges.forEach(edge => {
            const sourcePos = nodePositions[edge.source];
            const targetPos = nodePositions[edge.target];
            if (!sourcePos || !targetPos) return;

            ctx.beginPath();
            ctx.moveTo(sourcePos.x, sourcePos.y);
            ctx.lineTo(targetPos.x, targetPos.y);

            // Edge styling based on type
            if (edge.type === EDGE_TYPES.INTERACTION) {
                ctx.strokeStyle = `rgba(99, 102, 241, ${0.3 + edge.opacity * 0.5})`;
                ctx.lineWidth = 1 + edge.weight * 2;
                if (edge.style === 'dashed') {
                    ctx.setLineDash([5, 5]);
                } else {
                    ctx.setLineDash([]);
                }
            } else {
                ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
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

            // Node shadow
            ctx.shadowColor = color;
            ctx.shadowBlur = isHovered ? 20 : 10;

            // Node circle
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + (isHovered ? 3 : 0), 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Reset shadow
            ctx.shadowBlur = 0;

            // Node border
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Node label
            ctx.fillStyle = '#f1f5f9';
            ctx.font = '11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';

            // Truncate label if too long
            const label = node.label.length > 12
                ? node.label.substring(0, 10) + '...'
                : node.label;
            ctx.fillText(label, pos.x, pos.y + radius + 5);
        });
    }, [nodes, edges, nodePositions, dimensions, hoveredNode]);

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
