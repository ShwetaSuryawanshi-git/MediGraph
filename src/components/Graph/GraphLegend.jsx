// File: MediGraph/src/components/Graph/GraphLegend.jsx
// Path: /Users/shweta/Desktop/major project fy/MediGraph/src/components/Graph/GraphLegend.jsx

/**
 * Graph Legend Component
 * Displays legend for node types in the graph
 */

import React from 'react';

const GraphLegend = () => {
    return (
        <div className="graph-legend">
            <div className="legend-item">
                <div className="legend-dot legend-dot-drug"></div>
                <span>Drug</span>
            </div>
            <div className="legend-item">
                <div className="legend-dot legend-dot-target"></div>
                <span>Target</span>
            </div>
            <div className="legend-item">
                <div className="legend-dot legend-dot-disease"></div>
                <span>Disease</span>
            </div>
        </div>
    );
};

export default GraphLegend;
