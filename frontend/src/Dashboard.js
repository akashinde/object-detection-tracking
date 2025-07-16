import React, { useEffect, useState } from 'react';

// Mock data for demonstration
const mockDetections = [
  { track_id: 1, number_plate: 'ABC123', type: 'SUV', model: 'Toyota Fortuner', color: [120, 120, 120] },
  { track_id: 2, number_plate: 'XYZ987', type: 'Sedan', model: 'Honda City', color: [200, 200, 255] },
  { track_id: 3, number_plate: 'LMN456', type: 'SUV', model: 'Toyota Fortuner', color: [100, 100, 100] },
  { track_id: 4, number_plate: 'JKL321', type: 'Hatchback', model: 'Maruti Swift', color: [255, 0, 0] },
  { track_id: 5, number_plate: 'ABC123', type: 'SUV', model: 'Toyota Fortuner', color: [120, 120, 120] },
];

function getModelCounts(detections) {
  const counts = {};
  detections.forEach(d => {
    counts[d.model] = (counts[d.model] || 0) + 1;
  });
  return counts;
}

function Dashboard() {
  const [detections, setDetections] = useState([]);
  const [modelCounts, setModelCounts] = useState({});

  useEffect(() => {
    // In real app, fetch from backend API
    setDetections(mockDetections);
    setModelCounts(getModelCounts(mockDetections));
  }, []);

  return (
    <div style={{ padding: 32 }}>
      <h1>Car Detection Dashboard</h1>
      <h2>Summary</h2>
      <ul>
        <li>Total Cars Detected: {detections.length}</li>
        <li>Unique Models: {Object.keys(modelCounts).length}</li>
      </ul>
      <h2>Model Counts</h2>
      <table border="1" cellPadding="8">
        <thead>
          <tr>
            <th>Model</th>
            <th>Count</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(modelCounts).map(([model, count]) => (
            <tr key={model}>
              <td>{model}</td>
              <td>{count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Detections</h2>
      <table border="1" cellPadding="8">
        <thead>
          <tr>
            <th>Track ID</th>
            <th>Number Plate</th>
            <th>Type</th>
            <th>Model</th>
            <th>Color</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((d, idx) => (
            <tr key={idx}>
              <td>{d.track_id}</td>
              <td>{d.number_plate}</td>
              <td>{d.type}</td>
              <td>{d.model}</td>
              <td>
                <div style={{ width: 30, height: 20, background: `rgb(${d.color.join(',')})`, display: 'inline-block', border: '1px solid #ccc' }} />
                <span style={{ marginLeft: 8 }}>{`rgb(${d.color.join(',')})`}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <h2>Analytics (Car Selling Perspective)</h2>
      <ul>
        <li>Most Popular Model: {Object.entries(modelCounts).sort((a, b) => b[1] - a[1])[0]?.[0]}</li>
        <li>Most Common Type: {(() => {
          const typeCounts = {};
          detections.forEach(d => { typeCounts[d.type] = (typeCounts[d.type] || 0) + 1; });
          return Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
        })()}</li>
        <li>Unique Number Plates: {[...new Set(detections.map(d => d.number_plate))].length}</li>
      </ul>
    </div>
  );
}

export default Dashboard; 