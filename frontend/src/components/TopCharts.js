import React from 'react';
import { Pie, Bar } from 'react-chartjs-2';

function TopCharts({ brandData, brandOptions, colorData, colorOptions, modelData, modelOptions }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
      <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <h2 className="text-lg font-semibold text-blue-400 mb-4">Brand Exposure</h2>
        <div className="h-64">
          <Bar data={brandData} options={brandOptions} />
        </div>
      </div>
      <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <h2 className="text-lg font-semibold text-blue-400 mb-4">Color Distribution</h2>
        <div className="h-64 flex justify-center">
          <Pie data={colorData} options={colorOptions} />
        </div>
      </div>
      <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <h2 className="text-lg font-semibold text-blue-400 mb-4">Model Breakdown</h2>
        <div className="h-64">
          <Bar data={modelData} options={modelOptions} />
        </div>
      </div>
    </div>
  );
}

export default TopCharts;
