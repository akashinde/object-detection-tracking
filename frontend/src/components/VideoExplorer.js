import React from 'react';

function VideoExplorer({
  videos,
  searchTerm,
  setSearchTerm,
  filterBrand,
  setFilterBrand,
  getUniqueBrands,
  colorMap,
  sortConfig,
  handleSort,
  getTopBrands,
  getColorDistribution,
  getTopRegion,
  onViewInsights,
}) {
  return (
    <div className="mb-8 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
      <div className="flex flex-col md:flex-row justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-blue-400">Video Explorer</h2>
        <div className="flex flex-col sm:flex-row gap-4">
          <input
            type="text"
            placeholder="Search videos..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 placeholder-gray-400"
          />
          <select
            value={filterBrand}
            onChange={(e) => setFilterBrand(e.target.value)}
            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100"
          >
            <option value="">All Brands</option>
            {getUniqueBrands().map((brand) => (
              <option key={brand} value={brand}>
                {brand}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-700 border-b border-gray-600">
              <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('filename')}>
                File Name {sortConfig.key === 'filename' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('durationSec')}>
                Duration {sortConfig.key === 'durationSec' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('totalCarsDetected')}>
                Cars Detected {sortConfig.key === 'totalCarsDetected' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('avgCarsPerFrame')}>
                Avg Cars/sec {sortConfig.key === 'avgCarsPerFrame' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-4 py-3 text-left text-gray-300">Brand Logos</th>
              <th className="px-4 py-3 text-left text-gray-300">Color Distribution</th>
              <th className="px-4 py-3 text-left text-gray-300">Regions</th>
              <th className="px-4 py-3 text-left text-gray-300">Action</th>
            </tr>
          </thead>
          <tbody>
            {videos.map((video, index) => (
              <tr key={video.videoId} className={`border-b border-gray-600 ${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'} hover:bg-gray-700`}>
                <td className="px-4 py-3 text-gray-300">
                  {video.filename ? video.filename.split('/').pop() : 'Unknown'}
                </td>
                <td className="px-4 py-3 text-gray-300">
                  {video.durationSec ? `${(video.durationSec / 60).toFixed(1)} min` : 'N/A'}
                </td>
                <td className="px-4 py-3 text-gray-300">{video.totalCarsDetected || 0}</td>
                <td className="px-4 py-3 text-gray-300">
                  {video.mostActiveSegment?.avgCarsPerFrame?.toFixed(2) || 'N/A'}
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {getTopBrands(video.brandLogoStats).map((brand, idx) => (
                      <span key={idx} className="px-2 py-1 bg-blue-600 text-white text-xs rounded">
                        {brand}
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {getColorDistribution(video.colorDistribution).map(([color, count], idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 text-xs rounded"
                        style={{
                          backgroundColor: colorMap[color.toLowerCase()] || '#808080',
                          color: ['white', 'silver'].includes(color.toLowerCase()) ? '#000' : '#fff',
                        }}
                      >
                        {color} ({count})
                      </span>
                    ))}
                  </div>
                </td>
                <td className="px-4 py-3 text-gray-300">{getTopRegion(video.numberPlateSummary?.estimatedRegions)}</td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => onViewInsights(video)}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-xs"
                  >
                    View Insights
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default VideoExplorer;
