import React, { useEffect, useState } from 'react';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart, ArcElement, Tooltip, Legend } from 'chart.js';
import { CategoryScale, LinearScale, BarElement } from 'chart.js';

Chart.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Color mapping for better display
const colorNameToCss = {
  'black': '#000000',
  'white': '#ffffff',
  'silver': '#c0c0c0',
  'gray': '#808080',
  'red': '#ff0000',
  'blue': '#0000ff',
  'green': '#008000',
  'yellow': '#ffff00',
  'orange': '#ffa500',
  'purple': '#800080',
  'pink': '#ffc0cb',
  'brown': '#a52a2a',
  'gold': '#ffd700',
  'navy': '#000080',
  'maroon': '#800000',
  'olive': '#808000',
  'lime': '#00ff00',
  'aqua': '#00ffff',
  'teal': '#008080',
  'fuchsia': '#ff00ff'
};

function Dashboard() {
  const [dashboardSummary, setDashboardSummary] = useState(null);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Video states
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [progressData, setProgressData] = useState(null);
  const [progressInterval, setProgressInterval] = useState(null);
  
  // Video Explorer Table states
  const [searchTerm, setSearchTerm] = useState('');
  const [filterBrand, setFilterBrand] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });
  const [showInsightsModal, setShowInsightsModal] = useState(false);
  const [selectedVideoInsights, setSelectedVideoInsights] = useState(null);
  
  // Optional Filters states
  const [filterColor, setFilterColor] = useState('');
  const [filterVehicleType, setFilterVehicleType] = useState('');
  const [filterRegion, setFilterRegion] = useState('');
  const [sponsorBrand, setSponsorBrand] = useState('');
  const [showHighExposureOnly, setShowHighExposureOnly] = useState(false);
  
  // Final Detections Table states
  const [detectionsSearchTerm, setDetectionsSearchTerm] = useState('');
  const [detectionsCurrentPage, setDetectionsCurrentPage] = useState(1);
  const [detectionsPerPage] = useState(10);
  const [showDetectionsImageModal, setShowDetectionsImageModal] = useState(false);
  const [selectedDetectionImage, setSelectedDetectionImage] = useState(null);
  const [vehicles, setVehicles] = useState([]);
  const [vehiclesLoading, setVehiclesLoading] = useState(false);

  useEffect(() => {
    fetchDashboard();
    fetchVehicles();
  }, []);

  const fetchDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/dashboard`);
      if (!res.ok) throw new Error('Failed to fetch dashboard data');
      const json = await res.json();
      setDashboardSummary(json.dashboardSummary || null);
      setVideos(json.videos || []);
    } catch (err) {
      setError(err.message || 'Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  };

  const fetchVehicles = async () => {
    setVehiclesLoading(true);
    try {
      // Build query parameters from filters
      const params = new URLSearchParams();
      if (detectionsSearchTerm) params.append('search', detectionsSearchTerm);
      if (filterBrand) params.append('brand', filterBrand);
      if (filterColor) params.append('color', filterColor);
      if (filterVehicleType) params.append('vehicleType', filterVehicleType);
      if (filterRegion) params.append('region', filterRegion);
      if (sponsorBrand) params.append('sponsorBrand', sponsorBrand);
      if (showHighExposureOnly) params.append('highExposureOnly', 'true');
      
      const res = await fetch(`${API_BASE_URL}/api/vehicles?${params.toString()}`);
      if (!res.ok) throw new Error('Failed to fetch vehicles data');
      const json = await res.json();
      setVehicles(json.vehicles || []);
    } catch (err) {
      console.error('Error fetching vehicles:', err);
      setVehicles([]);
    } finally {
      setVehiclesLoading(false);
    }
  };

  // Refetch vehicles when filters change
  useEffect(() => {
    fetchVehicles();
  }, [detectionsSearchTerm, filterBrand, filterColor, filterVehicleType, filterRegion, sponsorBrand, showHighExposureOnly]);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadMsg('');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMsg('Please select a video file.');
      return;
    }
    setUploading(true);
    setUploadMsg('');
    setProgressData(null);
    setCurrentJobId(null);
    
    const formData = new FormData();
    formData.append('video', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/api/process_video`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (response.ok && result.status === 'processing') {
        setCurrentJobId(result.job_id);
        setUploadMsg('Video processing started. Monitoring progress...');
        startProgressPolling(result.job_id);
      } else {
        setUploadMsg(result.error || result.stderr || 'Processing failed.');
        setUploading(false);
      }
    } catch (err) {
      setUploadMsg('Upload failed. Please try again.');
      setUploading(false);
    }
  };

  const startProgressPolling = (jobId) => {
    // Clear any existing interval
    if (progressInterval) {
      clearInterval(progressInterval);
    }
    
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/progress/${jobId}`);
        const progressData = await response.json();
        
        setProgressData(progressData);
        
        if (progressData.status === 'completed' || progressData.status === 'error' || progressData.status === 'redis_unavailable') {
          clearInterval(interval);
          setProgressInterval(null);
          setUploading(false);
          setCurrentJobId(null);
          
          if (progressData.status === 'completed') {
            setUploadMsg('Video processed successfully!');
            await fetchDashboard();
            await fetchVehicles();
          } else if (progressData.status === 'redis_unavailable') {
            setUploadMsg('Video processing started (progress tracking unavailable). Please wait...');
            // Continue polling for a longer time since we can't track progress
            setTimeout(() => {
              clearInterval(interval);
              setProgressInterval(null);
              setUploading(false);
              setCurrentJobId(null);
              setUploadMsg('Video processing completed (progress tracking was unavailable).');
              fetchDashboard();
              fetchVehicles();
            }, 300000); // Wait 5 minutes then assume completion
          } else {
            setUploadMsg(progressData.message || 'Processing failed.');
          }
        }
      } catch (err) {
        console.error('Error polling progress:', err);
      }
    }, 1000); // Poll every second
    
    setProgressInterval(interval);
  };

  const stopProgressPolling = () => {
    if (progressInterval) {
      clearInterval(progressInterval);
      setProgressInterval(null);
    }
  };

  // Cleanup interval on component unmount
  useEffect(() => {
    return () => {
      stopProgressPolling();
    };
  }, []);

  const handleSort = (key) => {
    setSortConfig(prev => ({
      key,
      direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const getTopBrands = (brandLogoStats) => {
    if (!brandLogoStats) return [];
    return Object.entries(brandLogoStats)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([brand]) => brand);
  };

  const getColorDistribution = (colorDistribution) => {
    if (!colorDistribution) return [];
    return Object.entries(colorDistribution)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3);
  };

  const getTopRegion = (estimatedRegions) => {
    if (!estimatedRegions || Object.keys(estimatedRegions).length === 0) return 'Unknown';
    return Object.entries(estimatedRegions)
      .sort(([,a], [,b]) => b - a)[0][0];
  };

  // Dynamic filter options from data
  const getUniqueBrands = () => {
    const brands = new Set();
    videos.forEach(video => {
      if (video.brandLogoStats) {
        Object.keys(video.brandLogoStats).forEach(brand => {
          if (brand && brand.toLowerCase() !== 'unknown') {
            brands.add(brand);
          }
        });
      }
    });
    return Array.from(brands).sort();
  };

  const getUniqueColors = () => {
    const colors = new Set();
    videos.forEach(video => {
      if (video.colorDistribution) {
        Object.keys(video.colorDistribution).forEach(color => {
          if (color && color.toLowerCase() !== 'unknown') {
            colors.add(color);
          }
        });
      }
    });
    return Array.from(colors).sort();
  };

  const getUniqueVehicleTypes = () => {
    const types = new Set();
    videos.forEach(video => {
      if (video.vehicleTypes) {
        Object.keys(video.vehicleTypes).forEach(type => {
          if (type && type.toLowerCase() !== 'unknown') {
            types.add(type);
          }
        });
      }
    });
    return Array.from(types).sort();
  };

  const getUniqueRegions = () => {
    const regions = new Set();
    videos.forEach(video => {
      if (video.numberPlateSummary?.estimatedRegions) {
        Object.keys(video.numberPlateSummary.estimatedRegions).forEach(region => {
          if (region && region.toLowerCase() !== 'unknown') {
            regions.add(region);
          }
        });
      }
    });
    return Array.from(regions).sort();
  };

  const filteredAndSortedVideos = videos
    .filter(video => {
      const matchesSearch = video.filename?.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          video.videoId?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesBrand = !filterBrand || 
                          getTopBrands(video.brandLogoStats).some(brand => 
                            brand.toLowerCase().includes(filterBrand.toLowerCase())
                          );
      const matchesColor = !filterColor || 
                          getColorDistribution(video.colorDistribution).some(([color]) => 
                            color.toLowerCase().includes(filterColor.toLowerCase())
                          );
      const matchesVehicleType = !filterVehicleType || 
                                Object.keys(video.vehicleTypes || {}).some(type => 
                                  type.toLowerCase().includes(filterVehicleType.toLowerCase())
                                );
      const matchesRegion = !filterRegion || 
                           getTopRegion(video.numberPlateSummary?.estimatedRegions)?.toLowerCase().includes(filterRegion.toLowerCase());
      const matchesSponsorBrand = !sponsorBrand || 
                                 getTopBrands(video.brandLogoStats).some(brand => 
                                   brand.toLowerCase().includes(sponsorBrand.toLowerCase())
                                 );
      
      // High exposure filter (top 10% by car count)
      const allCarCounts = videos.map(v => v.totalCarsDetected || 0).sort((a, b) => b - a);
      const top10PercentIndex = Math.floor(allCarCounts.length * 0.1);
      const highExposureThreshold = allCarCounts[top10PercentIndex] || 0;
      const matchesHighExposure = !showHighExposureOnly || (video.totalCarsDetected || 0) >= highExposureThreshold;
      
      return matchesSearch && matchesBrand && matchesColor && matchesVehicleType && 
             matchesRegion && matchesSponsorBrand && matchesHighExposure;
    })
    .sort((a, b) => {
      if (!sortConfig.key) return 0;
      
      let aVal = a[sortConfig.key];
      let bVal = b[sortConfig.key];
      
      if (sortConfig.key === 'filename') {
        aVal = a.filename || '';
        bVal = b.filename || '';
      } else if (sortConfig.key === 'durationSec') {
        aVal = a.durationSec || 0;
        bVal = b.durationSec || 0;
      } else if (sortConfig.key === 'totalCarsDetected') {
        aVal = a.totalCarsDetected || 0;
        bVal = b.totalCarsDetected || 0;
      } else if (sortConfig.key === 'avgCarsPerFrame') {
        aVal = a.mostActiveSegment?.avgCarsPerFrame || 0;
        bVal = b.mostActiveSegment?.avgCarsPerFrame || 0;
      }
      
      if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });

  const handleViewInsights = (video) => {
    setSelectedVideoInsights(video);
    setShowInsightsModal(true);
  };

  // Chart data preparation for video insights
  const prepareMotionStatsChart = (motionStats) => {
    if (!motionStats) return null;
    
    return {
      labels: ['Moving Vehicles', 'Static Vehicles'],
      datasets: [{
        data: [motionStats.movingVehicles || 0, motionStats.staticVehicles || 0],
        backgroundColor: ['#10b981', '#6b7280'],
        borderColor: '#1e293b',
        borderWidth: 2,
      }]
    };
  };

  const prepareRegionChart = (estimatedRegions) => {
    if (!estimatedRegions || Object.keys(estimatedRegions).length === 0) return null;
    
    const regions = Object.keys(estimatedRegions);
    const counts = Object.values(estimatedRegions);
    
    return {
      labels: regions,
      datasets: [{
        label: 'Plate Count',
        data: counts,
        backgroundColor: '#3b82f6',
        borderColor: '#1e293b',
        borderWidth: 1,
      }]
    };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-xl text-gray-300">Loading...</div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-xl text-red-400">{error}</div>
      </div>
    );
  }
  if (!dashboardSummary) {
    return null;
  }

  // Extract KPI values
  const totalVideos = dashboardSummary.totalVideosProcessed || 0;
  const totalCars = dashboardSummary.totalCarsDetected || 0;
  const distinctPlates = dashboardSummary.distinctPlatesDetected || 0;
  const totalDurationMin = dashboardSummary.totalDurationSec ? (dashboardSummary.totalDurationSec / 60).toFixed(1) : '0.0';
  const avgCarsPerSec = (dashboardSummary.totalDurationSec && dashboardSummary.totalCarsDetected)
    ? (dashboardSummary.totalCarsDetected / dashboardSummary.totalDurationSec).toFixed(2)
    : '0.00';

  const summaryCards = [
    { label: 'Total Videos Processed', value: totalVideos, color: 'bg-blue-700' },
    { label: 'Total Cars Detected', value: totalCars, color: 'bg-green-700' },
    { label: 'Distinct Plates', value: distinctPlates, color: 'bg-purple-700' },
    { label: 'Total Duration (min)', value: totalDurationMin, color: 'bg-yellow-700' },
    { label: 'Avg Cars per Sec', value: avgCarsPerSec, color: 'bg-pink-700' },
  ];

  // Chart data preparation
  const topBrands = dashboardSummary.topBrandsOverall || [];
  const topColors = dashboardSummary.topColorsOverall || [];
  const topModels = dashboardSummary.topModelsOverall || [];

  // Brand Exposure Chart (Horizontal Bar)
  const brandData = {
    labels: topBrands.map(brand => brand.name || brand),
    datasets: [{
      label: 'Brand Exposure',
      data: topBrands.map(brand => brand.count || 1), // Use real count data
      backgroundColor: topBrands.map((brand, index) => 
        '#4ecdc4'
      ),
      borderColor: '#1e293b',
      borderWidth: 1,
    }]
  };

  const brandOptions = {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.parsed.x;
            const total = brandData.datasets[0].data.reduce((sum, val) => sum + val, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${context.label}: ${value} (${percentage}%)`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        ticks: { color: '#cbd5e1' },
        grid: { color: '#334155' }
      },
      y: {
        ticks: { color: '#cbd5e1' },
        grid: { color: '#334155' }
      }
    }
  };

  // Color Distribution Chart (Pie)
  const colorData = {
    labels: topColors.map(color => color.name || color),
    datasets: [{
      data: topColors.map(color => color.count || 1), // Use real count data
      backgroundColor: topColors.map(color => colorNameToCss[(color.name || color).toLowerCase()] || '#808080'),
      borderColor: '#1e293b',
      borderWidth: 2,
    }]
  };

  const colorOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#fff',
          font: { size: 12 },
          generateLabels: (chart) => {
            const data = chart.data;
            if (!data.labels) return [];
            return data.labels.map((label, i) => {
              const value = data.datasets[0].data[i];
              return {
                text: `${label} (${value})`,
                fillStyle: data.datasets[0].backgroundColor[i],
                strokeStyle: data.datasets[0].borderColor,
                lineWidth: data.datasets[0].borderWidth,
                hidden: false,
                index: i,
                fontColor: '#fff',
              };
            });
          }
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed;
            const total = colorData.datasets[0].data.reduce((sum, val) => sum + val, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  };

  // Model Breakdown Chart (Stacked Bars)
  const modelData = {
    labels: topModels.map(model => model.name || 'Uknown'),
    datasets: [{
      label: 'Model Count',
      data: topModels.map(model => model.count || 1), // Use real count data
      backgroundColor: '#8b5cf6',
      borderColor: '#1e293b',
      borderWidth: 1,
    }]
  };

  const modelOptions = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.parsed.y;
            const total = modelData.datasets[0].data.reduce((sum, val) => sum + val, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${context.label}: ${value} (${percentage}%)`;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#cbd5e1', maxRotation: 45 },
        grid: { color: '#334155' }
      },
      y: {
        beginAtZero: true,
        ticks: { color: '#cbd5e1' },
        grid: { color: '#334155' }
      }
    }
  };

  return (
    <div className="p-6 font-sans bg-gray-900 min-h-screen text-gray-100">
      <h1 className="mt-8 mb-12 text-5xl font-bold">
        Car Detection & Tracking Dashboard
      </h1>

      {/* Video Upload UI */}
      <div className="mb-8 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <h2 className="text-xl font-semibold text-blue-400 mb-4">Upload Video</h2>
        <div className="flex flex-col sm:flex-row gap-4">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100"
          />
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-semibold"
          >
            {uploading ? 'Processing...' : 'Upload'}
          </button>
        </div>
        
        {/* Progress Bar */}
        {uploading && progressData && (
          <div className="mt-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-300">{progressData.message}</span>
              <span className="text-sm text-blue-400">{progressData.progress}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progressData.progress}%` }}
              ></div>
            </div>
            {progressData.status === 'processing' && (
              <div className="mt-2 text-xs text-gray-400">
                Job ID: {currentJobId}
              </div>
            )}
          </div>
        )}
        
        {uploadMsg && !uploading && (
          <div className={`mt-4 p-3 rounded ${uploadMsg.includes('success') ? 'bg-green-700' : 'bg-red-700'} text-white`}>
            {uploadMsg}
          </div>
        )}
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 mb-12">
        {summaryCards.map(card => (
          <div key={card.label} className={`rounded-lg shadow-lg p-6 border border-gray-700 flex flex-col items-center ${card.color}`}>
            <div className="text-4xl font-extrabold mb-2">{card.value}</div>
            <div className="text-md font-semibold text-gray-100 text-center">{card.label}</div>
          </div>
        ))}
      </div>

      {/* Top Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        {/* Brand Exposure Chart */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
          <h2 className="text-lg font-semibold text-blue-400 mb-4">Brand Exposure</h2>
          <div className="h-64">
            <Bar data={brandData} options={brandOptions} />
          </div>
        </div>

        {/* Color Distribution Chart */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
          <h2 className="text-lg font-semibold text-blue-400 mb-4">Color Distribution</h2>
          <div className="h-64 flex justify-center">
            <Pie data={colorData} options={colorOptions} />
          </div>
        </div>

        {/* Model Breakdown Chart */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
          <h2 className="text-lg font-semibold text-blue-400 mb-4">Model Breakdown</h2>
          <div className="h-64">
            <Bar data={modelData} options={modelOptions} />
          </div>
        </div>
      </div>

      {/* Optional Filters / Insights */}
      <div className="mb-8 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
          <h2 className="text-xl font-semibold text-blue-400">Filters & Insights</h2>
          <div className="flex flex-wrap gap-2 mt-4 md:mt-0">
            <button
              onClick={() => {
                setFilterBrand('');
                setFilterColor('');
                setFilterVehicleType('');
                setFilterRegion('');
                setSponsorBrand('');
                setShowHighExposureOnly(false);
              }}
              className="px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded text-sm"
            >
              Clear All Filters
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Brand Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Brand</label>
            <select
              value={filterBrand}
              onChange={(e) => setFilterBrand(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
            >
              <option value="">All Brands</option>
              {getUniqueBrands().map(brand => (
                <option key={brand} value={brand}>{brand}</option>
              ))}
            </select>
          </div>

          {/* Color Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Color</label>
            <select
              value={filterColor}
              onChange={(e) => setFilterColor(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
            >
              <option value="">All Colors</option>
              {getUniqueColors().map(color => (
                <option key={color} value={color}>{color}</option>
              ))}
            </select>
          </div>

          {/* Vehicle Type Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Vehicle Type</label>
            <select
              value={filterVehicleType}
              onChange={(e) => setFilterVehicleType(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
            >
              <option value="">All Types</option>
              {getUniqueVehicleTypes().map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          {/* Region Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Region</label>
            <select
              value={filterRegion}
              onChange={(e) => setFilterRegion(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
            >
              <option value="">All Regions</option>
              {getUniqueRegions().map(region => (
                <option key={region} value={region}>{region}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Sponsor-Focused Toggles */}
        <div className="border-t border-gray-600 pt-6">
          <h3 className="text-lg font-semibold text-green-400 mb-4">🎯 Sponsor-Focused Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sponsor Brand Toggle */}
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-gray-300">Show only videos with specific brand</label>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={sponsorBrand !== ''}
                    onChange={(e) => {
                      if (!e.target.checked) setSponsorBrand('');
                    }}
                    className="mr-2"
                  />
                </div>
              </div>
              <select
                value={sponsorBrand}
                onChange={(e) => setSponsorBrand(e.target.value)}
                disabled={sponsorBrand === ''}
                className="w-full px-3 py-2 bg-gray-600 border border-gray-500 rounded text-gray-100 text-sm disabled:opacity-50"
              >
                <option value="">Select Brand</option>
                {getUniqueBrands().map(brand => (
                  <option key={brand} value={brand}>{brand}</option>
                ))}
              </select>
            </div>

            {/* High Exposure Toggle */}
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm font-medium text-gray-300">Show top 10% high exposure clips</label>
                  <p className="text-xs text-gray-400 mt-1">Videos with highest car detection counts</p>
                </div>
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    checked={showHighExposureOnly}
                    onChange={(e) => setShowHighExposureOnly(e.target.checked)}
                    className="mr-2"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Active Filters Display */}
        {(filterBrand || filterColor || filterVehicleType || filterRegion || sponsorBrand || showHighExposureOnly) && (
          <div className="mt-6 pt-4 border-t border-gray-600">
            <h4 className="text-sm font-medium text-gray-300 mb-3">Active Filters:</h4>
            <div className="flex flex-wrap gap-2">
              {filterBrand && (
                <span className="px-3 py-1 bg-blue-600 text-white rounded-full text-xs">
                  Brand: {filterBrand} ✕
                </span>
              )}
              {filterColor && (
                <span className="px-3 py-1 bg-green-600 text-white rounded-full text-xs">
                  Color: {filterColor} ✕
                </span>
              )}
              {filterVehicleType && (
                <span className="px-3 py-1 bg-purple-600 text-white rounded-full text-xs">
                  Type: {filterVehicleType} ✕
                </span>
              )}
              {filterRegion && (
                <span className="px-3 py-1 bg-yellow-600 text-white rounded-full text-xs">
                  Region: {filterRegion} ✕
                </span>
              )}
              {sponsorBrand && (
                <span className="px-3 py-1 bg-red-600 text-white rounded-full text-xs">
                  Sponsor: {sponsorBrand} ✕
                </span>
              )}
              {showHighExposureOnly && (
                <span className="px-3 py-1 bg-orange-600 text-white rounded-full text-xs">
                  High Exposure Only ✕
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Video Explorer Table */}
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
              {getUniqueBrands().map(brand => (
                <option key={brand} value={brand}>{brand}</option>
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
              {filteredAndSortedVideos.map((video, index) => (
                <tr key={video.videoId} className={`border-b border-gray-600 ${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'} hover:bg-gray-700`}>
                  <td className="px-4 py-3 text-gray-300">
                    {video.filename ? video.filename.split('/').pop() : 'Unknown'}
                  </td>
                  <td className="px-4 py-3 text-gray-300">
                    {video.durationSec ? `${(video.durationSec / 60).toFixed(1)} min` : 'N/A'}
                  </td>
                  <td className="px-4 py-3 text-gray-300">
                    {video.totalCarsDetected || 0}
                  </td>
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
                        <span key={idx} className="px-2 py-1 text-xs rounded" 
                              style={{ backgroundColor: colorNameToCss[color.toLowerCase()] || '#808080', color: ['white', 'silver'].includes(color.toLowerCase()) ? '#000' : '#fff' }}>
                          {color} ({count})
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-gray-300">
                    {getTopRegion(video.numberPlateSummary?.estimatedRegions)}
                  </td>
                  <td className="px-4 py-3">
                    <button
                      onClick={() => handleViewInsights(video)}
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

      {/* Final Detections Table */}
      <div className="mb-8 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <div className="flex flex-col md:flex-row justify-between items-center mb-6">
          <h2 className="text-xl font-semibold text-blue-400">Final Detections</h2>
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              placeholder="Search detections..."
              value={detectionsSearchTerm}
              onChange={(e) => setDetectionsSearchTerm(e.target.value)}
              className="px-4 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 placeholder-gray-400"
            />
            <select
              value={filterColor} // Reusing filterColor for vehicle type filter
              onChange={(e) => setFilterVehicleType(e.target.value)}
              className="px-4 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100"
            >
              <option value="">All Vehicle Types</option>
              {getUniqueVehicleTypes().map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>
        </div>

        {vehiclesLoading ? (
          <div className="flex justify-center items-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
            <span className="ml-2 text-gray-300">Loading detections...</span>
          </div>
        ) : vehicles.length === 0 ? (
          <div className="flex justify-center items-center py-8">
            <span className="text-gray-400">No vehicle detections found. Try adjusting your filters.</span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div className="text-sm text-gray-400 mb-2">
              Showing {vehicles.length} vehicle detection{vehicles.length !== 1 ? 's' : ''}
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-700 border-b border-gray-600">
                  <th className="px-4 py-3 text-left text-gray-300">Thumbnail</th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('track_id')}>
                    Track ID {sortConfig.key === 'track_id' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('first_seen_sec')}>
                    Start Time {sortConfig.key === 'first_seen_sec' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('last_seen_sec')}>
                    End Time {sortConfig.key === 'last_seen_sec' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('color')}>
                    Color {sortConfig.key === 'color' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('brand')}>
                    Make/Model {sortConfig.key === 'brand' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('license_plate')}>
                    License Plate {sortConfig.key === 'license_plate' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('license_region')}>
                    Region {sortConfig.key === 'license_region' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300 cursor-pointer hover:bg-gray-600" onClick={() => handleSort('dwell_time_seconds')}>
                    Dwell Time {sortConfig.key === 'dwell_time_seconds' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                  <th className="px-4 py-3 text-left text-gray-300">View Details</th>
                </tr>
              </thead>
              <tbody>
                {vehicles.map((vehicle, index) => (
                  <tr key={vehicle.id} className={`border-b border-gray-600 ${index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'} hover:bg-gray-700`}>
                    <td className="px-4 py-3 text-gray-300">
                      {/* Thumbnail */}
                      {vehicle.image_path ? (
                        <img 
                          src={`${API_BASE_URL}/api/car_image?path=${encodeURIComponent(vehicle.image_path)}`}
                          alt="Vehicle"
                          className="w-16 h-16 object-cover rounded-md cursor-pointer"
                          onClick={() => {
                            setSelectedDetectionImage(vehicle);
                            setShowDetectionsImageModal(true);
                          }}
                        />
                      ) : (
                        <div className="w-16 h-16 bg-gray-600 rounded-md flex items-center justify-center text-gray-400 text-sm">🖼️</div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.track_id}
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.first_seen_sec?.toFixed(1)}s
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.last_seen_sec?.toFixed(1)}s
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-1 text-xs rounded" 
                              style={{ backgroundColor: colorNameToCss[vehicle.color?.toLowerCase()] || '#808080', color: ['white', 'silver'].includes(vehicle.color?.toLowerCase())  ? '#000' : '#fff' }}>
                          {vehicle.color}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.brand && vehicle.model ? `${vehicle.brand} ${vehicle.model}` : vehicle.brand || vehicle.model || 'Unknown'}
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.license_plate || '-'}
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {vehicle.license_region || 'Unknown'}
                    </td>
                    <td className="px-4 py-3 text-gray-300">
                      {/* Calculate average confidence if we have multiple confidence scores */}
                      {vehicle.dwell_time_seconds ? `${vehicle.dwell_time_seconds.toFixed(1)}s` : 'N/A'}
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => {
                          setSelectedDetectionImage(vehicle);
                          setShowDetectionsImageModal(true);
                        }}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Enhanced Video Insights Modal */}
      {showInsightsModal && selectedVideoInsights && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80 p-4">
          <div className="relative w-full max-w-6xl mx-auto bg-gray-800 rounded-lg shadow-lg border border-gray-700 max-h-[90vh] overflow-y-auto">
            <button
              className="absolute top-4 right-4 z-10 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow-lg"
              onClick={() => setShowInsightsModal(false)}
            >
              ✕
            </button>
            
            <div className="p-6">
              <h2 className="text-2xl font-bold text-blue-400 mb-6">Video Insights: {selectedVideoInsights.filename?.split('/').pop()}</h2>
              
              {/* Video Player Section */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-300 mb-4">Video Playback</h3>
                <div className="bg-gray-700 rounded-lg p-4">
                  <video
                    className="w-full rounded-lg shadow-lg"
                    controls
                    preload="metadata"
                    style={{ maxHeight: '400px' }}
                  >
                    <source src={`${API_BASE_URL}/api/videos/${selectedVideoInsights.filename}`} type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  <div className="mt-2 text-sm text-gray-300 text-center">
                    <p>Video showing car detection and tracking with bounding boxes and labels</p>
                    <p className="mt-1">Green boxes indicate detected cars with their track IDs</p>
                  </div>
                </div>
              </div>
              
              {/* Video Details Header */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-green-400">{selectedVideoInsights.totalCarsDetected || 0}</div>
                  <div className="text-sm text-gray-300">Cars Detected</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400">{(selectedVideoInsights.durationSec / 60).toFixed(1)}</div>
                  <div className="text-sm text-gray-300">Duration (min)</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-yellow-400">{selectedVideoInsights.mostActiveSegment?.avgCarsPerFrame?.toFixed(2) || 'N/A'}</div>
                  <div className="text-sm text-gray-300">Avg Cars/Frame</div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-purple-400">{selectedVideoInsights.numberPlateSummary?.distinctPlates || 0}</div>
                  <div className="text-sm text-gray-300">Distinct Plates</div>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Brand-Color Matrix */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Brand-Color Matrix</h3>
                  {selectedVideoInsights.brandColorMatrix && Object.keys(selectedVideoInsights.brandColorMatrix).length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-gray-600">
                            <th className="px-3 py-2 text-left text-gray-300">Brand</th>
                            <th className="px-3 py-2 text-left text-gray-300">Colors</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(selectedVideoInsights.brandColorMatrix).map(([brand, colors]) => (
                            <tr key={brand} className="border-b border-gray-600">
                              <td className="px-3 py-2 text-gray-300 font-semibold">{brand}</td>
                              <td className="px-3 py-2">
                                <div className="flex flex-wrap gap-1">
                                  {Object.entries(colors).map(([color, count]) => (
                                    <span key={color} className="px-2 py-1 text-xs rounded" 
                                          style={{ backgroundColor: colorNameToCss[color.toLowerCase()] || '#808080', color: color.toLowerCase() === 'white' ? '#000' : '#fff' }}>
                                      {color} ({count})
                                    </span>
                                  ))}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-gray-400 text-center py-8">No brand-color data available</div>
                  )}
                </div>

                {/* Plate List */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Top Plates</h3>
                  {selectedVideoInsights.numberPlateSummary?.topPlates && selectedVideoInsights.numberPlateSummary.topPlates.length > 0 ? (
                    <div className="space-y-2">
                      {selectedVideoInsights.numberPlateSummary.topPlates.map((plate, index) => (
                        <div key={index} className="flex justify-between items-center bg-gray-600 rounded-lg p-3">
                          <span className="font-mono text-blue-300">{plate.plate}</span>
                          <span className="text-gray-300">{plate.frameCount} frames</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-gray-400 text-center py-8">No plate data available</div>
                  )}
                </div>

                {/* Motion Stats */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Motion Statistics</h3>
                  {selectedVideoInsights.motionStats ? (
                    <div className="h-48">
                      <Pie data={prepareMotionStatsChart(selectedVideoInsights.motionStats)} options={{
                        responsive: true,
                        plugins: {
                          legend: {
                            position: 'bottom',
                            labels: { color: '#fff', font: { size: 12 } }
                          }
                        }
                      }} />
                    </div>
                  ) : (
                    <div className="text-gray-400 text-center py-8">No motion data available</div>
                  )}
                </div>

                {/* Most Active Segment */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Most Active Segment</h3>
                  {selectedVideoInsights.mostActiveSegment ? (
                    <div className="space-y-4">
                      <div className="bg-gray-600 rounded-lg p-4">
                        <div className="text-lg font-semibold text-green-400">
                          {selectedVideoInsights.mostActiveSegment.startTimeSec?.toFixed(1)}s - {selectedVideoInsights.mostActiveSegment.endTimeSec?.toFixed(1)}s
                        </div>
                        <div className="text-sm text-gray-300">Time Range</div>
                      </div>
                      <div className="bg-gray-600 rounded-lg p-4">
                        <div className="text-lg font-semibold text-blue-400">
                          {selectedVideoInsights.mostActiveSegment.avgCarsPerFrame?.toFixed(2)}
                        </div>
                        <div className="text-sm text-gray-300">Avg Cars per Frame</div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-gray-400 text-center py-8">No segment data available</div>
                  )}
                </div>

                {/* Performance Stats */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Performance Statistics</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-600 rounded-lg p-3">
                      <div className="text-lg font-semibold text-yellow-400">
                        {selectedVideoInsights.averageDetectionConfidence?.toFixed(3) || 'N/A'}
                      </div>
                      <div className="text-xs text-gray-300">Avg Detection Confidence</div>
                    </div>
                    <div className="bg-gray-600 rounded-lg p-3">
                      <div className="text-lg font-semibold text-green-400">
                        {selectedVideoInsights.carVisibilityPercent?.toFixed(1) || 'N/A'}%
                      </div>
                      <div className="text-xs text-gray-300">Car Visibility %</div>
                    </div>
                  </div>
                </div>

                {/* Region Map */}
                <div className="bg-gray-700 rounded-lg p-6">
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Region Distribution</h3>
                  {selectedVideoInsights.numberPlateSummary?.estimatedRegions && Object.keys(selectedVideoInsights.numberPlateSummary.estimatedRegions).length > 0 ? (
                    <div className="h-48">
                      <Bar data={prepareRegionChart(selectedVideoInsights.numberPlateSummary.estimatedRegions)} options={{
                        responsive: true,
                        plugins: {
                          legend: { display: false },
                          tooltip: {
                            callbacks: {
                              label: function(context) {
                                return `Plates: ${context.parsed.y}`;
                              }
                            }
                          }
                        },
                        scales: {
                          x: { ticks: { color: '#cbd5e1' }, grid: { color: '#334155' } },
                          y: { beginAtZero: true, ticks: { color: '#cbd5e1' }, grid: { color: '#334155' } }
                        }
                      }} />
                    </div>
                  ) : (
                    <div className="text-gray-400 text-center py-8">No region data available</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Final Detections Image Modal */}
      {showDetectionsImageModal && selectedDetectionImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80 p-4">
          <div className="relative w-full max-w-4xl mx-auto bg-gray-800 rounded-lg shadow-lg border border-gray-700">
            <button
              className="absolute top-4 right-4 z-10 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow-lg"
              onClick={() => setShowDetectionsImageModal(false)}
            >
              ✕
            </button>
            
            <div className="p-6">
              <h2 className="text-2xl font-bold text-blue-400 mb-6">Vehicle Detection Details</h2>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Image Section */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-300 mb-4">Vehicle Image</h3>
                  <div className="bg-gray-700 rounded-lg p-4">
                    {selectedDetectionImage.image_path ? (
                      <img 
                        src={`${API_BASE_URL}/api/car_image?path=${encodeURIComponent(selectedDetectionImage.image_path)}`}
                        alt="Vehicle"
                        className="w-full h-64 object-cover rounded-md"
                      />
                    ) : (
                      <div className="w-full h-64 bg-gray-600 rounded-md flex items-center justify-center text-gray-400">
                        🖼️ No Image Available
                      </div>
                    )}
                  </div>
                </div>

                {/* Metadata Section */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-300 mb-4">Detection Metadata</h3>
                  <div className="space-y-4">
                    <div className="bg-gray-700 rounded-lg p-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="text-sm text-gray-400">Track ID</div>
                          <div className="text-lg font-semibold text-blue-400">{selectedDetectionImage.track_id}</div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Time Range</div>
                          <div className="text-lg font-semibold text-green-400">
                            {selectedDetectionImage.first_seen_sec?.toFixed(1)}s - {selectedDetectionImage.last_seen_sec?.toFixed(1)}s
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Color</div>
                          <div className="text-lg font-semibold text-yellow-400">{selectedDetectionImage.color || 'Unknown'}</div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Make & Model</div>
                          <div className="text-lg font-semibold text-purple-400">
                            {selectedDetectionImage.brand && selectedDetectionImage.model 
                              ? `${selectedDetectionImage.brand} ${selectedDetectionImage.model}`
                              : selectedDetectionImage.brand || selectedDetectionImage.model || 'Unknown'
                            }
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">License Plate</div>
                          <div className="text-lg font-semibold text-orange-400">{selectedDetectionImage.license_plate || 'Not detected'}</div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Region</div>
                          <div className="text-lg font-semibold text-cyan-400">{selectedDetectionImage.license_region || 'Unknown'}</div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Vehicle Type</div>
                          <div className="text-lg font-semibold text-pink-400">{selectedDetectionImage.type || 'Unknown'}</div>
                        </div>
                        <div>
                          <div className="text-sm text-gray-400">Dwell Time</div>
                          <div className="text-lg font-semibold text-indigo-400">
                            {selectedDetectionImage.dwell_time_seconds?.toFixed(1)}s
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Additional Info */}
                    <div className="bg-gray-700 rounded-lg p-4">
                      <h4 className="text-md font-semibold text-gray-300 mb-3">Additional Information</h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Video ID:</span>
                          <span className="text-sm font-semibold text-blue-400">{selectedDetectionImage.video_id}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Motion Status:</span>
                          <span className="text-sm font-semibold text-green-400">
                            {selectedDetectionImage.is_moving ? 'Moving' : 'Static'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Video Duration:</span>
                          <span className="text-sm font-semibold text-yellow-400">
                            {selectedDetectionImage.video_duration ? `${(selectedDetectionImage.video_duration / 60).toFixed(1)} min` : 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm text-gray-400">Video File:</span>
                          <span className="text-sm font-semibold text-purple-400">
                            {selectedDetectionImage.video_filename ? selectedDetectionImage.video_filename.split('/').pop() : 'Unknown'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;