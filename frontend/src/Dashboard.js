import React, { useEffect, useState } from 'react';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart, ArcElement, Tooltip, Legend } from 'chart.js';
import { CategoryScale, LinearScale, BarElement } from 'chart.js';

Chart.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [videoList, setVideoList] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [showOverlay, setShowOverlay] = useState(false);
  const [processing, setProcessing] = useState(false);
  // Remove jobId and pollingRef state
  // const [jobId, setJobId] = useState(null);
  // const pollingRef = useRef(null);
  const [showImageModal, setShowImageModal] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState(null);

  useEffect(() => {
    fetchData();
    fetchSummary();
    fetchVideos();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/cars`);
      if (!res.ok) throw new Error('Failed to fetch car data');
      const json = await res.json();
      const cars = json.cars || [];
      setData(prev => ({ ...(prev || {}), cars }));
    } catch (err) {
      setError(err.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/summary`);
      if (!res.ok) throw new Error('Failed to fetch summary');
      const summary = await res.json();
      // Convert color_counts_json to demographics.color_distribution
      setData(prev => ({
        ...(prev || {}),
        summary: {
          total_cars: summary.total_cars,
          unique_models: summary.unique_models,
          unique_license_plates: summary.unique_license_plates,
          average_dwell_time: summary.average_dwell_time,
          peak_hour_range: summary.peak_hour_range,
          peak_hour_count: summary.peak_hour_count
        },
        demographics: {
          ...(prev && prev.demographics ? prev.demographics : {}),
          color_distribution: summary.color_counts_json || {},
        }
      }));
    } catch (err) {
      // fallback: do nothing
    }
  };

  const fetchVideos = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/videos`);
      const json = await res.json();
      setVideoList(json.videos || []);
    } catch (err) {
      setVideoList([]);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadMsg('');
  };

  // Remove pollJobStatus function
  // const pollJobStatus = (jobId) => {
  //   setProcessing(true);
  //   pollingRef.current = setInterval(async () => {
  //     try {
  //       const res = await fetch(`${API_BASE_URL}/api/job_status/${jobId}`);
  //       const json = await res.json();
  //       if (json.status === 'completed') {
  //         setUploadMsg('Video processed successfully!');
  //         setProcessing(false);
  //         setUploading(false);
  //         setJobId(null);
  //         clearInterval(pollingRef.current);
  //         await fetchVideos();
  //       } else if (json.status === 'failed') {
  //         setUploadMsg('Processing failed.');
  //         setProcessing(false);
  //         setUploading(false);
  //         setJobId(null);
  //         clearInterval(pollingRef.current);
  //       } else if (json.status === 'queued') {
  //         setUploadMsg('Processing... (queued)');
  //       } else {
  //         setUploadMsg('Processing...');
  //       }
  //     } catch (err) {
  //       setUploadMsg('Error checking processing status.');
  //       setProcessing(false);
  //       setUploading(false);
  //       setJobId(null);
  //       clearInterval(pollingRef.current);
  //     }
  //   }, 2000);
  // };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMsg('Please select a video file.');
      return;
    }
    setUploading(true);
    setProcessing(false);
    setUploadMsg('');
    // setJobId(null);
    const formData = new FormData();
    formData.append('video', selectedFile);
    try {
      const response = await fetch(`${API_BASE_URL}/api/process_video`, {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();
      if (response.ok && result.status === 'success') {
        setUploadMsg('Video processed successfully!');
        setUploading(false);
        await fetchData();
        await fetchVideos();
      } else {
        setUploadMsg(result.error || result.stderr || 'Processing failed.');
        setUploading(false);
      }
    } catch (err) {
      setUploadMsg('Upload failed. Please try again.');
      setUploading(false);
    }
  };

  const handleVideoSelect = (videoObj) => {
    setSelectedVideo(videoObj);
    setShowOverlay(true);
  };

  const handleCloseOverlay = () => {
    setShowOverlay(false);
    setSelectedVideo(null);
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
  if (!data) {
    return null;
  }

  // Prepare color distribution data for Pie chart
  const colorLabels = data?.demographics?.color_distribution ? Object.keys(data.demographics.color_distribution) : [];
  const colorCounts = data?.demographics?.color_distribution ? Object.values(data.demographics.color_distribution) : [];
  const totalColors = colorCounts.reduce((sum, count) => sum + count, 0);
  // Assign a color palette (dark theme friendly, now mapped to color names)
  const colorNameToCss = {
    white: "#f5f5f5",
    black: "#0f0f0f",
    silver: "#c0c0c0",
    gray: "#808080",
    blue: "#192a60",
    red: "#8a0303",
    green: "#00563f",
    brown: "#654321",
    gold: "#d4af37",
    beige: "#f5f5dc",
    yellow: "#ffd300",
    orange: "#ff8c00",
    maroon: "#800000",
    purple: "#4b0082",
    teal: "#008080",
    pink: "#ffb6c1",
    navy: "#000080",
    cyan: "#00ffff",
  };
  const pieData = {
    labels: colorLabels,
    datasets: [
      {
        data: colorCounts,
        backgroundColor: colorLabels.map(label => colorNameToCss[label.toLowerCase()] || 'dimgray'),
        borderColor: '#1e293b', // dark border
        borderWidth: 2,
      },
    ],
  };
  const pieOptions = {
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#fff', // white for better visibility
          font: { size: 14 },
          // Use a custom render function to set label color to white
          generateLabels: (chart) => {
            const data = chart.data;
            if (!data.labels) return [];
            return data.labels.map((label, i) => {
              const value = data.datasets[0].data[i];
              const percentage = totalColors ? ((value / totalColors) * 100).toFixed(1) : 0;
              return {
                text: `${label} (${percentage}%)`,
                fillStyle: data.datasets[0].backgroundColor[i],
                strokeStyle: data.datasets[0].borderColor,
                lineWidth: data.datasets[0].borderWidth,
                hidden: isNaN(data.datasets[0].data[i]) || chart.getDataVisibility(i) === false,
                index: i,
                fontColor: '#fff', // <-- This is for Chart.js v2, but v3+ uses 'color' above
                font: { size: 14, color: '#fff' }, // Try to enforce white font
              };
            });
          },
          // For Chart.js v3+, use a 'color' function to force white
          color: (context) => '#fff',
        },
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.label || '';
            const value = context.parsed;
            const percentage = totalColors ? ((value / totalColors) * 100).toFixed(1) : 0;
            return `${label}: ${value} (${percentage}%)`;
          }
        }
      }
    },
  };

  // Assign a unique color to each video_id
  const videoIds = Array.from(new Set(data.cars.map(car => car.video_id)));
  const colorKeys = Object.keys(colorNameToCss);
  const videoIdToColor = {};
  videoIds.forEach((vid, idx) => {
    videoIdToColor[vid] = colorNameToCss[colorKeys[idx % colorKeys.length]] || 'lightblue';
  });
  // Prepare dwell time data for Bar chart
  const dwellLabels = data.cars.map(car => car.label);
  const dwellTimes = data.cars.map(car => car.dwell_time_seconds);
  const dwellBarColors = data.cars.map(car => videoIdToColor[car.video_id]);
  const dwellBarData = {
    labels: dwellLabels,
    datasets: [
      {
        label: 'Dwell Time (s)',
        data: dwellTimes,
        backgroundColor: dwellBarColors,
        borderWidth: 2,
      },
    ],
  };
  const dwellBarOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.parsed.y;
            return `Dwell Time: ${value}s`;
          }
        }
      },
      // Add a custom legend for video colors
      customLegend: true,
    },
    scales: {
      x: {
        ticks: { color: '#cbd5e1', font: { size: 14 } },
        grid: { color: '#334155' },
      },
      y: {
        beginAtZero: true,
        ticks: { color: '#cbd5e1', font: { size: 14 } },
        grid: { color: '#334155' },
        title: {
          display: true,
          text: 'Seconds',
          color: '#cbd5e1',
          font: { size: 16 }
        }
      },
    },
  };

  // Prepare summary cards data
  const totalCars = data?.summary?.total_cars ?? 0;
  // Unreadable plates: license_plate is empty
  const unreadablePlates = data?.cars ? data.cars.filter(car => !car.license_plate || car.license_plate.trim() === '').length : 0;
  const unreadablePlatesPercent = totalCars > 0 ? ((unreadablePlates / totalCars) * 100).toFixed(1) : '0.0';
  // Most common make/model (show 'NA')
  const mostCommonMakeModel = 'NA';
  // Peak traffic time (show 'NA' unless you have timestamps)
  const peakTrafficTime = data?.summary?.peak_hour_range || 'NA';

  const summaryCards = [
    {
      label: 'Total Cars',
      value: data?.summary?.total_cars ?? 0,
      color: 'bg-blue-700',
    },
    {
      label: 'Peak Traffic Time',
      value: data?.summary?.peak_hour_range || 'NA',
      color: 'bg-blue-700',
    },
    {
      label: 'Unreadable Plates',
      value: totalCars > 0 ? `${unreadablePlatesPercent}%` : '0.0%',
      color: 'bg-green-700',
    },
    {
      label: 'Most Common Make/Model',
      value: mostCommonMakeModel,
      color: 'bg-purple-700',
    },
    {
      label: 'Unique Models',
      value: data?.summary?.unique_models ?? 0,
      color: 'bg-purple-700',
    },
    {
      label: 'Total Number Plates',
      value: data?.summary?.unique_license_plates ?? 0,
      color: 'bg-green-700',
    },
    {
      label: 'Avg. Dwell Time (s)',
      value: data?.summary?.average_dwell_time ?? 0,
      color: 'bg-yellow-700',
    },
  ];

  return (
    <div className="p-6 font-sans bg-gray-900 min-h-screen text-gray-100">
      <h1 className="mt-8 mb-12 text-5xl font-bold">
        Car Detection & Tracking Dashboard
      </h1>

      {/* Video Upload UI */}
      <div className="mb-8 flex flex-col md:flex-row items-center gap-4 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <input
          type="file"
          accept="video/*"
          className="block w-full md:w-auto text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          onChange={handleFileChange}
          disabled={uploading || processing}
        />
        <button
          onClick={handleUpload}
          disabled={uploading || processing}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded disabled:opacity-50 flex items-center gap-2"
        >
          {(uploading || processing) && (
            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
            </svg>
          )}
          {uploading || processing ? 'Uploading & Processing...' : 'Upload & Process Video'}
        </button>
        {uploadMsg && (
          <span className={`ml-4 text-sm ${uploadMsg.includes('success') ? 'text-green-400' : uploadMsg.includes('Processing') ? 'text-yellow-400' : 'text-red-400'}`}>{uploadMsg}</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Summary Cards (now includes high-level KPIs) */}
        <div className="mb-10 grid grid-cols-1 md:grid-cols-3 lg:grid-cols-3 gap-6">
          {summaryCards.map(card => {
            return (
              <div
                key={card.label}
                className={`rounded-lg shadow-lg p-6 flex flex-col items-center justify-center border border-gray-700 bg-gray-800`}
              >
                <div className="text-4xl font-bold mb-2">{card.value}</div>
                <div className="text-lg font-semibold text-blue-400">{card.label}</div>
              </div>
            );
          })}
        </div>

        {/* Merged Video List & Player Card */}
        <div className="mb-8 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold text-blue-400 mb-4">Processed Videos</h2>
          {videoList.length === 0 ? (
            <div className="text-gray-400">No processed videos found.</div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-700 border-b border-gray-600">
                  <th className="px-4 py-2 text-left text-gray-300">Filename</th>
                  <th className="px-4 py-2 text-left text-gray-300">Action</th>
                </tr>
              </thead>
              <tbody>
                {videoList.map((videoObj) => (
                  <tr
                    key={videoObj.folder + '-' + videoObj.filename}
                    className={`border-b border-gray-600 hover:bg-gray-700 cursor-pointer`}
                    onClick={() => handleVideoSelect(videoObj)}
                  >
                    <td className="px-4 py-2 text-left">{videoObj.filename}</td>
                    <td className="px-4 py-2 text-left">
                      <button
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
                        onClick={e => { e.stopPropagation(); handleVideoSelect(videoObj); }}
                      >
                        Play
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        {/* Video Overlay Modal */}
        {showOverlay && selectedVideo && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80">
            <div className="relative w-full max-w-4xl mx-4 bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
              <button
                className="absolute top-2 right-2 z-10 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow-lg"
                onClick={handleCloseOverlay}
              >
                Close
              </button>
              <video
                className="w-full rounded-lg shadow-lg"
                controls
                preload="metadata"
                poster="/video-poster.jpg"
                key={selectedVideo.folder + '-' + selectedVideo.filename}
                autoPlay
                style={{ height: '80vh' }}
              >
                <source src={`${API_BASE_URL}/api/videos/${selectedVideo.folder}/${selectedVideo.filename}`} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              <div className="mt-4 text-sm text-gray-300 text-center">
                <p>Video showing car detection and tracking with bounding boxes and labels</p>
                <p className="mt-1">Green boxes indicate detected cars with their track IDs</p>
              </div>
            </div>
          </div>
        )}

      </div>


      {/* Demographics Section */}
      <div className="mb-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dwell Time Bar Chart */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700 flex flex-col items-center">
          <h3 className="text-lg font-semibold text-blue-400 mb-4">
            Car Dwell Time (Seconds)
          </h3>
          <div className="w-full h-64 flex justify-center">
            <Bar data={dwellBarData} options={dwellBarOptions} />
          </div>
          {/* Video color legend */}
          <div className="flex flex-wrap gap-4 mt-4 justify-center">
            {videoIds.map((vid, idx) => {
              let filename = (data.cars.find(car => car.video_id === vid) || {}).video_filename || `Video ${vid}`;
              // Shorten: remove extension and truncate if long
              if (filename.includes('.')) filename = filename.substring(0, filename.lastIndexOf('.'));
              if (filename.length > 16) filename = filename.substring(0, 13) + '...';
              return (
                <div key={vid} className="flex items-center gap-2">
                  <span className="inline-block w-5 h-5 rounded" style={{ backgroundColor: videoIdToColor[vid], border: '1px solid #1e293b' }}></span>
                  <span className="text-gray-300 text-sm">{filename}</span>
                </div>
              );
            })}
          </div>
        </div>
        {/* Car Color Distribution Pie Chart */}
        <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700 flex flex-col items-center">
          <h3 className="text-lg font-semibold text-blue-400 mb-4">
            Car Color Distribution
          </h3>
          <div className="w-full flex justify-center">
            <div className="w-64 h-64">
              <Pie data={pieData} options={pieOptions} />
            </div>
          </div>
        </div>
      </div>

      {/* Car Details Table */}
      <div className="my-8 bg-gray-800 rounded-lg shadow-lg overflow-hidden border border-gray-700">
        <h2 className="text-xl font-semibold text-blue-400 p-6 pb-4">
          Detected Cars Details
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-700 border-b border-gray-600">
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Car Label</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Model</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Color</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">License Plate</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Plate Conf.</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Type</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Track Frames</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Dwell Time</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Video</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Video ID</th>
                <th className="px-6 py-4 text-left font-semibold text-gray-300">Car Image</th>
              </tr>
            </thead>
            <tbody>
              {data.cars.map((car, index) => (
                <tr 
                  key={car.track_id + '-' + car.video_id} 
                  className={`border-b border-gray-600 ${
                    index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-750'
                  } hover:bg-gray-700 transition-colors text-left`}
                >
                  <td className="px-6 py-4 font-bold text-blue-400 text-left">
                    {car.label}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.model}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.color}
                  </td>
                  <td className="px-6 py-4 font-mono font-bold text-blue-300 text-left">
                    {car.license_plate && car.license_plate.trim() !== '' ? car.license_plate : '-'}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.license_plate_confidence !== undefined && car.license_plate_confidence !== null && car.license_plate_confidence !== '' ? car.license_plate_confidence.toFixed(3) : '-'}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.type}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.track_frame_counts}
                  </td>
                  <td className="px-6 py-4 text-gray-300 text-left">
                    {car.dwell_time_seconds}s
                  </td>
                 <td className="px-6 py-4 text-gray-300 text-left">
                   {car.video_filename}
                 </td>
                 <td className="px-6 py-4 text-gray-300 text-left">
                   {car.video_id}
                 </td>
                <td className="px-6 py-4 text-gray-300 text-left">
                  {car.image_path || (car.video_filename && car.track_id) ? (
                    <button
                      className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded"
                      onClick={() => {
                        // Use video name (without extension) and car track_id
                        let videoName = car.video_filename ? car.video_filename.split('.')[0] : 'unknown_video';
                        let imagePath = `videos/processed/${videoName}/car_${car.track_id}.jpg`;
                        setModalImageUrl(`${API_BASE_URL}/api/car_image?path=${encodeURIComponent(imagePath)}`);
                        setShowImageModal(true);
                      }}
                    >
                      View Image
                    </button>
                    ) : (
                      <span className="text-gray-500">No Image</span>
                  )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Car Image Modal */}
      {showImageModal && modalImageUrl && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80"
          onClick={() => setShowImageModal(false)}
        >
          <div
            className="relative bg-gray-900 rounded-lg shadow-lg p-6 border border-gray-700 flex flex-col items-center"
            onClick={e => e.stopPropagation()}
          >
            <button
              className="absolute top-2 right-2 z-10 px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold rounded shadow-lg"
              onClick={() => setShowImageModal(false)}
            >
              Close
            </button>
            <img style={{height: '500px'}} src={modalImageUrl} alt="Car" className="mt-10 max-w-full max-h-[80vh] rounded-lg shadow-lg" />
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard; 