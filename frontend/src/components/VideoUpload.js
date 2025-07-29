import React, { useState, useEffect } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function VideoUpload({ onComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [progressData, setProgressData] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [progressInterval, setProgressInterval] = useState(null);

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
    if (progressInterval) {
      clearInterval(progressInterval);
    }
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/progress/${jobId}`);
        const data = await response.json();
        setProgressData(data);
        if (
          data.status === 'completed' ||
          data.status === 'error' ||
          data.status === 'redis_unavailable'
        ) {
          clearInterval(interval);
          setProgressInterval(null);
          setUploading(false);
          setCurrentJobId(null);

          if (data.status === 'completed') {
            setUploadMsg('Video processed successfully!');
            if (onComplete) onComplete();
          } else if (data.status === 'redis_unavailable') {
            setUploadMsg('Video processing started (progress tracking unavailable). Please wait...');
            setTimeout(() => {
              clearInterval(interval);
              setProgressInterval(null);
              setUploading(false);
              setCurrentJobId(null);
              setUploadMsg('Video processing completed (progress tracking was unavailable).');
              if (onComplete) onComplete();
            }, 300000);
          } else {
            setUploadMsg(data.message || 'Processing failed.');
          }
        }
      } catch (err) {
        console.error('Error polling progress:', err);
      }
    }, 1000);

    setProgressInterval(interval);
  };

  useEffect(() => {
    return () => {
      if (progressInterval) clearInterval(progressInterval);
    };
  }, [progressInterval]);

  return (
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
            <div className="mt-2 text-xs text-gray-400">Job ID: {currentJobId}</div>
          )}
        </div>
      )}
      {uploadMsg && !uploading && (
        <div className={`mt-4 p-3 rounded ${uploadMsg.includes('success') ? 'bg-green-700' : 'bg-red-700'} text-white`}>
          {uploadMsg}
        </div>
      )}
    </div>
  );
}

export default VideoUpload;
