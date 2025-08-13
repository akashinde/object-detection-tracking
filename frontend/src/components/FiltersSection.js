import React from 'react';

function FiltersSection({
  filterBrand,
  setFilterBrand,
  filterColor,
  setFilterColor,
  filterVehicleType,
  setFilterVehicleType,
  filterRegion,
  setFilterRegion,
  sponsorBrand,
  setSponsorBrand,
  showHighExposureOnly,
  setShowHighExposureOnly,
  getUniqueBrands,
  getUniqueColors,
  getUniqueVehicleTypes,
  getUniqueRegions,
}) {
  return (
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
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Brand</label>
          <select
            value={filterBrand}
            onChange={(e) => setFilterBrand(e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
          >
            <option value="">All Brands</option>
            {getUniqueBrands().map((brand) => (
              <option key={brand} value={brand}>
                {brand}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Color</label>
          <select
            value={filterColor}
            onChange={(e) => setFilterColor(e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
          >
            <option value="">All Colors</option>
            {getUniqueColors().map((color) => (
              <option key={color} value={color}>
                {color}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Vehicle Type</label>
          <select
            value={filterVehicleType}
            onChange={(e) => setFilterVehicleType(e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
          >
            <option value="">All Types</option>
            {getUniqueVehicleTypes().map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Filter by Region</label>
          <select
            value={filterRegion}
            onChange={(e) => setFilterRegion(e.target.value)}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 text-sm"
          >
            <option value="">All Regions</option>
            {getUniqueRegions().map((region) => (
              <option key={region} value={region}>
                {region}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="border-t border-gray-600 pt-6">
        <h3 className="text-lg font-semibold text-green-400 mb-4">🎯 Sponsor-Focused Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
              {getUniqueBrands().map((brand) => (
                <option key={brand} value={brand}>
                  {brand}
                </option>
              ))}
            </select>
          </div>

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

      {(filterBrand || filterColor || filterVehicleType || filterRegion || sponsorBrand || showHighExposureOnly) && (
        <div className="mt-6 pt-4 border-t border-gray-600">
          <h4 className="text-sm font-medium text-gray-300 mb-3">Active Filters:</h4>
          <div className="flex flex-wrap gap-2">
            {filterBrand && (
              <span className="px-3 py-1 bg-blue-600 text-white rounded-full text-xs">Brand: {filterBrand} ✕</span>
            )}
            {filterColor && (
              <span className="px-3 py-1 bg-green-600 text-white rounded-full text-xs">Color: {filterColor} ✕</span>
            )}
            {filterVehicleType && (
              <span className="px-3 py-1 bg-purple-600 text-white rounded-full text-xs">Type: {filterVehicleType} ✕</span>
            )}
            {filterRegion && (
              <span className="px-3 py-1 bg-yellow-600 text-white rounded-full text-xs">Region: {filterRegion} ✕</span>
            )}
            {sponsorBrand && (
              <span className="px-3 py-1 bg-red-600 text-white rounded-full text-xs">Sponsor: {sponsorBrand} ✕</span>
            )}
            {showHighExposureOnly && (
              <span className="px-3 py-1 bg-orange-600 text-white rounded-full text-xs">High Exposure Only ✕</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default FiltersSection;
