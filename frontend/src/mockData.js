const mockData = {
  cars: [
    {
      track_id: 1,
      label: "car1",
      model: "Toyota Camry",
      color: "red",
      license_plate: "ABC1234",
      track_frame_counts: 320,
      scene_count: 3,
      dwell_time_seconds: 12.8,
      type: "sedan"
    },
    {
      track_id: 2,
      label: "car2", 
      model: "Honda CRV",
      color: "black",
      license_plate: "XYZ5678",
      track_frame_counts: 210,
      scene_count: 2,
      dwell_time_seconds: 8.4,
      type: "SUV"
    },
    {
      track_id: 3,
      label: "car3",
      model: "Ford F-150",
      color: "white",
      license_plate: "DEF9012",
      track_frame_counts: 180,
      scene_count: 1,
      dwell_time_seconds: 7.2,
      type: "truck"
    },
    {
      track_id: 4,
      label: "car4",
      model: "BMW 3 Series",
      color: "blue",
      license_plate: "GHI3456",
      track_frame_counts: 450,
      scene_count: 4,
      dwell_time_seconds: 18.0,
      type: "sedan"
    },
    {
      track_id: 5,
      label: "car5",
      model: "Toyota RAV4",
      color: "silver",
      license_plate: "JKL7890",
      track_frame_counts: 290,
      scene_count: 2,
      dwell_time_seconds: 11.6,
      type: "SUV"
    },
    {
      track_id: 6,
      label: "car6",
      model: "Honda Civic",
      color: "gray",
      license_plate: "MNO1234",
      track_frame_counts: 150,
      scene_count: 1,
      dwell_time_seconds: 6.0,
      type: "sedan"
    },
    {
      track_id: 7,
      label: "car7",
      model: "Chevrolet Silverado",
      color: "black",
      license_plate: "PQR5678",
      track_frame_counts: 220,
      scene_count: 2,
      dwell_time_seconds: 8.8,
      type: "truck"
    },
    {
      track_id: 8,
      label: "car8",
      model: "Nissan Altima",
      color: "white",
      license_plate: "STU9012",
      track_frame_counts: 340,
      scene_count: 3,
      dwell_time_seconds: 13.6,
      type: "sedan"
    }
  ],
  summary: {
    total_cars: 8,
    unique_models: 8,
    unique_license_plates: 8,
    average_dwell_time: 10.8
  },
  demographics: {
    type_distribution: {
      sedan: 4,
      SUV: 2,
      truck: 2
    },
    color_distribution: {
      red: 1,
      black: 2,
      white: 2,
      blue: 1,
      silver: 1,
      gray: 1
    },
    model_distribution: {
      "Toyota Camry": 1,
      "Honda CRV": 1,
      "Ford F-150": 1,
      "BMW 3 Series": 1,
      "Toyota RAV4": 1,
      "Honda Civic": 1,
      "Chevrolet Silverado": 1,
      "Nissan Altima": 1
    }
  }
};

export default mockData; 