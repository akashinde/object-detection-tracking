import React from 'react';

function SummaryCards({ cards }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 mb-12">
      {cards.map(card => (
        <div
          key={card.label}
          className={`rounded-lg shadow-lg p-6 border border-gray-700 flex flex-col items-center ${card.color}`}
        >
          <div className="text-4xl font-extrabold mb-2">{card.value}</div>
          <div className="text-md font-semibold text-gray-100 text-center">{card.label}</div>
        </div>
      ))}
    </div>
  );
}

export default SummaryCards;
