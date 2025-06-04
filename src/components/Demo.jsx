import React from 'react';

export default function Demo() {
  return (
    <div className="bg-gray-50 py-24 sm:py-32" id="demo">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">See it in action</h2>
          <p className="mt-6 text-lg leading-8 text-gray-600">
            Watch how our AI system analyzes surveillance footage in real-time
          </p>
        </div>
        <div className="mt-16 flow-root sm:mt-24">
          <div className="relative rounded-xl bg-gray-900 p-8">
            <div className="aspect-w-16 aspect-h-9">
              <iframe
                className="w-full aspect-video rounded-lg shadow-xl"
                src="https://www.youtube.com/embed/37MydYtoo4U?si=q2ORMya2w-RSUT6n"
                title="Surveillance Video Summarizer Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}