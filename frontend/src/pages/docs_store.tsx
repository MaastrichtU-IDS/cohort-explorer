 'use client';

import React, { useState, useEffect } from 'react';
import { apiUrl } from '@/utils';
import Link from 'next/link';
import { HardDrive, AlertTriangle, Check, Info, FileText } from 'react-feather'; // Added more icons

export default function DocsStorePage() {
  const [documents, setDocuments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${apiUrl}/docs-api/documents`);
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: `Failed to fetch documents: ${response.status} ${response.statusText}` }));
          throw new Error(errorData.detail || `Failed to fetch documents: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        setDocuments(data);
      } catch (err: any) {
        setError(err.message);
        console.error("Error fetching documents:", err);
        setDocuments([]); // Clear documents on error
      } finally {
        setIsLoading(false);
      }
    };
    fetchDocuments();
  }, []);

  return (
    <main className="flex flex-col items-center justify-start p-8 min-h-screen bg-base-200">
      <div className="w-full max-w-4xl space-y-8">
        <h1 className="text-3xl font-bold text-center mb-8 flex items-center justify-center">
          <HardDrive size={32} className="mr-3" /> Project Documents
        </h1>

        {isLoading && (
          <div className="flex flex-col items-center opacity-70 text-slate-500 mt-[10%]">
            <span className="loading loading-spinner loading-lg mb-4"></span>
            <p>Loading documents...</p>
          </div>
        )}

        {!isLoading && error && (
          <div role="alert" className="alert alert-error mt-4">
            <AlertTriangle size={20} className="mr-2 shrink-0" />
            <span>Error loading documents: {error}. <br/>Please ensure the backend is running, the API is reachable, and the documents_store folder exists at the project root.</span>
          </div>
        )}

        {!isLoading && !error && documents.length === 0 && (
          <div className="text-center text-slate-500 p-10 border-2 border-dashed border-base-300 rounded-lg mt-[5%]">
            <Info size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-xl">No documents found.</p>
            <p className="text-sm mt-2">Upload files to the documents_store directory in the project root on the server.</p>
          </div>
        )}

        {!isLoading && !error && documents.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-6">
            {documents.map((docName) => {
              const isPdf = docName.toLowerCase().endsWith('.pdf');
              return (
                <Link 
                  key={docName} 
                  href={`${apiUrl}/docs-api/documents/${encodeURIComponent(docName)}`}
                  target={isPdf ? "_blank" : "_blank"}
                  rel="noopener noreferrer"
                  className="card bg-base-100 shadow-xl hover:shadow-2xl transition-shadow duration-300 ease-in-out transform hover:-translate-y-1 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-50"
                >
                  <div className="card-body items-center text-center p-6 min-h-[180px] flex flex-col justify-between">
                    <div className="flex flex-col items-center flex-grow justify-center">
                      {isPdf ? (
                        <FileText size={36} className="mb-4 opacity-70 text-red-500"/>
                      ) : (
                        <HardDrive size={36} className="mb-4 opacity-70"/>
                      )}
                      <h2 className="card-title text-sm leading-tight text-center break-words hyphens-auto w-full" title={docName}>
                        {docName}
                      </h2>
                    </div>
                    <p className="text-xs text-base-content/70 mt-3">
                      {isPdf ? "Click to view" : "Click to download"}
                    </p>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
