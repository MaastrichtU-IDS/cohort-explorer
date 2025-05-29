 'use client';

import React, { useState, useEffect } from 'react';
import { useCohorts } from '@/components/CohortsContext';
import { apiUrl } from '@/utils';
import Link from 'next/link';
import { HardDrive, AlertTriangle, Check, Info } from 'react-feather'; // Added more icons

export default function DocsStorePage() {
  const { userEmail } = useCohorts();
  const [documents, setDocuments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (userEmail) { // Only fetch if logged in
      const fetchDocuments = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const response = await fetch(`${apiUrl}/docs-api/documents`, {
            credentials: 'include',
          });
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
    } else {
        setIsLoading(false); // Not logged in, so not loading
        setDocuments([]);
    }
  }, [userEmail]);

  if (userEmail === null && !isLoading) { // Ensure loading state is false before showing auth message
    return (
      <main className="flex flex-col items-center justify-start p-8 min-h-screen bg-base-200">
        <div className="w-full max-w-4xl space-y-8 text-center mt-[20%]">
            <h1 className="text-3xl font-bold mb-8 flex items-center justify-center">
              <HardDrive size={32} className="mr-3" /> Project Documents
            </h1>
            <p className="text-error">
                Please authenticate to access the project documents.
            </p>
        </div>
      </main>
    );
  }

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
            <span>Error loading documents: {error}. <br/>Please ensure the backend is running, the API is reachable, and the 'documents_store' folder exists at the project root.</span>
          </div>
        )}

        {!isLoading && !error && documents.length === 0 && (
          <div className="text-center text-slate-500 p-10 border-2 border-dashed border-base-300 rounded-lg mt-[5%]">
            <Info size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-xl">No documents found.</p>
            <p className="text-sm mt-2">Upload files to the 'documents_store' directory in the project root on the server.</p>
          </div>
        )}

        {!isLoading && !error && documents.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-6">
            {documents.map((docName) => (
              <Link 
                key={docName} 
                href={`${apiUrl}/docs-api/documents/${encodeURIComponent(docName)}`}
                target="_blank" 
                rel="noopener noreferrer"
                className="card bg-base-100 shadow-xl hover:shadow-2xl transition-shadow duration-300 ease-in-out transform hover:-translate-y-1 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-50"
              >
                <div className="card-body items-center text-center p-4">
                  <HardDrive size={28} className="mb-2 opacity-70"/>
                  <h2 className="card-title truncate w-full text-base leading-tight" title={docName}>{docName}</h2>
                  <p className="text-xs text-base-content/70 mt-1">Click to download</p>
                </div>
              </Link>
            ))}\
          </div>
        )}
      </div>
    </main>
  );
}