 'use client';

import React, { useState, useEffect } from 'react';
import { apiUrl } from '@/utils';
import Link from 'next/link';
import { HardDrive, AlertTriangle, Check, Info, FileText, Video } from 'react-feather'; // Added more icons

interface VideoLink {
  title: string;
  url: string;
}

export default function DocsStorePage() {
  const [documents, setDocuments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'providers' | 'analysts'>('analysts');
  const [videoLinks, setVideoLinks] = useState<VideoLink[]>([]);

  // Filter documents based on view mode
  // 'providers' view: documents WITHOUT 'user' in the name
  // 'analysts' view: documents WITH 'user' in the name
  // Always exclude video_links.txt from display
  const filteredDocuments = documents.filter((docName) => {
    if (docName.toLowerCase() === 'video_links.txt') return false;
    const hasUser = docName.toLowerCase().includes('user');
    return viewMode === 'analysts' ? hasUser : !hasUser;
  });

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

  // Fetch and parse video_links.txt
  useEffect(() => {
    const fetchVideoLinks = async () => {
      try {
        const response = await fetch(`${apiUrl}/docs-api/documents/video_links.txt`);
        if (!response.ok) return;
        const text = await response.text();
        // Parse the file: items separated by blank lines, each item has title on first line, URL on second
        const items = text.split(/\n\s*\n/).filter(item => item.trim());
        const parsed: VideoLink[] = items.map(item => {
          const lines = item.trim().split('\n');
          return {
            title: lines[0]?.trim() || '',
            url: lines[1]?.trim() || ''
          };
        }).filter(v => v.title && v.url);
        setVideoLinks(parsed);
      } catch (err) {
        console.error('Error fetching video links:', err);
      }
    };
    fetchVideoLinks();
  }, []);

  return (
    <main className="flex flex-col items-center justify-start p-8 min-h-screen bg-base-200">
      <div className="w-full max-w-4xl space-y-8">
        <h1 className="text-3xl font-bold text-center mb-4 flex items-center justify-center">
          <HardDrive size={32} className="mr-3" /> Project Documents
        </h1>

        {/* View Toggle Buttons */}
        <div className="flex justify-center mb-6">
          <div className="join">
            <button
              className={`join-item btn ${viewMode === 'providers' ? '' : 'btn-ghost'}`}
              style={viewMode === 'providers' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
              onClick={() => setViewMode('providers')}
            >
              For Data Providers
            </button>
            <button
              className={`join-item btn ${viewMode === 'analysts' ? '' : 'btn-ghost'}`}
              style={viewMode === 'analysts' ? { backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' } : {}}
              onClick={() => setViewMode('analysts')}
            >
              For Analysts
            </button>
          </div>
        </div>

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

        {!isLoading && !error && documents.length > 0 && filteredDocuments.length === 0 && (
          <div className="text-center text-slate-500 p-10 border-2 border-dashed border-base-300 rounded-lg mt-[5%]">
            <Info size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-xl">No documents in this category.</p>
            <p className="text-sm mt-2">Try switching to the other view.</p>
          </div>
        )}

        {!isLoading && !error && filteredDocuments.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6 mt-6" style={{gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))'}}>
            {filteredDocuments.map((docName) => {
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
                      <h2 className="card-title text-sm leading-tight text-center w-full" style={{overflowWrap: 'break-word', wordBreak: 'keep-all'}} title={docName}>
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

        {/* Feedback Form Button - For Data Providers view (shown here) */}
        {viewMode === 'providers' && (
          <div className="mt-32 mb-16 flex justify-center">
            <a
              href="https://docs.google.com/forms/d/e/1FAIpQLSd7EmQJgfNJJej8cuKN_eOv5ROYcjVVE-aM_sruNW6P0wySOQ/viewform?hl=en%2Fedit&hl=en"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-lg"
              style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' }}
            >
              Problems? Suggestions? Please fill out our feedback form!
            </a>
          </div>
        )}

        {/* Video Tutorials Section - Only shown in Analysts view */}
        {viewMode === 'analysts' && videoLinks.length > 0 && (
          <div className="mt-12 pt-8 border-t border-base-300">
            <h2 className="text-2xl font-bold text-center mb-2 flex items-center justify-center">
              <Video size={28} className="mr-3" /> Video Tutorials
            </h2>
            <p className="text-center text-sm text-base-content/60 mb-6">
              (requires access to the SharePoint folder)
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {videoLinks.map((video, index) => (
                <a
                  key={index}
                  href={video.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="card bg-base-100 shadow-md hover:shadow-lg transition-shadow duration-200 p-4 flex flex-row items-center gap-3"
                >
                  <Video size={24} className="text-primary shrink-0" />
                  <span className="text-sm font-medium hover:text-primary transition-colors" style={{overflowWrap: 'break-word', wordBreak: 'keep-all'}}>
                    {video.title}
                  </span>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Feedback Form Button - For Analysts view (shown below video tutorials) */}
        {viewMode === 'analysts' && (
          <div className="mt-32 mb-16 flex justify-center">
            <a
              href="https://docs.google.com/forms/d/e/1FAIpQLSd7EmQJgfNJJej8cuKN_eOv5ROYcjVVE-aM_sruNW6P0wySOQ/viewform?hl=en%2Fedit&hl=en"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-lg"
              style={{ backgroundColor: '#dbeafe', color: '#1e3a8a', border: '1px solid #bfdbfe' }}
            >
              Problems? Suggestions? Please fill out our feedback form!
            </a>
          </div>
        )}
      </div>
    </main>
  );
}
