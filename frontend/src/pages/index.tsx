import Image from 'next/image';
import Link from 'next/link';
import {Inter} from 'next/font/google';
import { useCohorts } from '@/components/CohortsContext';
import React from 'react';
import { Cohort } from '@/types';

const inter = Inter({subsets: ['latin']});

export default function Home() {
  // Get statistics and calculation function from context
  const { cohortStatistics, calculateStatistics } = useCohorts();
  
  // State to store statistics from API
  const [stats, setStats] = React.useState<{
    totalCohorts: number | string;
    cohortsWithMetadata: number | string;
    cohortsWithAggregateAnalysis: number | string;
    totalPatients: number | string;
    patientsInCohortsWithMetadata: number | string;
    totalVariables: number | string;
  }>({
    totalCohorts: "waiting to refresh...",
    cohortsWithMetadata: "waiting to refresh...",
    cohortsWithAggregateAnalysis: "waiting to refresh...",
    totalPatients: 0,
    patientsInCohortsWithMetadata: 0,
    totalVariables: 0
  });
  
  // State to track if statistics are being recalculated
  const [isRecalculating, setIsRecalculating] = React.useState(false);
  
  // Fetch statistics from API on component mount
  React.useEffect(() => {
    const fetchStatistics = async () => {
      try {
        // Check if the page was reloaded (user hit refresh)
        const navigationEntries = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];
        const isPageRefresh = navigationEntries.length > 0 && navigationEntries[0].type === 'reload';
        
        // Only recalculate statistics if user explicitly refreshed the page
        if (isPageRefresh && calculateStatistics) {
          console.log('Page refresh detected, recalculating statistics...');
          setIsRecalculating(true);
          setStats({
            totalCohorts: "recalculating...",
            cohortsWithMetadata: "recalculating...",
            cohortsWithAggregateAnalysis: "recalculating...",
            totalPatients: "recalculating...",
            patientsInCohortsWithMetadata: "recalculating...",
            totalVariables: "recalculating..."
          });
          await calculateStatistics();
          // Wait a bit for the file to be written
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Add cache-busting parameter to prevent browser caching
        const cacheBuster = Date.now();
        const response = await fetch(`/api/get-statistics?_=${cacheBuster}`);
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        }
        setIsRecalculating(false);
      } catch (error) {
        console.error('Error fetching statistics:', error);
        setIsRecalculating(false);
        // Fallback to context statistics if API fails
        if (cohortStatistics) {
          setStats(cohortStatistics);
        }
      }
    };
    
    fetchStatistics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  return (
    <main className={`flex flex-col items-center justify-between p-24 ${inter.className}`}>
      <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700/10 after:dark:from-sky-900 after:dark:via-[#0141ff]/40 before:lg:h-[360px]">
        <Image
          className="relative dark:invert"
          src="/icare4cvd_logo.png"
          alt="iCARE4CVD Logo"
          width={180}
          height={37}
          priority
        />
      </div>

      {/* Statistics Cards */}
      <div className="mt-16 grid grid-cols-2 md:grid-cols-3 gap-6 w-full max-w-5xl">
        {/* Total Cohorts */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-primary text-3xl">{stats.totalCohorts}</div>
          <div className="stat-title text-sm">Registered Cohorts</div>
        </div>

        {/* Cohorts with Metadata */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-secondary text-3xl">{stats.cohortsWithMetadata}</div>
          <div className="stat-title text-sm">Cohorts with Uploaded Metadata</div>
        </div>
        
        {/* Cohorts with Aggregate Analysis */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-success text-3xl">{stats.cohortsWithAggregateAnalysis}</div>
          <div className="stat-title text-sm">Cohorts with Aggregate Data Added</div>
        </div>

        {/* Total Patients */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-accent text-3xl">{typeof stats.totalPatients === 'number' ? stats.totalPatients.toLocaleString() : stats.totalPatients}</div>
          <div className="stat-title text-sm">Total Patients Across All Cohorts</div>
        </div>
        
        {/* Patients in Cohorts with Metadata */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-warning text-3xl">{typeof stats.patientsInCohortsWithMetadata === 'number' ? stats.patientsInCohortsWithMetadata.toLocaleString() : stats.patientsInCohortsWithMetadata}</div>
          <div className="stat-title text-sm">Patients in Cohorts with Uploaded Metadata</div>
        </div>
        
        {/* Total Variables */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-value text-info text-3xl">{typeof stats.totalVariables === 'number' ? stats.totalVariables.toLocaleString() : stats.totalVariables}</div>
          <div className="stat-title text-sm">Variables in Cohorts with Uploaded Metadata</div>
        </div>
      </div>

      <div className="mt-32 grid text-center lg:max-w-7xl lg:w-full lg:mb-0 lg:grid-cols-4 lg:text-left">
        <Link
          href="/upload"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Upload Cohorts{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            Upload a cohort data dictionary describing its variables
          </p>
        </Link>

        <Link
          href="/cohorts"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Explore Cohorts{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            Explore and search data dictionaries of available Cohorts
          </p>
        </Link>

        <Link
          href="/docs_store"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Project Documents{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            Access user manual and other project documents
          </p>
        </Link>

        <Link
          href="https://github.com/MaastrichtU-IDS/cohort-explorer"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Technical details{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>View the source code on GitHub</p>
        </Link>
      </div>
    </main>
  );
}
