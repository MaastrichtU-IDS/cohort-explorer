import Image from 'next/image';
import Link from 'next/link';
import {Inter} from 'next/font/google';
import { useCohorts } from '@/components/CohortsContext';
import { useEffect, useState } from 'react';
import { Cohort } from '@/types';

const inter = Inter({subsets: ['latin']});

export default function Home() {
  const { cohortsData } = useCohorts();
  const [stats, setStats] = useState({
    totalCohorts: 0,
    cohortsWithMetadata: 0,
    totalPatients: 0,
    totalVariables: 0
  });

  useEffect(() => {
    if (Object.keys(cohortsData).length > 0) {
      // Calculate statistics
      const totalCohorts = Object.keys(cohortsData).length;
      
      // Cohorts with metadata (has variables)
      const cohortsWithMetadata = Object.values(cohortsData as Record<string, Cohort>).filter(
        (cohort: Cohort) => Object.keys(cohort.variables || {}).length > 0
      ).length;
      
      // Total patients across all cohorts
      const totalPatients = Object.values(cohortsData as Record<string, Cohort>).reduce(
        (sum: number, cohort: Cohort) => sum + (cohort.study_participants || 0), 
        0
      );
      
      // Total unique variables across all cohorts
      const uniqueVariables = new Set<string>();
      Object.values(cohortsData as Record<string, Cohort>).forEach((cohort: Cohort) => {
        Object.keys(cohort.variables || {}).forEach((varId: string) => {
          uniqueVariables.add(varId);
        });
      });
      const totalVariables = uniqueVariables.size;
      
      setStats({
        totalCohorts,
        cohortsWithMetadata,
        totalPatients,
        totalVariables
      });
    }
  }, [cohortsData]);

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
      <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-6 w-full max-w-5xl">
        {/* Total Cohorts */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-figure text-primary">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="inline-block w-8 h-8 stroke-current">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <div className="stat-value text-primary text-3xl">{stats.totalCohorts}</div>
          <div className="stat-title text-sm">Cohorts Registered</div>
        </div>

        {/* Cohorts with Metadata */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-figure text-secondary">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="inline-block w-8 h-8 stroke-current">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path>
            </svg>
          </div>
          <div className="stat-value text-secondary text-3xl">{stats.cohortsWithMetadata}</div>
          <div className="stat-title text-sm">Cohorts with Metadata</div>
        </div>

        {/* Total Patients */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-figure text-accent">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="inline-block w-8 h-8 stroke-current">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path>
            </svg>
          </div>
          <div className="stat-value text-accent text-3xl">{stats.totalPatients.toLocaleString()}</div>
          <div className="stat-title text-sm">Total Patients</div>
        </div>

        {/* Total Variables */}
        <div className="stat bg-base-100 shadow rounded-lg p-4">
          <div className="stat-figure text-info">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" className="inline-block w-8 h-8 stroke-current">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3zm0 5h16"></path>
            </svg>
          </div>
          <div className="stat-value text-info text-3xl">{stats.totalVariables.toLocaleString()}</div>
          <div className="stat-title text-sm">Total Variables</div>
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
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>View the documentation and source code on GitHub</p>
        </Link>
      </div>
    </main>
  );
}
