import '@/styles/globals.css';
import type {AppProps} from 'next/app';
import React, {createContext, useState, useEffect, useContext} from 'react';
import Head from 'next/head';
import {Nav} from '@/components/Nav';
import {CohortsProvider} from '@/components/CohortsContext';

export default function App({Component, pageProps}: AppProps) {
  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/icare4cvd_heart_logo.png" />
        <meta name="description" content="Explore cohorts for the iCARE4CVD project" />
        <title>Cohort Explorer</title>
      </Head>
      <CohortsProvider>
        <Nav />
        {/* Banner for submitting clinical research questions */}
        <div className="w-full bg-blue-100 text-blue-900 py-3 px-4 text-center font-semibold text-base shadow-sm" role="region" aria-label="Submit Clinical Research Question" style={{borderBottom: '1px solid #b6d4fe'}}>
          <a
            href="https://forms.office.com/Pages/ResponsePage.aspx?id=ZjsnCBAizUCLIFz2zdwnmivGAWehv9FJqQkwLj2vey9UOE5CRUc4MEc5NVRJVkVGVlM5VkdXVjRCNS4u"
            target="_blank"
            rel="noopener noreferrer"
            className="underline hover:text-blue-700"
          >
            Please tell us about the clinical research you are interested in
          </a>
        </div>
        <Component {...pageProps} />
      </CohortsProvider>
    </>
  );
}

// import '@/styles/globals.css'
// import { SessionProvider } from "next-auth/react"
// import type { AppProps } from "next/app"

// export default function App({
//   Component,
//   pageProps: { session, ...pageProps },
// }: AppProps) {
//   return (
//     <SessionProvider session={session}>
//       <Component {...pageProps} />
//     </SessionProvider>
//   )
// }
