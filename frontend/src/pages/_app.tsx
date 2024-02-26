import '@/styles/globals.css';
import type {AppProps} from 'next/app';
import React, {createContext, useState, useEffect, useContext} from 'react';
import Head from 'next/head'
import {Nav} from '@/components/Nav';
import {CohortsProvider} from '@/components/CohortsContext';

export default function App({Component, pageProps}: AppProps) {
  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/icare4cvd_heart_logo.png" />
        <meta name="description" content="Explore cohorts for the iCare4CVD project" />
        <title>Cohort Explorer</title>
      </Head>
      <CohortsProvider>
        <Nav />
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
