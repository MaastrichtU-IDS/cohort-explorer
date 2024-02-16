import '@/styles/globals.css';
import type {AppProps} from 'next/app';
import React, {createContext, useState, useEffect, useContext} from 'react';
import {Nav} from '../components/Nav';
import {CohortsProvider} from '../components/CohortsContext';

export default function App({Component, pageProps}: AppProps) {
  return (
    <CohortsProvider>
      <Nav />
      <Component {...pageProps} />
    </CohortsProvider>
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
