import React from 'react';
import {apiUrl} from '@/utils';

// The standard "authenticate to access" message shown on pages that need a
// login. The whole message is a link: clicking it starts the login flow and
// carries the CURRENT path along, so after logging in the user lands back on
// the page they were trying to access (not always the explore page).

export function goToLogin() {
  const here = window.location.pathname + window.location.search;
  window.location.href = `${apiUrl}/login?redirect=${encodeURIComponent(here)}`;
}

export default function LoginPrompt({message = 'Authenticate to access the explorer'}: {message?: string}) {
  return (
    <p className="text-red-500 text-center mt-[20%]" role="alert">
      <a
        href={`${apiUrl}/login`}
        className="underline underline-offset-2 hover:text-red-700"
        onClick={e => {
          e.preventDefault();
          goToLogin();
        }}
      >
        {message}
      </a>
    </p>
  );
}
