'use client';

import React, {useEffect, useState} from 'react';
import {apiUrl} from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';

// Renders its children only for an administrator. Everyone else gets the sign-in
// prompt (no session) or a short refusal (signed in, not an admin).
//
// The check needs only the session cookie, so it runs at once rather than
// waiting for the cohorts payload. It gates the page, not the data: the
// endpoints a page calls apply their own rules.
export default function AdminOnly({children}: {children: React.ReactNode}) {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);

  useEffect(() => {
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(r => {
        if (r.status === 401) {
          setNeedsLogin(true);
          return {is_admin: false};
        }
        return r.ok ? r.json() : {is_admin: false};
      })
      .then(d => setIsAdmin(!!d.is_admin))
      .catch(() => setIsAdmin(false));
  }, []);

  if (needsLogin) return <LoginPrompt message="Authenticate to access this page (administrators only)" />;
  if (isAdmin === false) {
    return (
      <div className="container mx-auto p-6">
        <div className="alert alert-warning">This page is for administrators only.</div>
      </div>
    );
  }
  if (isAdmin === null) {
    return (
      <div className="container mx-auto p-6 flex items-center gap-3 text-base-content/70">
        <span className="loading loading-spinner loading-md" /> <span>Loading…</span>
      </div>
    );
  }
  return <>{children}</>;
}
