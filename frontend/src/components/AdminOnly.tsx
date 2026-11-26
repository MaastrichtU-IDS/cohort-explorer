'use client';

import React, {useEffect, useState} from 'react';
import {apiUrl} from '@/utils';
import LoginPrompt from '@/components/LoginPrompt';

// Renders its children only for an administrator. Everyone else gets the sign-in
// prompt (no session, or an expired or invalid one: the backend answers 401 or
// 403 for those) or a short refusal (signed in, not an admin). A failed request
// says so rather than claiming the user is not an admin.
//
// The check needs only the session cookie, so it runs at once rather than
// waiting for the cohorts payload. It gates the page, not the data: the
// endpoints a page calls apply their own rules.
export default function AdminOnly({children}: {children: React.ReactNode}) {
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [needsLogin, setNeedsLogin] = useState(false);
  const [checkFailed, setCheckFailed] = useState(false);

  useEffect(() => {
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(async r => {
        if (r.status === 401 || r.status === 403) {
          setNeedsLogin(true);
          return;
        }
        if (!r.ok) {
          setCheckFailed(true);
          return;
        }
        const d = await r.json();
        setIsAdmin(!!d.is_admin);
      })
      .catch(() => setCheckFailed(true));
  }, []);

  if (needsLogin) return <LoginPrompt message="Authenticate to access this page (administrators only)" />;
  if (checkFailed) {
    return (
      <div className="container mx-auto p-6">
        <div className="alert alert-error">Could not check your access to this page. Reload to try again.</div>
      </div>
    );
  }
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
