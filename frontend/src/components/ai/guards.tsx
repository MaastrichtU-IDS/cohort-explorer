'use client';

// Access guards for the iCARE-AI pages:
// - The main interface (/ai) requires a logged-in user.
// - The alternative interfaces (/ai/alternatives and the individual layouts)
//   are for evaluation only and additionally require admin rights, checked via
//   the same /admin/check endpoint the admin settings page uses.
import React, {ComponentType, useEffect, useState} from 'react';
import {AlertTriangle, Shield} from 'react-feather';
import {useCohorts} from '@/components/CohortsContext';
import {apiUrl} from '@/utils';

export function AiAccessGuard({
  requireAdmin = false,
  children
}: {
  requireAdmin?: boolean;
  children: React.ReactNode;
}) {
  const {userEmail} = useCohorts();
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  useEffect(() => {
    if (!requireAdmin || !userEmail) return;
    let cancelled = false;
    fetch(`${apiUrl}/admin/check`, {credentials: 'include'})
      .then(res => (res.ok ? res.json() : {is_admin: false}))
      .then(data => {
        if (!cancelled) setIsAdmin(!!data.is_admin);
      })
      .catch(() => {
        if (!cancelled) setIsAdmin(false);
      });
    return () => {
      cancelled = true;
    };
  }, [requireAdmin, userEmail]);

  if (!userEmail) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="alert alert-warning max-w-md">
          <AlertTriangle size={20} />
          <span>
            Please{' '}
            <a className="link" href={`${apiUrl}/login`}>
              log in
            </a>{' '}
            to use iCARE-AI.
          </span>
        </div>
      </div>
    );
  }

  if (requireAdmin) {
    if (isAdmin === null) {
      return (
        <div className="flex justify-center items-center min-h-[60vh]">
          <span className="loading loading-spinner loading-lg"></span>
        </div>
      );
    }
    if (!isAdmin) {
      return (
        <div className="flex justify-center items-center min-h-[60vh]">
          <div className="alert alert-error max-w-md">
            <Shield size={20} />
            <span>Access denied. The alternative iCARE-AI interfaces are restricted to administrators.</span>
          </div>
        </div>
      );
    }
  }

  return <>{children}</>;
}

export function withAiAccess<P extends object>(
  Component: ComponentType<P>,
  opts: {requireAdmin?: boolean} = {}
) {
  const Wrapped = (props: P) => (
    <AiAccessGuard requireAdmin={opts.requireAdmin}>
      <Component {...props} />
    </AiAccessGuard>
  );
  Wrapped.displayName = `withAiAccess(${Component.displayName || Component.name || 'Component'})`;
  return Wrapped;
}
