import {describe, expect, it} from 'vitest';
import {resolveServerApiUrl} from '@/utils/apiUrls';

describe('server API URL resolution', () => {
  it('uses the container-internal backend URL for server-side EDA proxying', () => {
    expect(
      resolveServerApiUrl({
        INTERNAL_API_URL: 'http://backend:80',
        NEXT_PUBLIC_API_URL: 'http://localhost:3000'
      })
    ).toBe('http://backend:80');
  });

  it('falls back to the browser URL for a local process, then the legacy default', () => {
    expect(resolveServerApiUrl({NEXT_PUBLIC_API_URL: 'http://localhost:3000'})).toBe('http://localhost:3000');
    expect(resolveServerApiUrl({})).toBe('http://localhost:8000');
  });
});
