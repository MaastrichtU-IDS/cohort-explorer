export interface ServerApiEnvironment {
  INTERNAL_API_URL?: string;
  NEXT_PUBLIC_API_URL?: string;
}

export const resolveServerApiUrl = (environment: ServerApiEnvironment): string =>
  environment.INTERNAL_API_URL || environment.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const resolveMappingApiUrl = (baseUrl: string, path: string): string =>
  `${baseUrl.replace(/\/$/, '')}/api/${path.replace(/^\//, '')}`;
