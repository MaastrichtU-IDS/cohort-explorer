/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'nextjs.org',
        port: '',
        pathname: '/**',
      },
    ],
  },
  async redirects() {
    // The no-code DCR feature first shipped under "guided" URLs.
    return [
      {source: '/guided-analysis', destination: '/nocode-dcr', permanent: true},
      {source: '/guided-results', destination: '/nocode-results', permanent: true},
    ];
  },
  async headers() {
    return [
      {
        // Apply cache control headers to the main page to prevent browser caching
        source: '/',
        headers: [
          {
            key: 'Cache-Control',
            value: 'no-cache, no-store, must-revalidate, max-age=0'
          },
          {
            key: 'Pragma',
            value: 'no-cache'
          },
          {
            key: 'Expires',
            value: '0'
          }
        ],
      }
    ]
  }
};

module.exports = nextConfig;
