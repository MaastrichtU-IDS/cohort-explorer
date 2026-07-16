import path from 'path';

import {defineConfig, devices} from '@playwright/test';

const baseURL = process.env.DEMO_BROWSER_URL || 'http://localhost:3001';

export default defineConfig({
  testDir: './e2e',
  outputDir: path.resolve(__dirname, '../artifacts/browser-demo/test-results'),
  timeout: 900_000,
  expect: {timeout: 15_000},
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL,
    actionTimeout: 15_000,
    headless: process.env.DEMO_BROWSER_HEADED !== 'true',
    acceptDownloads: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off'
  },
  projects: [
    {
      name: 'chromium',
      use: {...devices['Desktop Chrome']}
    }
  ]
});
