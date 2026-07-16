import path from 'node:path';
import {configDefaults, defineConfig} from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  test: {
    environment: 'node',
    exclude: [...configDefaults.exclude, 'e2e/**']
  }
});
