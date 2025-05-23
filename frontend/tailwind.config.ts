import type {Config} from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}'
  ],
  theme: {
    extend: {
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))'
      },
      // Remove backticks from inline code
      typography: {
        DEFAULT: {
          css: {
            // Fix <code> rendering
            'code::before': {
              content: '""'
            },
            'code::after': {
              content: '""'
            },
            code: {
              'border-radius': '0.375rem',
              padding: '0.35em',
              color: 'var(--tw-prose-pre-code)',
              'background-color': 'var(--tw-prose-pre-bg)',
              'font-weight': 'normal'
            },
            // Explicitly style <strong> globally
            strong: {
              'font-weight': 'bold',
              color: 'inherit'
            },
            // Also explicitly style <strong> inside list items
            'li strong': { // More specific selector
              'font-weight': 'bold'
            },
            // Ensure paragraphs inside lists don't interfere (less likely but possible)
            'li p strong': {
                'font-weight': 'bold'
            }
          }
        }
      }
    }
  },
  plugins: [require('@tailwindcss/typography'), require('daisyui')],
  daisyui: {
    themes: ['cupcake', 'light', 'dark']
  },
  darkMode: 'class'
  // darkMode: ['class', '[data-mode="dark"]'],
};
export default config;
