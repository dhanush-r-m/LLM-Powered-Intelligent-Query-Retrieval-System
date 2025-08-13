/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      boxShadow: {
        glass: '0 8px 32px rgba(31, 38, 135, 0.08)',
      },
      colors: {
        primary: {
          500: '#6366f1',
          600: '#4f46e5',
        }
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}

