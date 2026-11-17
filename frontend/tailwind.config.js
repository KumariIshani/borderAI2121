/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: '#0A0D10',
        surface: '#0F1318',
        card: {
          DEFAULT: '#11161B',
          hover: '#151C22',
          border: 'rgba(255, 255, 255, 0.08)',
          active: 'rgba(249, 115, 22, 0.15)',
        },
        primary: {
          DEFAULT: '#F97316',
          hover: '#EA580C',
          light: '#FB923C',
        },
        accent: {
          cyan: '#22D3EE',
          green: '#22C55E',
          red: '#EF4444',
          yellow: '#EAB308',
        },
        text: {
          primary: '#FFFFFF',
          body: '#E5E7EB',
          secondary: '#9CA3AF',
          muted: '#6B7280',
        }
      },
      fontFamily: {
        sans: ['"Space Grotesk"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      borderRadius: {
        'card': '12px',
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'glow-primary': '0 0 20px rgba(249, 115, 22, 0.25)',
        'glow-cyan': '0 0 20px rgba(34, 211, 238, 0.25)',
      }
    },
  },
  plugins: [],
}

